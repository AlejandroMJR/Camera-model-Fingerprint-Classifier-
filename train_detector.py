"""
Main training script
binary classification: real vs synthetic
"""

# Libraries import #
from typing import List
import os
import argparse
import pandas as pd
import torch
import numpy as np
torch.manual_seed(21)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from utils import architectures, data_rgb
from utils.utils import make_train_tag
from utils.params import pkl_dir, log_dir, models_dir


# Helper functions and classes #

def get_augmentation(aug_list: List, aug_prob: float, jpeg_aug_prob: float, patch_size: int) -> List:
    transform = []
    if 'flip' in aug_list:
        transform.append(A.HorizontalFlip(p=aug_prob))
        transform.append(A.VerticalFlip(p=aug_prob))
    if 'rotate' in aug_list:
        transform.append(A.RandomRotate90(p=aug_prob))
    if 'clahe' in aug_list:
        transform.append(A.CLAHE(p=aug_prob))
    if 'blur' in aug_list:
        transform.append(A.Blur(p=aug_prob))
    if 'crop&resize' in aug_list:
        transform.append(A.RandomSizedCrop((64, patch_size - 1), height=patch_size, width=patch_size, p=aug_prob))
    if 'brightness&contrast' in aug_list:
        transform.append(A.RandomBrightnessContrast(p=aug_prob))
    if 'jitter' in aug_list:
        transform.append(A.ColorJitter(p=aug_prob))
    if 'downscale' in aug_list:
        transform.append(A.Downscale(p=aug_prob))
    if 'hsv' in aug_list:
        transform.append(A.HueSaturationValue(p=aug_prob))
    if 'resize&crop' in aug_list:
        transform.append(A.RandomScale(scale_limit=(1.1, 4), p=aug_prob))
        transform.append(A.RandomCrop(height=patch_size, width=patch_size, always_apply=True, p=1.0))
    if 'jpeg' in aug_list:
        transform.append(A.ImageCompression(quality_lower=40, quality_upper=100, p=jpeg_aug_prob))

    return transform


def save_model(net: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loss: float, val_loss: float,
               batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


def batch_forward(net: torch.nn.Module, device, criterion, data: torch.Tensor, labels: torch.Tensor) -> (
        torch.Tensor, float, int):
    if torch.cuda.is_available():
        data = data.to(device)
        labels = labels.to(device)
    out = net(data)
    loss = criterion(out, labels)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', help='Model name', type=str, default='EfficientNetB0')
    parser.add_argument('--db', help='Database name', type=str, default='BasicLoader')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--es_patience', type=int, default=50, help='Patience for stopping the training if no improvement'
                                                                    'on the validation loss is seen')
    parser.add_argument('--workers', type=int, default=cpu_count() // 2)
    parser.add_argument('--subsample', type=float, help='Fraction to subsample datasets')
    parser.add_argument('--aug', help='Augmentation to perform with probability `aug_prob`', nargs='+') #,
                        # default=['flip', 'rotate', 'clahe', 'blur', 'crop&resize', 'brightness&contrast', 'jitter',
                        #          'downscale', 'hsv', 'jpeg'])
    parser.add_argument('--aug_prob', help='Probability of augmentation', type=float, default=0.5)
    parser.add_argument('--jpeg_aug_prob', help='Probability of JPEG augmentation', type=float, default=0.8)
    parser.add_argument('--patch_size', type=int, default=128, help='P where PxP is the dimension of the patch')
    parser.add_argument('--patch_number', help='N, number of patches per image to extract', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2, help='binary classifier')
    parser.add_argument('--log_dir', type=str, help='Directory for saving the training logs',
                        default=log_dir)
    parser.add_argument('--models_dir', type=str, help='Directory for saving the models weights',
                        default=models_dir)
    parser.add_argument('--init', type=str, help='Weight initialization file')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')

    args = parser.parse_args()

    # Parse arguments
    gpu = args.gpu
    model_name = args.model
    db_name = args.db
    batch_size = args.batch_size
    lr = args.lr
    min_lr = args.min_lr
    es_patience = args.es_patience
    epochs = args.epochs
    workers = args.workers
    subsample = args.subsample
    aug_list = args.aug
    aug_prob = args.aug_prob
    jpeg_aug_prob = args.jpeg_aug_prob
    P = args.patch_size
    N = args.patch_number
    num_classes = args.num_classes
    classes = np.arange(0, num_classes)
    weights_folder = args.models_dir
    logs_folder = args.log_dir
    initial_model = args.init
    train_from_scratch = args.scratch

    # GPU configuration
    device = 'cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu'

    # Instantiate network
    network_class = getattr(architectures, model_name)
    net = network_class(n_classes=num_classes, pretrained=True).to(device)

    # Transformer and augmentation
    net_normalizer = net.get_normalizer()
    transform = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        A.pytorch.transforms.ToTensorV2(),
    ]
    aug_transform = get_augmentation(aug_list=aug_list, aug_prob=aug_prob, jpeg_aug_prob=jpeg_aug_prob, patch_size=P) \
        if aug_list else None

    # Instantiate Dataset and DataLoader
    dataset_class = getattr(data_rgb, db_name)

    train_df = pd.read_pickle(os.path.join('utils', pkl_dir, 'train_P-{}.pkl'.format(P)))
    val_df = pd.read_pickle(os.path.join('utils', pkl_dir, 'val_P-{}.pkl'.format(P)))

    # Sort by classes, take n_classes
    train_df = train_df.loc[train_df['label'].isin(classes)]
    val_df = val_df.loc[val_df['label'].isin(classes)]

    train_ds = dataset_class(db=train_df, transformer=transform, aug_transformers=aug_transform, subsample=subsample,
                             patch_size=P, patch_number=N)
    val_ds = dataset_class(db=val_df, transformer=transform, aug_transformers=aug_transform, subsample=subsample,
                           patch_size=P, patch_number=N)

    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)

    # Optimization
    optimizer = torch.optim.Adam(net.get_trainable_parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    # Checkpoint paths
    train_tag = make_train_tag(network_class, lr, aug_list, aug_prob, P, N, batch_size, num_classes)
    bestval_path = os.path.join(weights_folder, train_tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, train_tag, 'last.pth')

    os.makedirs(os.path.join(weights_folder, train_tag), exist_ok=True)

    # Load model from checkpoint
    val_loss = min_val_loss = 10
    epoch = 0
    net_state = None
    opt_state = None
    if initial_model is not None:
        # If given load initial model
        print('Loading model form: {}'.format(initial_model))
        state = torch.load(initial_model, map_location='cpu')
        net_state = state['net']
    elif not train_from_scratch and os.path.exists(last_path):
        print('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        net_state = state['net']
        opt_state = state['opt']
        epoch = state['epoch']
    if not train_from_scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']
    if net_state is not None:
        incomp_keys = net.load_state_dict(net_state, strict=False)
        print(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = lr
        optimizer.load_state_dict(opt_state)

    # Initialize Tensorboard
    logdir = os.path.join(logs_folder, train_tag)

    # Tensorboard instance
    tb = SummaryWriter(log_dir=logdir)

    # Training-validation loop
    train_tot_it = 0
    val_tot_it = 0
    es_counter = 0
    init_epoch = epoch
    for epoch in range(init_epoch, epochs):

        # Training
        net.train()
        optimizer.zero_grad()
        train_loss = train_num = 0
        for batch_data in tqdm(train_dl, desc='Training epoch {}'.format(epoch), leave=False, total=len(train_dl)):

            # Fetch data
            batch_img, batch_label = batch_data

            # to work with multiple patches per image
            batch_img = batch_img.view(-1, 3, P, P)
            batch_label = batch_label.view(-1)

            # Forward pass
            batch_loss = batch_forward(net, device, criterion, batch_img, batch_label)

            # Backpropagation
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Statistics
            batch_num = len(batch_label)
            train_num += batch_num
            train_tot_it += batch_num
            train_loss += batch_loss.item() * batch_num

            # Iteration logging
            tb.add_scalar('train/it-loss', batch_loss.item(), train_tot_it)

        print('\nTraining loss epoch {}: {:.4f}'.format(epoch, train_loss / train_num))

        # Validation
        net.eval()
        val_loss = val_num = 0
        for batch_data in tqdm(val_dl, desc='Validating epoch {}'.format(epoch), leave=False, total=len(val_dl)):
            # Fetch data
            batch_img, batch_label = batch_data

            # to work with multiple patches per image
            batch_img = batch_img.view(-1, 3, P, P)
            batch_label = batch_label.view(-1)

            with torch.no_grad():
                # Forward pass
                batch_loss = batch_forward(net, device, criterion, batch_img, batch_label)

            # Statistics
            batch_num = len(batch_label)
            val_num += batch_num
            val_tot_it += batch_num
            val_loss += batch_loss.item() * batch_num

            # Iteration logging
            tb.add_scalar('validation/it-loss', batch_loss.item(), val_tot_it)

        print('\nValidation loss epoch {}: {:.4f}'.format(epoch, val_loss / val_num))

        # Logging
        train_loss /= train_num
        val_loss /= val_num
        tb.add_scalar('train/epoch-loss', train_loss, epoch)
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        tb.add_scalar('validation/epoch-loss', val_loss, epoch)
        tb.flush()

        # Learning rate scheduling
        lr_scheduler.step(val_loss)

        # save last model
        save_model(net, optimizer, train_loss, val_loss, batch_size, epoch, last_path)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(net, optimizer, train_loss, val_loss, batch_size, epoch, bestval_path)
            es_counter = 0
        else:
            es_counter += 1

        if optimizer.param_groups[0]['lr'] <= min_lr:
            print('Reached minimum learning rate. Stopping.')
            break
        # check the early stopping: not to be confused with the condition to reduce the learning rate.
        elif es_counter == es_patience:
            print('Early stopping patience reached. Stopping.')
            break

    # Needed to flush out last events
    tb.close()

    print('Training completed! Bye!')


if __name__ == '__main__':
    main()
