import torch
from torch.utils.data import DataLoader
from utils import architectures
import pandas as pd
from dataloader import ModelClassifierDataset
from torch.multiprocessing import cpu_count
import torch.multiprocessing as mp

gpu = 0
device = 'cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu'
model_name = "ResNet18"
num_classes = 64
batch_size = 32
workers = cpu_count() // 2


def main():
    # Instantiate network
    network_class = getattr(architectures, model_name)
    net = network_class(n_classes=num_classes, pretrained=True).to(device)

    weights_path = 'weights_rgb/net-ResNet18_lr-0.001_aug-None_aug_p-0_batch_size-32_num_classes-64/last.pth'

    net.load_state_dict(torch.load(weights_path)["net"])

    net.eval()

    correct = 0
    total = 0

    test_df = pd.read_csv("test_classifier.csv")
    test_ds = ModelClassifierDataset(csv_file = test_df)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for images, labels in test_dl:
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    # Calculate accuracy
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

