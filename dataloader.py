import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.transforms.functional import crop
from model import DnCNN
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
gpu = 0
device = 'cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu'

class ModelClassifierDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.img_labels = csv_file
        self.transform = transform
        label_encoder = LabelEncoder()
        label_encoder.fit(self.img_labels.model.unique())
        self.labels = label_encoder.transform(self.img_labels['model'])

        self.minHeight = 99999
        self.minWidth = 99999
        for i in range(len(self.img_labels)):
            if self.img_labels.loc[i, "height"] < self.minHeight:
                self.minHeight = self.img_labels.loc[i, "height"]
        for i in range(len(self.img_labels)):
            if self.img_labels.loc[i, "width"] < self.minWidth:
                self.minWidth = self.img_labels.loc[i, "width"]

        self.model = DnCNN()
        self.model.load_state_dict(
            torch.load("bestval.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        self.model.to(device)
        self.model.eval()

        print("check")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.loc[self.img_labels["id"] == idx, "probe"].values[0]
        image = Image.open(img_path).convert("L")
        image = np.array(image) / 255.
        image = np.expand_dims(image, axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = crop(image, 0, 0, self.minHeight, self.minWidth)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        with torch.no_grad():
            image = self.model(image.to(device))
        image = image.repeat(3, 1, 1)
        image = image.to("cpu")
        return image, label



if __name__ == '__main__':

    train_df = pd.read_csv("train_classifier.csv")
    val_df = pd.read_csv("val_classifier.csv")

    train_ds = ModelClassifierDataset(csv_file=train_df)
    val_ds = ModelClassifierDataset(csv_file=val_df)

    train_dl = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=32,  shuffle=True)

    for batch_data in tqdm(train_dl, desc='Training epoch {}'.format(0), leave=False, total=len(train_dl)):
        # Fetch data
        batch_img, batch_label = batch_data
        print("end")

