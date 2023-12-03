import os
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BasicLoader(Dataset):
    def __init__(self, db: pd.DataFrame, subsample: float = None, transformer: A.BasicTransform = ToTensorV2(),
                 aug_transformers: List[A.BasicTransform] = None, patch_size: int = 128, patch_number: int = 5):
        super(BasicLoader, self).__init__()

        if subsample is not None:
            db = db.sample(frac=subsample)
        self.db = db
        self.transformer = transformer
        self.aug_transformers = aug_transformers
        self.patch_size = patch_size
        self.patch_number = patch_number

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        # note: we are considering progressive indexing of db, not actual index
        record = self.db.iloc[idx]
        cropper = A.RandomCrop(width=self.patch_size, height=self.patch_size, always_apply=True, p=1.)

        # open the image in gray scale and convert it in 3 equal RGB channels
        img = np.asarray(Image.open(record.image).convert('L'))
        img2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img2[:, :, 0] = img
        img2[:, :, 1] = img
        img2[:, :, 2] = img

        # caso piu semplice, con N = 1

        # # extract patches from image
        # img_crop = cropper(image=img2)['image']
        #
        # if self.aug_transformers is None:
        #     trans = A.Compose(self.transformer)
        # else:
        #     trans = A.Compose(self.aug_transformers + self.transformer)
        #
        # trans_img = trans(image=img_crop)['image']
        # target = torch.tensor(record.label)

        # extract patches from image
        patches = [cropper(image=img2)['image'] for x in range(self.patch_number)]

        if self.aug_transformers is None:
            trans = A.Compose(self.transformer)
        else:
            trans = A.Compose(self.aug_transformers + self.transformer)

        transf_patch_list = []
        patch_label_list = []
        for patch in patches:
            transf_patch_list.append(trans(image=patch)['image'])
            patch_label_list.append(record.label)

        # create batch:
        trans_img = torch.stack(transf_patch_list, dim=0)
        target = torch.tensor(patch_label_list)

        return trans_img, target
