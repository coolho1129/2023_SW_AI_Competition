import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from lib.utils.utils import *

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, patch_size, stride, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        patches = split_image(image, self.patch_size, self.stride)
        mask_patches = split_mask(mask, self.patch_size, self.stride)
        transformed_patches = [self.transform(image=patch, mask=mask)["image"] for patch,mask in zip(patches, mask_patches)]
        transformed_masks = [self.transform(image=patch, mask=mask)["mask"] for patch,mask in zip(patches, mask_patches)]

        return transformed_patches, transformed_masks



