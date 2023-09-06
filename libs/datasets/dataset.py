import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from libs.utils.utils import *

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, patch_size=None, stride=None, transform=None, infer=False, splits=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.splits = splits
        if splits:
            self.patch_size = patch_size
            self.stride = stride
        

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
        
        if self.splits:
            patches = split_image(image, self.patch_size, self.stride)
            mask_patches = split_mask(mask, self.patch_size, self.stride)
            transformed_patches = [self.transform(image=patch, mask=mask)["image"] for patch,mask in zip(patches, mask_patches)]
            transformed_masks = [self.transform(image=patch, mask=mask)["mask"] for patch,mask in zip(patches, mask_patches)]

            return transformed_patches, transformed_masks
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image =augmented['image']
            mask = augmented['mask']
            
            # 모델에 입력할 이미지와 마스크의 크기 조정
            image = image.unsqueeze(0)   # 배치 차원 추가
            mask = mask.unsqueeze(0)     # 배치 차원 추가

        return image, mask
        
        
         



