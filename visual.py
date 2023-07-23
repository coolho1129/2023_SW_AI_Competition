import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, patch_size=224, stride=112, transform=None, infer=False):
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


def split_image(image, patch_size, stride):
    patches = []
    height, width = image.shape[:2]
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
    return patches

def split_mask(mask, patch_size, stride):
    patches = []
    height, width = mask.shape[:2]
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = mask[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def visualize_images(start, end):
      # transform 설정
    transform = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )
    test=pd.read_csv('./test.csv')
    submit=pd.read_csv(PATH)
    test['mask_rle']=submit['mask_rle']

    test.to_csv('./viusal_test.csv',index=False)

    dataset = SatelliteDataset(csv_file='./viusal_test.csv',transform=transform)
    for x in range(start, end+1, 10):
        fig = plt.figure(figsize=(12, 6))
        for i in range(x, x+10):
            images, masks = dataset[i]
            for image,mask in zip(images,masks):
                image = image.permute(1, 2, 0)  # Transpose the image tensor
                ax = fig.add_subplot(2, 10 // 2, i%10+1)
                ax.imshow(image)
                ax.imshow(mask, alpha=0.7)
                ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(10)
        plt.close()




#print(df)
global PATH
PATH = "" # submit.csv 파일의 경로를 입력하세요.
visualize_images(70,99)


    
