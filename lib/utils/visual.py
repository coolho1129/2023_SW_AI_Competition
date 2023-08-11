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

from utils.utils import *
from lib.datasets.dataset import *


def visualize_images(path,start, end):
  
    test=pd.read_csv('./test.csv')
    submit=pd.read_csv(path)
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




# #print(df)
# global PATH
# PATH = "" # submit.csv 파일의 경로를 입력하세요.
# visualize_images(PATH,70,99)


    
