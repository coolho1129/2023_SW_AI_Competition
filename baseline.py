
import os
import cv2
import pandas as pd
import numpy as np
from typing import List, Union
from joblib import Parallel, delayed


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import multiprocessing as mp
from multiprocessing import freeze_support

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import gc
import matplotlib.pyplot as plt


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
            patches = split_image(image, self.patch_size, self.stride)
            transformed_patches = [self.transform(image=patch)["image"] for patch in patches]
            return transformed_patches

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


    # 간단한 U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    
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

# U-Net의 기본 구성 요소인 Double Convolution Block을 정의합니다.


def init():
    global MAINPATH, TRAINPATH, TESTPATH
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    MAINPATH=os.path.dirname(__file__)
    print(MAINPATH)
    os.chdir(MAINPATH)
    freeze_support()
    gc.collect()
    torch.cuda.empty_cache()
    TRAINPATH ='./train.csv'
    TESTPATH ='./test.csv'

    

def set_train_dataset(path,transform):
    dataset = SatelliteDataset(csv_file=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batsize, shuffle=True, num_workers=4)

    return dataset,dataloader

def set_test_dataset(path,transform):
    dataset = SatelliteDataset(csv_file=path, transform=transform,infer=True)
    dataloader = DataLoader(dataset, batch_size=batsize, shuffle=False, num_workers=4)

    return dataset,dataloader


def train(model,criterion,optimizer,dataloader):
    # training loop
    for epoch in range(epoches):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            for image, mask in zip(images, masks):
                image = image.float().to(device)
                mask = mask.float().to(device)

                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs, mask.unsqueeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
    
    return model

def predict(model,dataloader):
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            for image, mask in zip(images, masks):
                image = image.float().to(device)
                
                outputs = model(image)
                mask = torch.sigmoid(outputs).cpu().numpy()
                mask = np.squeeze(mask, axis=1)
                mask = (mask > 0.35).astype(np.uint8) # Threshold = 0.35

                for i in range(len(image)):
                    mask_rle = rle_encode(mask[i])
                    if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                        result.append(-1)
                    else:
                        result.append(mask_rle)
    return result


def sumbit_save(result):
    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    SUMMITDIR='submit'
    if(not os.path.exists(SUMMITDIR) or os.path.isfile(SUMMITDIR)):
        os.mkdir(SUMMITDIR)
    SUBMITPATH='./'+str(SUMMITDIR)+'/'+str(os.path.basename(__file__).split('.')[0]) +'_'+str(epoches)+'.csv'
    submit.to_csv(SUBMITPATH, index=False)


def load_model(model,path):
    model_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(model_state_dict)

    return model

def main():
    
    init()
    
    # transform 설정
    transform = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )
  
    #train dataset 설정
    train_dataset,train_dataloader=set_train_dataset(TRAINPATH,transform) 

    # model 초기화
    model = UNet().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    #epoches 및 batsize설정
    global epoches,batsize
    epoches=1
    batchsize=16
    
    #학습
    train(model,criterion,optimizer,train_dataloader)

    #test dataset 설정
    test_dataset,test_dataloader=set_test_dataset(TESTPATH,transform)

    # 예측
    result=predict(model,test_dataloader)
    
    #제출 파일 저장
    sumbit_save(result)

if __name__=='__main__':
    main()
