import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import multiprocessing as mp
from multiprocessing import freeze_support

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import gc

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    # 간단한 U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 64)

        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 128)

        self.dconv_down5 = double_conv(128, 256)
        self.dconv_down6 = double_conv(256, 256)

        self.dconv_down7 = double_conv(256, 512)    
        self.dconv_down8 = double_conv(512, 512) 

        self.dconv_down9 = double_conv(512, 1024)
        self.dconv_down10 = double_conv(1024, 1024)    

        self.maxpool = nn.MaxPool2d(2)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2, padding = 0)       

        self.dconv_up1 = double_conv(1024, 512)
        self.dconv_up2 = double_conv(512, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2, padding = 0)  

        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up4 = double_conv(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2, padding = 0)

        self.dconv_up5 = double_conv(256, 128)
        self.dconv_up6 = double_conv(128, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2, padding = 0    )

        self.dconv_up7 = double_conv(128, 64)
        self.dconv_up8 = double_conv(64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        conv4 = self.dconv_down4(conv3)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        conv6 = self.dconv_down6(conv5)
        x = self.maxpool(conv6)

        conv7 = self.dconv_down7(x)
        conv8 = self.dconv_down8(conv7)
        x = self.maxpool(conv8)

        conv9 = self.dconv_down9(x)     
        conv10 = self.dconv_down10(conv9)

        x = self.upconv1(conv10)        
        x = torch.cat([x, conv8], dim=1)
 
        x = self.dconv_up1(x)
        x = self.dconv_up2(x)   
        x = self.upconv2(x)      
        x = torch.cat([x, conv6], dim=1)       

        x = self.dconv_up3(x)
        x = self.dconv_up4(x) 
        x = self.upconv3(x)       
        x = torch.cat([x, conv4], dim=1)   

        x = self.dconv_up5(x)
        x = self.dconv_up6(x)    
        x = self.upconv4(x)    
        x = torch.cat([x, conv2], dim=1)  

        x = self.dconv_up7(x)
        x = self.dconv_up8(x)  

        out = self.conv_last(x)

        return out

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
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride = 1, bias = True),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )   

def run():
    freeze_support()

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    
    # model 초기화
    model = UNet().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(50):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

    test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            
            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result

    submit.to_csv('./submit.csv', index=False)

if __name__=='__main__':
    run()
