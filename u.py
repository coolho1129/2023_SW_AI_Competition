import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
#import torchvision.transforms as A
import torch.nn.functional as F

import multiprocessing as mp
from multiprocessing import freeze_support

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import gc

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
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = F.relu(self.bn(self.conv2(out)))
        return out

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)

        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(1024, 512)

        self.upconv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(512, 256)

        self.upconv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 128)

        self.upconv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(128, 64)

        if self.deep_supervision:
            self.outconv1 = nn.Conv2d(64, out_channels, kernel_size=1)
            self.outconv2 = nn.Conv2d(128, out_channels, kernel_size=1)
            self.outconv3 = nn.Conv2d(256, out_channels, kernel_size=1)
            self.outconv4 = nn.Conv2d(512, out_channels, kernel_size=1)

        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2, stride=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2, stride=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2, stride=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2, stride=2)

        conv5 = self.conv5(pool4)

        upconv6 = self.upconv6(conv5)
        concat6 = torch.cat([upconv6, conv4], dim=1)
        conv6 = self.conv6(concat6)

        upconv7 = self.upconv7(conv6)
        concat7 = torch.cat([upconv7, conv3], dim=1)
        conv7 = self.conv7(concat7)

        upconv8 = self.upconv8(conv7)
        concat8 = torch.cat([upconv8, conv2], dim=1)
        conv8 = self.conv8(concat8)

        upconv9 = self.upconv9(conv8)
        concat9 = torch.cat([upconv9, conv1], dim=1)
        conv9 = self.conv9(concat9)

        if self.deep_supervision:
            out1 = self.outconv1(conv9)
            out2 = self.outconv2(conv8)
            out3 = self.outconv3(conv7)
            out4 = self.outconv4(conv6)

            return [self.outconv(conv9), out4, out3, out2, out1]
        else:
            output = self.outconv(conv9)
            output = torch.sigmoid(output)

            return output
        


def run():
    freeze_support()

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose(
        [   
            A.SmallestMaxSize(max_size=160),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=128, width=128),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    
    # model 초기화
    input_channels = 3
    output_channels = 1
    model = UNetPlusPlus(input_channels,output_channels).to(device)

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
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

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