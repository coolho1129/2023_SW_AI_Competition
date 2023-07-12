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
        transformed_patches = [self.transform(image=patch, mask=mask)["image"] for patch in patches]
        transformed_masks = [self.transform(image=patch, mask=mask)["mask"] for patch in patches]

        return transformed_patches, transformed_masks


def split_image(image, patch_size, stride):
    patches = []
    height, width = image.shape[:2]
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size, :]
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


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
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
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    '''
    Calculate Dice Score between two binary masks.
    '''
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)


def calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)) -> List[float]:
    '''
    Calculate Dice scores for a dataset.
    '''

    # Keep only the rows in the prediction dataframe that have matching img_ids in the ground truth dataframe
    prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]
    prediction_df.index = range(prediction_df.shape[0])

    # Extract the mask_rle columns
    pred_mask_rle = prediction_df.iloc[:, 1]
    gt_mask_rle = ground_truth_df.iloc[:, 1]

    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)

        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None

    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )

    dice_scores = [score for score in dice_scores if score is not None]  # Exclude None values

    return np.mean(dice_scores)


import matplotlib.pyplot as plt


def visualize_images(dataset: Dataset, num_images: int = 10):
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_images):
        image, mask = dataset[i]
        image = image.permute(1, 2, 0)  # Transpose the image tensor
        ax = fig.add_subplot(2, num_images // 2, i + 1)
        ax.imshow(image)
        ax.imshow(mask, alpha=0.3)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_predictions(ground_truth_df: pd.DataFrame, prediction_df: pd.DataFrame, dataset: Dataset,
                          num_images: int = 5):
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 12))
    for i in range(num_images):
        img_id = ground_truth_df['img_id'].iloc[i]
        ground_truth_mask_rle = ground_truth_df['mask_rle'].iloc[i]
        predicted_mask_rle = prediction_df[prediction_df['img_id'] == img_id]['mask_rle'].values[0]

        image, _ = dataset[i]
        image = image.permute(1, 2, 0)

        ground_truth_mask = rle_decode(ground_truth_mask_rle, image.shape[:2])
        predicted_mask = rle_decode(predicted_mask_rle, image.shape[:2])

        axes[i, 0].imshow(image)
        axes[i, 0].imshow(ground_truth_mask, alpha=0.3)
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(image)
        axes[i, 1].imshow(predicted_mask, alpha=0.3)
        axes[i, 1].set_title('Predicted')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def run():
    freeze_support()
    MAINPATH = os.path.dirname(__file__)
    print(MAINPATH)
    os.chdir(MAINPATH)

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2()
        ]
    )

    patch_size = 224  # 패치 크기
    stride = 112  # 스트라이드

    dataset_df = pd.read_csv('./train.csv')
    train_dataset_df = dataset_df.sample(frac=0.8)
    valid_dataset_df = dataset_df.drop(train_dataset_df.index)

    train_dataset_df = train_dataset_df.reset_index(drop=True)
    train_dataset_df.to_csv('./train_dataset.csv', index=False)
    valid_dataset_df = valid_dataset_df.reset_index(drop=True)
    valid_dataset_df.to_csv('./valid_dataset.csv', index=False)

    train_dataset = SatelliteDataset(csv_file='./train_dataset.csv', patch_size=patch_size, stride=stride,
                                     transform=transform)
    valid_dataset = SatelliteDataset(csv_file='./valid_dataset.csv', patch_size=patch_size, stride=stride,
                                     transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

    valid_ground_truth_df = valid_dataset_df[['img_id', 'mask_rle']]

    # 모델 초기화
    model = UNet().to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader)}')

    # 검증 루프
    with torch.no_grad():
        model.eval()
        val_result = []
        for images in tqdm(valid_dataloader):
            images = images.float().to(device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)  # Threshold = 0.35

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':
                    val_result.append(-1)
                else:
                    val_result.append(mask_rle)

        valid_prediction_df = pd.DataFrame({'img_id': list(valid_dataset_df['img_id'].values), 'mask_rle': val_result})

        dice_scores = calculate_dice_scores(valid_prediction_df, valid_ground_truth_df)
        mean_dice_score = np.mean(dice_scores)

        print("Mean Dice Score:", mean_dice_score)

    visualize_predictions(valid_ground_truth_df, valid_prediction_df, valid_dataset, num_images=5)


if __name__ == '__main__':
    run()
