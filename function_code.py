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
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def init():
    global MAINPATH, TRAINPATH, TESTPATH, VALIDPATH
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
    VALIDPATH ='./valid/valid.csv'

    

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
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
    

    return model

def predict(model,dataloader):
    
    # test
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(dataloader):
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
    return result


def sumbit_save(result):
    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    SUMMITDIR='submit'
    if(not os.path.exists(SUMMITDIR) or os.path.isfile(SUMMITDIR)):
        os.mkdir(SUMMITDIR)
    SUBMITPATH='./'+str(SUMMITDIR)+'/'+str(os.path.basename(__file__).split('.')[0]) +'_'+str(epoches)+'.csv'
    submit.to_csv(SUBMITPATH, index=False)

def split_valid_dataset(set_frac):
    dataset_df = pd.read_csv(TRAINPATH)
    global VALID_TRAIN,VALID_TEST,VALID_GROUND
    VALID_TRAIN='./valid/train_dataset.csv'
    VALID_TEST='./valid/valid_test_dataset.csv'
    VALID_GROUND='./valid/valid_ground_truth.csv'
    
    VALIDDIR='valid'
    if(not os.path.exists(VALIDDIR) or os.path.isfile(VALIDDIR)):
        os.mkdir(VALIDDIR)

    # train, validation dataset을 set_frac:1-set_frac 비율로 나눔
    train_dataset_df = dataset_df.sample(frac=set_frac)
    valid_dataset_df = dataset_df.drop(train_dataset_df.index)
    
    # train_dataset과 valid_dataset을 csv파일로 저장
    train_dataset_df.to_csv(VAILD_TRAIN, index=False)
    valid_dataset_df.to_csv(VALIDPATH,index=False)
    
    #dice score 계산에 사용될 valid_groud_truth df 설정 후 csv 파일로 저장
    valid_test_ground_truth_df = valid_dataset_df[['img_id', 'mask_rle']]
    valid_test_ground_truth_df.to_csv(VALID_GROUND,index=False)
    
    #예측(평가)에 이용될 valid_test_dataset_df 설정 후 csv 파일로 저장
    valid_test_dataset_df = valid_dataset_df[['img_id', 'img_path']]
    valid_test_dataset_df.to_csv(VALID_TEST, index=False)
    


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


def validation(model,dataloader):
    
        #validataset에 대해 예측
        val_result=predict(model,dataloader)
        valid_dataset_df=pd.read_csv(VALIDPATH)
        
        # val_result 를 dataframe 형태로 저장
        valid_prediction_df = pd.DataFrame({'img_id': list(valid_dataset_df['img_id'].values), 'mask_rle': val_result})
        valid_prediction_df.to_csv(".valid/valid_prediction.csv",index=False)
        
        #Calculate Dice Score
        valid_test_ground_truth_df=pd.read_csv(VALID_GROUND)
        valid_prediction_df=pd.read_csv(".valid/valid_prediction.csv")
        dice_scores = calculate_dice_scores(valid_prediction_df, valid_test_ground_truth_df)
        mean_dice_score = np.mean(dice_scores)
        
        print("Mean Dice Score:", mean_dice_score)
    
        

def visualize_images(dataset: Dataset, num_images: int = 10):
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_images):
        image, mask = dataset[i]
        image = image.permute(1, 2, 0)  # Transpose the image tensor
        ax = fig.add_subplot(2, num_images // 2, i + 1)
        ax.imshow(image)
        ax.imshow(mask, alpha=0.7)
        ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    
def visualize_comparison(dataset: Dataset, prediction_df: pd.DataFrame, ground_truth_dataset: Dataset, num_images: int = 5):
    fig = plt.figure(figsize=(12, 12))
    for i in range(num_images):
        img_id = prediction_df['img_id'][i]
        mask_pred_rle = prediction_df['mask_rle'][i]

        image_pred = dataset[i]
        image_pred = image_pred.permute(1, 2, 0).numpy()  # 예측된 이미지 텐서 변환

        mask_pred = rle_decode(mask_pred_rle, image_pred.shape[:2])

        image_gt, mask_gt = ground_truth_dataset[i]
        image_gt = image_gt.permute(1, 2, 0).numpy()  # 실제 이미지 텐서 변환

        ax1 = fig.add_subplot(num_images, 2, i * 2 + 1)
        ax1.imshow(image_gt)
        ax1.imshow(mask_gt, alpha=0.7)
        ax1.set_title(f'ground- {img_id}')
        ax1.axis('off')

        ax2 = fig.add_subplot(num_images, 2, i * 2 + 2)
        ax2.imshow(image_pred)
        ax2.imshow(mask_pred, alpha=0.7)
        ax2.set_title(f'predict- {img_id}')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

def load_model(model,path):
    model_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(model_state_dict)

    return model

def main():
    
    init()
    
    # transform 설정
    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )
  

    #train dataset 설정
    # valid없이 학습 시
    train_dataset,train_dataloader=set_train_dataset(TRAINPATH,transform) 
    
    # valid있을때 
    #valid dataset 분할
    # split_valid_dataset(0.8)
    # train_dataset,train_dataloader=set_train_dataset(VALID_TRAIN,transform)


    # model 초기화
    model = UNet().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    #epoches 및 batsize설정
    global epoches,batsize
    epoches=1
    batsize=16
    
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
