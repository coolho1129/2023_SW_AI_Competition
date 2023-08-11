import os
import cv2
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from multiprocessing import freeze_support

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import gc

from lib.models.hrnet import *
from lib.models.unet import *
from lib.models.deepunet import *
from lib.models.deeplabV3plus import *
from lib.models.deeplabV3plus_Xception import *

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
    dataset = SatelliteDataset(csv_file=path, transform=transform, patch_size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)

    return dataset,dataloader

def set_test_dataset(path,transform):
    dataset = SatelliteDataset(csv_file=path, transform=transform, infer=True, patch_size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=4)

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

        save_model(model, "./stride56_divided_unet_", epoch)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
    
    return model

def predict(model,test_dataloader):
    with torch.no_grad():
        model.eval()
        result = []
        for image in tqdm(test_dataloader):
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

def ensemble(models, test_dataloader):
    with torch.no_grad():
        results = []
        for model in models:
            model.eval()
            result = []
            for image in tqdm(test_dataloader):
                image = image.float().to(device)
                outputs = model(image)
                mask = torch.sigmoid(outputs).cpu().numpy()
                mask = np.squeeze(mask, axis=1)
                result.append(mask)

            results.append(result)

        # 각 모델의 mask들을 평균하여 최종 mask를 생성
        final_mask = np.mean(results, axis=0)

        # Threshold를 적용하여 최종 mask 생성
        final_mask = (final_mask > 0.35).astype(np.uint8)

        # 각 이미지에 대해 RLE 인코딩을 수행하여 최종 결과 생성
        final_result = []
        for masks in final_mask:
            for mask in masks:
                mask_rle = rle_encode(mask)
                if mask_rle == '':
                    final_result.append(-1)
                else:
                    final_result.append(mask_rle)
    return final_result



def sumbit_save(result,name=""):
    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    SUMMITDIR='submit'
    if(not os.path.exists(SUMMITDIR) or os.path.isfile(SUMMITDIR)):
        os.mkdir(SUMMITDIR)
    if(name==""):
        SUBMITPATH='./'+str(SUMMITDIR)+'/'+str(os.path.basename(__file__).split('.')[0]) +'_'+str(epoches)+'.csv'
    else:
        SUBMITPATH='./'+str(SUMMITDIR)+'/'+str(name)+'.csv'
    submit.to_csv(SUBMITPATH, index=False)

def save_model(model, path, epoch):
    if(epoch % 10 ==9):
        torch.save(model, path + str(epoch + 1) + ".pt")
        

def load_model(path):
    return torch.load(path)

def main():

    MODELPATH=""
    MODELNAME=MODELPATH.split('.')[1].split('/')[1]
    print(MODELNAME)
    ensemble_modelpath=['','','']
    
    global patch_size, stride
    patch_size = 224  # 패치 크기
    stride = 112  # 스트라이드

      #epoches 및 batsize설정
    global epoches, batchsize
    epoches = 100
    batchsize = 16
    
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
    #model = DeepUNet().to(device)
    #model=DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True).to(device)
    #model=DeepLabv3_plus_Xception(nInputChannels=3, n_classes=1, os=16, pretrained=True).to(device)
    #model=HRNet(config).to(device)
    

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #학습
    train(model,criterion,optimizer,train_dataloader)
    
    #모델 불러오기
    #model = load_model(MODELPATH)
    

    #test dataset 설정
    test_dataset,test_dataloader=set_test_dataset(TESTPATH,transform)

    # 예측
    result=predict(model,test_dataloader)
    
    # 앙상블
    models=[]
    for modelpath in ensemble_modelpath:
        model=load_model(modelpath)
        models.append(model)
    
    result=ensemble(models, test_dataloader)
    
    #제출 파일 저장
    sumbit_save(result, MODELNAME)

if __name__== '__main__':
    main()
