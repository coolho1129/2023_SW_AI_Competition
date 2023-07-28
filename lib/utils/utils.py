import os
import torch
import gc
import pandas as pd
import numpy as np

from multiprocessing import freeze_support
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    
    global patch_size, stride
    patch_size = 224  # 패치 크기
    stride = 112  # 스트라이드

    #epoches 및 batsize설정
    global epoches, batchsize
    epoches = 100
    batchsize = 16
    
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

transform = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )