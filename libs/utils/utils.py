import os
import torch
import gc
import pandas as pd
import numpy as np

from multiprocessing import freeze_support
import albumentations as A
from albumentations.pytorch import ToTensorV2

def init(path):
    
    mainpath=path
    print(mainpath)
    os.chdir(mainpath)
    
    freeze_support()
    gc.collect()
    torch.cuda.empty_cache()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu=torch.cuda.device_count()
    num_workers=n_gpu*4
    
    print('Device:', device)
    print('num_workers: ',num_workers)
    
    return device,num_workers
    
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
    
def test_sumbit_save(result,modelname=""):
    submit = pd.read_csv('./test_submission.csv')
    submit['mask_rle'] = result
    SUMMITDIR='submit_test'
    if(not os.path.exists(SUMMITDIR) or os.path.isfile(SUMMITDIR)):
        os.mkdir(SUMMITDIR)
    
    SUBMITPATH='./'+str(SUMMITDIR)+'/'+str(modelname) +'.csv'
    submit.to_csv(SUBMITPATH, index=False)
    

def save_model(model, epoch, classname =""):
    modelpt = classname + str(epoch) + ".pt"
    torch.save(model, modelpt)
        

def load_model(path):
    return torch.load(path)

