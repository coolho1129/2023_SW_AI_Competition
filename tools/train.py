
import torch
from torch.utils.data import  DataLoader

from lib.datasets.dataset import *
from lib.models.hrnet import *
from lib.models.unet import *
from lib.models.deepunet import *
from lib.models.deeplabV3plus import *
from lib.utils.utils import *
from lib.core.function import *

def set_train_dataset(path,transform,patch_size=224,stride=112,batchsize=16):
    dataset = SatelliteDataset(csv_file=path, transform=transform, patch_size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)

    return dataset,dataloader

def main():

    init()
    
    # train dataset 설정
    train_dataset,train_dataloader=set_train_dataset(TRAINPATH,transform) 

    # model 초기화
    model = UNet().to(device)
    #model = DeepUNet().to(device)
    #model=DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True).to(device)
    #model=get_seg_model(device)
    
    #저장된 모델 불러오기
    #MODELPATH=""
    #model = load_model(MODELPATH)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #학습
    train(model,criterion,optimizer,train_dataloader)
    
if __name__== '__main__':
    main()