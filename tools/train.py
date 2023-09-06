
import torch
from torch.utils.data import  DataLoader

from libs.datasets.dataset import *
from libs.models.hrnet import *
from libs.models.unet import *
from libs.models.deepunet import *
from libs.models.deeplabV3plus import *
from libs.utils.utils import *
from libs.core.function import *

def set_train_dataset(path,transform,batchsize,num_workers,splits=False,patch_size=None, stride=None):
    
    dataset = SatelliteDataset(csv_file=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)

    return dataset,dataloader

def main():

    device,num_wokers = init(path  = os.path.dirname(__file__))
    
    transform = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )
    epoches = 50
    batchsize=16
    
    # train dataset 설정
    trainpath = './train_all.csv'
    train_dataset,train_dataloader=set_train_dataset(trainpath,transform,batchsize,num_wokers) 

    # model 초기화
    model_name="unet"
    model = UNet().to(device)
    
    #model = DeepUNet().to(device)
    #model=DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True).to(device)
    #model=get_seg_model(device)
    
    
    #model=nn.DataParallel(model)
    #저장된 모델 불러오기
    #MODELPATH=""
    #model = load_model(MODELPATH)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #학습
    train(model,criterion,optimizer,train_dataloader,device,epoches,name=model_name)
    
if __name__== '__main__':
    main()