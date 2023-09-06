
import torch
from torch.utils.data import  DataLoader
from libs.datasets.dataset import *
from libs.models.hrnet import *
from libs.models.unet import *
from libs.models.deepunet import *
from libs.models.deeplabV3plus import *
from libs.utils.utils import *
from libs.core.function import *



def split_validation(path, trainpath, validpath, valid_groundpath):
    df=pd.read_csv(path)

    # train_img를 8:2로 나누어서 valid_train.csv와 valid_ground_truth.csv를 만든다.
    valid_train_df=df.sample(frac=0.8)
    valid_train_df.reset_index(drop=True, inplace=True)
    valid_train_df.to_csv(trainpath,index=False,encoding='utf-8')

    valid_ground_truth_df=df.drop(valid_train_df.index)
    valid_ground_truth_df=valid_ground_truth_df.drop(['img_path'],axis=1)
    valid_ground_truth_df.reset_index(drop=True, inplace=True)
    valid_ground_truth_df.to_csv(valid_groundpath,index=False,encoding='utf-8')

    valid_df=df.drop(valid_train_df.index)
    valid_df=valid_df.drop(['mask_rle'],axis=1)
    valid_df.reset_index(drop=True, inplace=True)
    valid_df.to_csv(validpath,index=False,encoding='utf-8')

def set_train_dataset(path,transform, patch_size, stride,batchsize=16):
    
    dataset = SatelliteDataset(csv_file=path, transform=transform, patch_size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)

    return dataset,dataloader

def set_valid_dataset(path,transform, patch_size, stride, batchsize=16):
    dataset = SatelliteDataset(csv_file=path, transform=transform, patch_size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=4)

    return dataset,dataloader

def main():

    device = init(os.path.dirname(__file__))
    
    transform = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )
    patch_size = 1024
    stride = 0
    epoches = 50
    
    #csv파일 경로 설정
    train_allpath = './train_all.csv'
    trainpath = './train.csv'
    validpath = './valid.csv'
    valid_groundpath = './valid_ground_truth.csv'
    
    #validdation set 분리
    split_validation(train_allpath,trainpath,validpath, valid_groundpath)
    
    # train dataset 설정
    train_dataset,train_dataloader=set_train_dataset(trainpath, transform, patch_size, stride)
    
    # valid dataset 설정
    valid_dataset,valid_dataloader=set_valid_dataset(validpath ,transform, patch_size, stride) 

    # model 초기화
    classname = "unet"
    model = UNet().to(device)
    #model = DeepUNet().to(device)
    #model=DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True).to(device)
    #model=get_seg_model(device)
    
    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #학습
    train(model,criterion,optimizer,train_dataloader,device,epoches,valid_csv=validpath,valid_ground_csv=valid_groundpath, 
                   classname=classname,isvalid=True, valid_dataloader=valid_dataloader)
    
    
    
if __name__== '__main__':
    main()