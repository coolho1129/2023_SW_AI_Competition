
from torch.utils.data import  DataLoader

from libs.datasets.dataset import *
from libs.models.hrnet import *
from libs.models.unet import *
from libs.models.deepunet import *
from libs.models.deeplabV3plus import *
from libs.utils.utils import *
from libs.core.function import *

def set_test_dataset(path,transform, batchsize,num_workers):
    dataset = SatelliteDataset(csv_file=path, transform=transform,infer=True)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)

    return dataset,dataloader

def main():
    
    device,num_workers = init(path  = os.path.dirname(__file__))
    transform = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )
    
    batchsize=16
    
    modelpt=""
    modelname=modelpt.split('.')[1].split('/')[1]
    print(modelname)
    
    #모델 불러오기
    model = load_model(modelpt)
    
    #test dataset 설정
    testpath = "./test.csv"
    test_dataset,test_dataloader=set_test_dataset(testpath,transform,batchsize,num_workers)

    # 예측
    result=predict(model,device, test_dataloader)
    
    #제출 파일 저장
    test_sumbit_save(result,modelname)

if __name__== '__main__':
    main()