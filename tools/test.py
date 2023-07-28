
from torch.utils.data import  DataLoader

from lib.datasets.dataset import *
from lib.models.hrnet import *
from lib.models.unet import *
from lib.models.deepunet import *
from lib.models.deeplabV3plus import *
from lib.utils.utils import *
from lib.core.function import *

def set_test_dataset(path,transform):
    dataset = SatelliteDataset(csv_file=path, transform=transform, infer=True, patch_size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=4)

    return dataset,dataloader

def main():
    
    init()

    MODELPATH=""
    MODELNAME=MODELPATH.split('.')[1].split('/')[1]
    print(MODELNAME)
    
    #모델 불러오기
    model = load_model(MODELPATH)
    
    #test dataset 설정
    test_dataset,test_dataloader=set_test_dataset(TESTPATH,transform)

    # 예측
    result=predict(model,test_dataloader)
    
    #제출 파일 저장
    sumbit_save(result, MODELNAME)

if __name__== '__main__':
    main()