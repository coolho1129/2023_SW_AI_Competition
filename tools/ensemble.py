
from torch.utils.data import  DataLoader

from lib.datasets.dataset import *
from lib.models.hrnet import *
from lib.models.unet import *
from lib.models.deepunet import *
from lib.models.deeplabV3plus import *
from lib.utils.utils import *
from lib.core.function import *

def set_test_dataset(path,transform,patch_size=224,stride=112,batchsize=16):
    dataset = SatelliteDataset(csv_file=path, transform=transform, infer=True, patch_size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=4)
    return dataset,dataloader

def main():

    init()
    
    MODELNAME="ensemble"
    ensemble_modelpath=['','','','']
    
    #test dataset 설정
    test_dataset,test_dataloader=set_test_dataset(TESTPATH,transform)

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



