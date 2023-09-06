
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

def get_imagesize(path):
     data = pd.read_csv(path)
     img_path = data.iloc[0, 1]
     image = cv2.imread(img_path)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     img_size = (image.shape[0],image.shape[1])
    
     return img_size
        

def main():
    
    #set hyperparameters
    device,num_workers = init(os.path.dirname(__file__))
    transform = A.Compose(
        [   
            A.Normalize(),
            ToTensorV2()
        ]
    )
    
    testpath = "./test.csv"
    MODELNAME="ensemble"
    ensemble_modelpath=['','','','']
    
    #test dataset 설정
    test_dataset,test_dataloader=set_test_dataset(testpath,transform,num_workers)

    # 앙상블
    models=[]
    for modelpath in ensemble_modelpath:
        model=load_model(modelpath)
        models.append(model)
    
    result=ensemble(models, test_dataloader,device,get_imagesize(testpath))
    
    #제출 파일 저장
    test_sumbit_save(result, MODELNAME)

if __name__== '__main__':
    main()



