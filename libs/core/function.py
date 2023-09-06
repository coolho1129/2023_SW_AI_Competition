from tqdm import tqdm
import numpy as np
import pandas as pd

from libs.utils.utils import*
from libs.datasets.dataset import*
from libs.metrics.dice_score import *

def valid(model,valid_dataloader, device, valid_csv, valid_ground_csv):
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        result = []
        valid_df=pd.read_csv(valid_csv)
        valid_ground_truth_df=pd.read_csv(valid_ground_csv)
        valid_imageshape=None
        for image in tqdm(valid_dataloader):
            if valid_imageshape==None:
                valid_imageshape=image.shape[2:4]
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
        
        valid_predict_df=pd.DataFrame()
        valid_predict_df['img_id']=valid_df['img_id']
        valid_predict_df['mask_rle']=result
        
        print('valid_score: ',calculate_dice_scores(valid_ground_truth_df,valid_predict_df,img_shape=valid_imageshape))
    
    
        
        
def train(model, criterion, optimizer,dataloader,device,epoches, valid_csv="", valid_ground_csv="", classname="",isvalid=False,valid_dataloader=None):
    # training loop
    for epoch in range(epoches):  # 에폭 동안 학습합니다.
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
        

        if(epoches % 5 == 4):
            save_model(model, epoch + 1, classname)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
        
        if(isvalid):
            valid(model, valid_dataloader, device, valid_csv, valid_ground_csv)
            
            
def predict(model, device, test_dataloader):
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

def ensemble(models, test_dataloader,device,image_size,weights=[]):
    with torch.no_grad():
        num_models = len(models)
        final_mask = np.zeros((len(test_dataloader), 1, image_size[0], image_size[1]), dtype=np.float32)  # 1 차원을 추가하여 형태를 (N, 1, height, width)로 만듭니다.
        if(weights==[]):
            weights=[1/num_models]*num_models
        
        for weight,model in zip(weights,models):
            model.eval()
            for idx, image in enumerate(tqdm(test_dataloader)):
                image = image.float().to(device)
                outputs = model(image)
                mask = torch.sigmoid(outputs).cpu().numpy()
                mask = np.squeeze(mask, axis=1)
                final_mask[idx] += mask *weight

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