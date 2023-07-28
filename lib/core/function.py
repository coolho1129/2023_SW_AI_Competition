from tqdm import tqdm
import numpy as np

from lib.utils.utils import*
from lib.datasets.dataset import*

def train(model,criterion,optimizer,dataloader,epoches=100,name=""):
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

        save_model(model, "./",name, epoch)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
    
    return model

def predict(model,test_dataloader):
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

def ensemble(models, test_dataloader,weights=[]):
    with torch.no_grad():
        num_models = len(models)
        final_mask = np.zeros((len(test_dataloader), 1, 224, 224), dtype=np.float32)  # 1 차원을 추가하여 형태를 (N, 1, 224, 224)로 만듭니다.
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