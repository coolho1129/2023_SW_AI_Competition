# 2023_SW_AI_Competition
[link - SW중심대학 공동 AI 경진대회 2023](https://dacon.io/competitions/official/236092/overview/description)

![image](https://github.com/mobuktodae/2023_SW_AI_Competition/assets/87495422/11a242d4-d820-4a6b-9794-420526138331)


## topic
위성 이미지 건물 영역 분할 (Satellite Image Building Area Segmentation)

## problems
위성 이미지의 건물 영역 분할(Image Segmentation)을 수행하는 AI모델을 개발
<br>

## dataset
- download data<br>
  https://drive.google.com/file/d/13EMmfsyjrEbtisVuPOl1VUuJnjI7-4Ez/view<br><br>
- dataset info<br>
  - train_img [dir]<br>
  TRAIN_0000.png ~ TRAIN_7139.png<br>
  1024 x 1024<br><br>
  - test_img [dir]<br>
  TEST_00000.png ~ TEST_60639.png<br>
  224 x 224<br><br>
  - train.csv [파일]<br>
  img_id : 학습 위성 이미지 샘플 ID<br>
  img_path : 학습 위성 이미지 경로 (상대 경로)<br>
  mask_rle : RLE 인코딩된 이진마스크(0 : 배경, 1 : 건물) 정보<br><br>
  학습 위성 이미지에는 반드시 건물이 포함되어 있습니다.<br>
  그러나 추론 위성 이미지에는 건물이 포함되어 있지 않을 수 있습니다.<br>
  학습 위성 이미지의 촬영 해상도는 0.5m/픽셀이며, 추론 위성 이미지의 촬영 해상도는 공개하지 않습니다.<br><br>
  - test.csv [파일]<br>
  img_id : 추론 위성 이미지 샘플 ID<br>
  img_path : 추론 위성 이미지 경로 (상대 경로)<br><br>
  - sample_submission.csv [파일] - 제출 양식<br>
  img_id : 추론 위성 이미지 샘플 ID<br>
  mask_rle : RLE 인코딩된 예측 이진마스크(0: 배경, 1 : 건물) 정보<br>
  단, 예측 결과에 건물이 없는 경우 반드시 -1 처리<br>
  <br><br>

## Library
<img src="https://img.shields.io/badge/python-3.10.1-3776AB"/>  <img src="https://img.shields.io/badge/pytorch-1.10.1-EE4C2C"/> 
<br>


## Strategy
1. data preprocessing
- split images
- sliding windows
  
  train set과 test set의 image size보정 및 Data augmentation

  ``` patches = split_image(image, self.patch_size, self.stride)
    mask_patches = split_mask(mask, self.patch_size, self.stride)
    transformed_patches = [self.transform(image=patch, mask=mask)["image"] for patch,mask in zip(patches, mask_patches)]
    transformed_masks = [self.transform(image=patch, mask=mask)["mask"] for patch,mask in zip(patches, mask_patches)]
  ```


2. model
- Segmentation 기본 모델
  
  - Unet (baseline으로 제공된 모델)
    
    stride : 56, epoch : 13
  - DeeplabV3+
    
    stride : 100,  epoch : 80
  - HRnet
    
    stride : 112 , epoch : 20
<br>

- 모델 변형 
  - Deepunet
    
    Unet에서 Contracting Step이 한 단계 더 진행된 구조  
    ```
    # 
    self.dconv_down9 = self.double_conv(512, 1024)
    self.dconv_down10 = self.double_conv(1024, 1024)    
  
    self.maxpool = nn.MaxPool2d(2)
  
    self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2, padding = 0)       
  
    self.dconv_up1 = self.double_conv(1024, 512)
    self.dconv_up2 = self.double_conv(512, 512)
  
    ```
    
    stride : 112, epoch : 40

<br>    

3. ensemble
- weight-mean ensemble

#|model|weight
|--|---|---|
|0|stride112_hrnet_20.pt|0.3|
|1|stride112_divided_deepUnet_40_transfer.pt|0.3|
|2|stride56_divided_unet_13.pt|0.2|
|3|stride100_divied_deeplab_80.pt|0.2|

<br>

## Final Ranks
- public : 29th (0.80069)
- private : 29th (0.79807)


## Reference
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919)

[Ensemble deep learning: A review](https://arxiv.org/abs/2104.02395)

<br>

## Team Info
### Team Name
티파니
### Team Members
Name|github|
|---|---|
| 문채원(ChaeWon Moon) | [mchaewon](https://github.com/mchaewon) |
| 김은지(EunJi Kim) | [EunJiKim02](https://github.com/EunJiKim02) |
| 김찬호(ChanHo Kim) | [coolho1129](https://github.com/coolho1129) |
| 송혜경(Hyegyeong Song) | [sosschs9](https://github.com/sosschs9) |
| 하재현(JaeHyeon Ha) | [jaehyeonha](https://github.com/jaehyeonha) |




