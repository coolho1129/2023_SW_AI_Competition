# 2023_SW_AI_Competition
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
<img src="https://img.shields.io/badge/python-3.10.1-3776AB"/>  <img src="https://img.shields.io/badge/pytorch-3.10.1-EE4C2C"/> 
<br>

## Usages
<br>

## Directory structure
<br>

## Strategy
1. data preprocessing
- 

2. model train
- Unet
- DeepnetV3+
- Deepunet
- HRnet

3. ensemble
- weight-mean ensemble
- mean ensemble

<br>

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
Name|Role|github|
|---|---|---|
| 문채원 | Team Leader | [mchaewon](https://github.com/mchaewon) |
| 김은지 | | [mobuktodae](https://github.com/mobuktodae) |
| 김찬호 | | [coolho1129](https://github.com/coolho1129) |
| 송혜경 | | [sosschs9](https://github.com/sosschs9) |
| 하재현 | | [jaehyeonha](https://github.com/jaehyeonha) |




