# TlED-Net

##### *Triple-loop Encoder-Decoder Architecture Based Semantic Segmentation Network With Dense Skip Connections*

## Model Overview

1. *We proposed a novel network architecture and experimentally validated its superiority in certain aspects while discussing its limitations.*

2.  *Employing various representation strategies to reconstruct a variant of this architecture for building the backbone of TlED-Net.*

3. *Enhancing the network's capacity of feature fusion through dense skip connections.*

4. *Constructed a novel dual attention mechanism and embedded it according to the architectural characteristics of the network.*

5. *Optimized the deep-level structures.*

## Model Structure

![image-20231115073013747](https://github.com/weiyuanhong623/TlED-Net/blob/master/images/TlED-Net.png?raw=true)

## **Environment**

*IDE: PyCharm 2020.1 Professional Edition.*

*Framework:  PyTorch 1.13.0.*

*Language: Python 3.8.15*

*CUDA: 11.7*

## Datasets

1. *LUNG dataset: https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data.*

2. *Skin Lesion dataset: https://www.kaggle.com/datasets/ojaswipandey/skin-lesion-dataset.*

3. *ISIC2017-Nevus: https://challenge.isic-archive.com/data/#2017*

4. *Digital Retinal Images for Vessel Extraction(DRIVE) dataset: [Introduction - Grand Challenge (grand-challenge.org)](https://drive.grand-challenge.org/)*

5. *Kaggle 2018 Data Science Bowl (CELL) dataset: https://www.kaggle.com/competitions/data-science-bowl-2018/data*

6. *MICCAI2015-CVC ClinicDB dataset: https://polyp.grand-challenge.org/CVCClinicDB/*

7. *ISIC2017-Seborrheic Keratosis dataset: https://challenge.isic-archive.com/data/#2017*

## Repository Overview

| Directory              | Contents                                      |
| ---------------------- | --------------------------------------------- |
| [Ablation-TlED-Net](https://github.com/YF-W/TlED-Net/tree/main/Ablation-TlED-Net)  | Ablation models based on TlED-Net             |
| [Ablation-Unet](https://github.com/YF-W/TlED-Net/tree/main/Ablation-Unet)      | Transplantation ablation models based on Unet |
| [Models-MlEDA](https://github.com/YF-W/TlED-Net/tree/main/Models-MlEDA)       | Models based on MlEDA architecture            |
| [Comparative-models](https://github.com/YF-W/TlED-Net/tree/main/Comparative-models) | 13 comparative models                         |

