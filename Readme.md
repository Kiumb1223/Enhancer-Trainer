# Enhancer-Trainer

> It contains the custom trainer designed specifically for the **Image Enhancer**, which is the core of my project [CDIE-Trainer](https://github.com/Kiumb1223/CDIE-Trainer).

## Update

1. 2025-12-05:  Improved DIP Architecture （PSNR 17.735，SSIM0.771):
   - Refined the network architectures of both the Vision Encoder and the DIP Module.
   - Added bilateral filtering to enhance noise suppression performance.

## 1. Environment

To run this project, set up the environment as specified in the **CDIE-Trainer** repository. Additionally, make sure to install the following two packages:

``` 
conda install albumentations=2.0.0
pip install pytorch_msssim
pip install kornia==0.8.1
```

## 2. Dataset

We use the PASCAL VOC 2007 and 2012 datasets and split them into training and validation subsets. The exact splitting strategy and file lists are defined in [`dataSplit.py`](dataSplit.py).

## 3. Usage

1. Check the configuration file.
2. Then, 

```
python train.py
```

