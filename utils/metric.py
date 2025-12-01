#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     metric.py
@Time     :     2025/12/01 19:21:06
@Author   :     Louis Swift
@Desc     :     
'''



import torch
from torch import Tensor 
from pytorch_msssim import ssim

def calc_psnr(pred:Tensor,gt:Tensor, max_val=1.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calc_ssim(pred:Tensor,gt:Tensor, max_val=1.0):
    return ssim(pred,gt, data_range=max_val,size_average=True)


