#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     sharpen.py
@Time     :     2025/12/08 15:16:43
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F 

__all__ = [
    'Sharpen'
]

class Sharpen(nn.Module):
    def __init__(
            self,
            sigma: float,
            kernel_size: int,
            num_of_params:int,
            value_range:list,
            encoder_output_dim:int,
            **kwargs,
        ):
        super().__init__()

        self.params_predictor = nn.Sequential(
            nn.Conv2d(encoder_output_dim,encoder_output_dim//2,3,2,1),
            nn.LeakyReLU(0.02,True),
            nn.Conv2d(encoder_output_dim//2,num_of_params,3,2,1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_of_params,num_of_params,1,1,0),
            nn.Tanh() # [-1,1]
        )
        self.value_range = value_range # [0.1,1]

        self.kernel_size = kernel_size
        gaussian_kernel  = self._make_gaussian_2d_kernel(sigma,self.kernel_size).repeat(3,1,1,1) # [1,3,kernel_size,kernel_size]

        self.register_buffer('kernel',gaussian_kernel)

    def tanh_range(self,params):
        return params * (self.value_range[1] - self.value_range[0]) / 2 + (self.value_range[1] + self.value_range[0]) / 2
    
    def _make_gaussian_2d_kernel(self,sigma, kernel_size=25, dtype=torch.float32):
        # 确定高斯核的中心点
        radius = kernel_size // 2
        
        # 生成一个坐标网格
        x_coords = torch.arange(kernel_size) - radius
        x_grid = x_coords.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        
        # 计算二维高斯核
        xytemp=(x_grid**2 + y_grid**2).float()
        gaussian_kernel = torch.exp(-(xytemp / (2 * sigma**2)))
        
        # 归一化使得和为1
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # 增加batch和channel维度以符合卷积操作的需求
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).type(dtype)
        
        return gaussian_kernel

    def forward(self, x:Tensor,laten_out:Tensor):
        params = self.params_predictor(laten_out) # [B, 1,1,1], in [-1,1]
        params = self.tanh_range(params)

        padding = (self.kernel_size - 1) // 2 

        x_pad   = F.pad(x, (padding, padding, padding, padding), mode='reflect')

        x_blur  = F.conv2d(x_pad,self.kernel,padding=0,groups=3)

        out = (x - x_blur) * params + x 
        # out = (out - out.min()) / (out.max() - out.min())
        return out 


