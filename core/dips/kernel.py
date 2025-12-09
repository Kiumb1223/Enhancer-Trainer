#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     kernel.py
@Time     :     2025/12/08 15:16:46
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn 
from torch import Tensor 
import torch.nn.functional as F 

__all__ = [
    'Kernel'
]

class Kernel(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            encoder_output_dim: int,
            **kwargs
        ):
        super().__init__()

        self.kernel_size = kernel_size
        num_of_params    = 6*self.kernel_size**2

        self.params_predictor = nn.Sequential(
            nn.Conv2d(encoder_output_dim,encoder_output_dim//2,3,2,1),
            nn.LeakyReLU(0.02,True),
            nn.Conv2d(encoder_output_dim//2,num_of_params,3,2,1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_of_params,num_of_params,1,1,0),
            nn.Tanh() # [-1,1]
        )

    def forward(self, x: Tensor,latent_out: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert C == 3, "Only 3-channel supported"

        params = self.params_predictor(latent_out).squeeze()  # [B, 2*3*k*k] in [-1,1]
        params = params.view(B, 2, C, self.kernel_size, self.kernel_size)

        K1 = params[:, 0]  # [B, 3, k, k]
        K2 = params[:, 1]  # [B, 3, k, k]
        K1 = K1 / (K1.abs().sum(dim=[2,3], keepdim=True) + 1e-6)
        K2 = K2 / (K2.abs().sum(dim=[2,3], keepdim=True) + 1e-6)

        # x = torch.clamp(x, 0.0, 1.0)

        # --- Vectorized convolution without loop ---
        padding = self.kernel_size // 2

        # Reshape input: [B, 3, H, W] -> [1, B*3, H, W]
        x_flat = x.view(1, B * C, H, W)

        # Reshape kernels: [B, 3, k, k] -> [B*3, 1, k, k]
        K1_flat = K1.reshape(B * C, 1, self.kernel_size, self.kernel_size)
        K2_flat = K2.reshape(B * C, 1, self.kernel_size, self.kernel_size)

        # Grouped convolution: each of the B*3 "channels" uses its own 1x1 kernel
        conv1_flat = F.conv2d(x_flat, K1_flat, padding=padding, groups=B * C)
        conv2_flat = F.conv2d(x_flat, K2_flat, padding=padding, groups=B * C)

        # Reshape back: [1, B*3, H, W] -> [B, 3, H, W]
        conv1 = conv1_flat.view(B, C, H, W)
        conv2 = conv2_flat.view(B, C, H, W)

        # Final output
        out = x * conv1 + conv2 + x
        out = (out - out.min()) / (out.max() - out.min())

        return out