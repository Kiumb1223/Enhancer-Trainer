#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     gdip.py
@Time     :     2025/11/24 20:04:23
@Author   :     TZM12138 @ https://github.com/TZM12138  && Louis Swift 
@Desc     :     

            Mainly reference: https://github.com/Gatedip/GDIP-Yolo
            Thanks a lot for their brilliant works.

'''


import math
import torch
from .dips import * 
import torch.nn as nn 
from torch import Tensor 
import torch.nn.functional as F 

# from typing import Optional
# from omegaconf import OmegaConf

__all__ = [
    'GatedDIP',
]

class GatedDIP(torch.nn.Module):
    def __init__(
            self,
            filters:list,
            base_channel: int,
            encoder_output_dim:int,
            # bezier_config:Optional[OmegaConf]=None,
            # kernel_config:Optional[OmegaConf]=None,
            # defog_config:Optional[OmegaConf]=None,
            # sharpen_config:Optional[OmegaConf]=None,
            **kwargs
        ):
        super().__init__()

        self.filters = filters 

        for filter in self.filters:
            self.add_module(
                filter,build_filters(filter,kwargs.get(filter,None))
            )

        out_channels = len(self.filters) * 3  # 每个filter 输出都是3通道的tensor
        self.gate_linspace = [i for i in range(0,out_channels,len(self.filters))]
        self.gate_module = nn.Sequential(
            nn.Conv2d(base_channel,base_channel//2, kernel_size=1),
            nn.LeakyReLU(0.02,True),
            nn.Conv2d(base_channel//2, base_channel//2, kernel_size=3,padding=1),
            nn.LeakyReLU(0.02,True),
            nn.Conv2d(base_channel//2, out_channels, kernel_size=1),
        )

    def forward(self,x:Tensor,latent_out:Tensor,feature_map:Tensor):
        
        b,c,h,w = x.shape 

        # 1. calc gate 
        gate = self.gate_module(feature_map) # [b,3*len(self.filters),h,w]
        gate = F.interpolate(
            gate,
            size=(h,w),
            mode='bilinear',
            align_corners=False
        )

        # 2. calc each filter
        out_list = []

        for filter in self.filters:
            out = getattr(self,filter)(x,latent_out)
            out_list.append(out) # [b,3,h,w]

        out_stack = torch.stack(out_list, dim=1)        # [b, n_filters, 3, h, w]
        out_stack = out_stack.permute(0, 2, 1, 3, 4)    # [b, 3, n_filters, h, w]
        b, c, n_f, h, w = out_stack.shape
        out_stack = out_stack.reshape(b, c * n_f, h, w)  # [b, 3*n_filters, h, w]

        # 3. calc final output
        x_f0 = torch.sum(
                F.softmax(gate[:,:self.gate_linspace[1],:,:],dim=1) * 
                out_stack[:,:self.gate_linspace[1],:,:],dim=1, keepdim=True
            )
        x_f1 = torch.sum(
                F.softmax(gate[:,self.gate_linspace[1]:self.gate_linspace[2],:,:],dim=1) *
                out_stack[:,self.gate_linspace[1]:self.gate_linspace[2],:,:],dim=1, keepdim=True
            )
        x_f2 = torch.sum(
                F.softmax(gate[:,self.gate_linspace[2]:,:,:],dim=1) *
                out_stack[:,self.gate_linspace[2]:,:,:],dim=1, keepdim=True
            )
        x_fusion = torch.cat([x_f0,x_f1,x_f2],dim=1)

        if self.training:
            return x_fusion,gate
        else:
            return x_fusion