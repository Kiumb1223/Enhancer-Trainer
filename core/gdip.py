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

import os 
import torch
from .dips import * 
import torch.nn as nn 
from torch import Tensor 
from datetime import datetime
import torch.nn.functional as F 
import torchvision.utils as vutils
# from typing import Optional
# from omegaconf import OmegaConf

__all__ = [
    'GatedDIP',
]

class GatedDIP(nn.Module):
    def __init__(
            self,
            idx:int,
            filters:list,
            output_dim_lst:list,
            **kwargs
        ):
        super().__init__()

        self.filters = filters 

        for filter in self.filters:

            filter_config = kwargs.get(filter,None)
            if filter != 'identity':
                assert filter_config is not None, f'filter [{filter}] config is None.'
                filter_config.encoder_output_dim = output_dim_lst[idx]

            self.add_module(
                filter,build_filters(filter,filter_config)
            )

        self.gate_module = build_gate(kwargs.get('gate',None))

    def forward(self,x:Tensor,latent_out:Tensor,feature_map:Tensor):
        
        # 1. calc each filter
        filters_out = []

        for filter in self.filters:
            out = getattr(self,filter)(x,latent_out)
            filters_out.append(out) # [b,3,h,w]

        # 2. calc gate weight
        gate = self.gate_module(x,filters_out,feature_map) # [b,n,3,1,1]

        # 3. calc fusion
        # gate: [B, N, 3, 1, 1] -> broadcast to H,W
        x_fusion = torch.stack(filters_out,dim=1) # [B, N, 3, H, W]
        x_fusion = (gate * x_fusion).sum(dim=1)   # sum over filters, result [B, 3, H, W]

        x_fusion = x_fusion.clamp(min=0,max=1)
        
        if self.training:
            return x_fusion,gate
        else:
            return x_fusion
    
        