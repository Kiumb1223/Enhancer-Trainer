#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     enhancer.py
@Time     :     2025/11/24 21:11:56
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn
from torch import Tensor
from omegaconf import OmegaConf
from kornia.filters import bilateral_blur

from .gdip import GatedDIP
from .vision_encoder import VisionEncoder


__all__ = [
    'Enhancer'
]

class Enhancer(nn.Module):
    def __init__(
            self,
            type: str, # optional single,multi
            num_dip: int,
            encoder: OmegaConf,
            gdip: OmegaConf,
            ckpt_path: str = None,
            **kwargs
        ):
        super().__init__()

        assert type in ['single','multi']

        if type == 'single':
            assert num_dip == 1, 'single type only support one GDIP'
        elif self.type == 'multi':
            assert num_dip <= 4, "num_dip cannot exceed number of encoder feature levels"

        self.type = type 
        self.num_dip = num_dip        

        self.vision_encoder = VisionEncoder(**encoder)

        if self.type == 'multi':
            for i in range(num_dip):
                self.add_module(
                    f'gdip_{i}',GatedDIP(idx=i,**gdip)
                )
        else:
            self.gdip = GatedDIP(idx=-1,**gdip)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    def forward(self,x:Tensor):

        """
        # denoise 
        x = bilateral_blur(
            input=x,
            kernel_size=5,
            sigma_color=0.1,
            sigma_space=(2.0,2.0)
        )
        """

        latent_out_lst,featmap_lst = self.vision_encoder(x)
        gdip_out = []
        gate_out = [] 
        if self.type == 'multi':
            for i in range(self.num_dip):
                gdip = getattr(self,f"gdip_{i}")
                latent_out = latent_out_lst[i]
                feat = featmap_lst[i]
                x,gate = gdip(x,latent_out,feat)
                gdip_out.append(x)
                gate_out.append(gate)
        
        else:
            x,gate = self.gdip(x,latent_out_lst[-1],featmap_lst[-1])
            gdip_out.append(x)
            gate_out.append(gate)

        if self.training:
            return gdip_out,gate_out
        else:
            return x,gate
        
        """
        x_min = x_fusion.amin(dim=(2,3), keepdim=True)   # [B,C,1,1]
        x_max = x_fusion.amax(dim=(2,3), keepdim=True)
        x_fusion = (x_fusion - x_min) / (x_max - x_min + 1e-6)
        """