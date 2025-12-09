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

        self.type = type 
        self.num_dip = num_dip        

        self.vision_encoder = VisionEncoder(**encoder)

        if self.type == 'multi':
            for i in range(num_dip):
                self.add_module(
                    f'gdip_{i}',GatedDIP(**gdip)
                )
        else:
            self.gdip = GatedDIP(**gdip)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path))
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

        latent_out_lst,multi_feats_lst = self.vision_encoder(x)
        gdip_out = []
        if self.type == 'multi':
            for i in range(self.num_dip):
                gdip = getattr(self,f"gdip_{i}")
                latent_out = latent_out_lst[i]
                feat = multi_feats_lst[i]
                x,gate = gdip(x,latent_out,feat)
                gdip_out.append(x)
        
        else:
            x,gate = self.gdip(x,latent_out[-1],multi_feats_lst)
            gdip_out.append(x)

        if self.training:
            return gdip_out,gate
        else:
            return x,gate