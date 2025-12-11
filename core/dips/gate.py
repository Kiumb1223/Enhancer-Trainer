#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     gate.py
@Time     :     2025/12/11 09:46:25
@Author   :     Louis Swift
@Desc     :     门控网络
'''

import torch 
import torch.nn as nn 
from torch import Tensor
from typing import List 

__all__ = [
    'GateModule'
]

class GateModule(nn.Module):
    def __init__(
            self,
            filters:list,
            bt_res_concat: bool,
            
            **kwargs
        ):
        super().__init__()

        self.num_filters = len(filters)
        self.bt_res_concat = bt_res_concat

        if self.bt_res_concat:
            in_channels = 2 * ( 3 * self.num_filters)
        else:
            in_channels = 3 * self.num_filters

        out_channels = 3 * self.num_filters

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2 ,padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2,True),
        )

        self.fuse_module = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2,True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self,x:Tensor,filters_out:List[Tensor],feature_map:Tensor):
        """
        Compute global, per-filter, per-channel gating weights.

        Args:
            x: Input image of the current stage, shape [B, 3, H, W].
            filters_out: List of N enhanced images from parallel filters,
                each of shape [B, 3, H, W] in RGB order.
            feature_map: Aligned backbone feature map used to guide gating,
                expected shape [B, 32, H/4, W/4].

        Returns:
            gate: Gating weights of shape [B, N, 3, 1, 1],
                where gate[:, i, c, 0, 0] is the weight for filter i and channel c (R/G/B),
                normalized across filters for each RGB channel.
        """
        b,c,h,w = x.shape 
        y = torch.stack(filters_out,dim=1) # [B,N,3,H,W]
        diff_y = y - x.unsqueeze(1)

        y = y.reshape(b,3*self.num_filters,h,w)           # [B,3*N,H,W]
        diff_y = diff_y.reshape(b,3*self.num_filters,h,w) # [B,3*N,H,W]

        if self.bt_res_concat:
            input_y = torch.cat([y,diff_y],dim=1)         # [B,6*N,H,W]
        else:
            input_y = diff_y
        
        enc_y = self.encoder(input_y)                     # [B,32,H/4,W/4]
        
        assert feature_map.shape == enc_y.shape,\
              RuntimeError(f"Feature map size mismatch: {feature_map.shape} vs {enc_y.shape}")

        fuse_y = torch.cat([enc_y,feature_map],dim=1)     # [B,64,H/4,W/4]
        fuse_y = self.fuse_module(fuse_y)                 # [B,32,H/8,W/8]

        gate = self.decoder(fuse_y)                       # [B,3*N,1,1]

        # actually gate is [f1_R, f1_G, f1_B, f2_R, ...]
        gate = gate.reshape(b,self.num_filters,3,1,1)     # [B,N,3,1,1]
        gate = torch.softmax(gate, dim=1)                 # sum_i gate[:,i,c] = 1

        return gate
