#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     vision_encoder.py
@Time     :     2025/11/24 20:56:20
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn 
from torch import Tensor
from .basicConv import *
import torch.nn.functional as F 
from typing import Union,List,Tuple


__all__ = [
    'VisionEncoder'
]


class VisionEncoder(nn.Module):
    def __init__(
            self,
            dip_type:str,
            use_pre_decoder:bool,
            num_layers:int = 5,
            base_channel:int = 32,
            encoder_output_dim:int = 256,
            **kwargs
        ):
        super().__init__()

        self.dip_type = dip_type

        in_channels = 3 
        out_channels = base_channel
        self.use_pre_decoder = use_pre_decoder
        if self.use_pre_decoder:
            self.pre_decoder = nn.Sequential(
                BasicConv(in_channels,base_channel//2,3,2,1), # downsample
                BasicConv(base_channel//2,base_channel//2,1,1,0), 
            )
            in_channels = base_channel // 2 

        self.convs = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.adconvs = nn.ModuleList() 

        self.num_layers = num_layers
        for i in range(num_layers):
            self.convs.append(
                nn.Sequential(
                    BasicConv(in_channels,out_channels,3,2,1), # downsample
                    BasicConv(out_channels,out_channels,1,1,0), 
                )
            )
            self.adconvs.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(out_channels,out_channels, kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(0.02,True),
                        nn.Conv2d(out_channels,encoder_output_dim,kernel_size=1,stride=1,padding=0)
                    )
                )
            # for gate 
            # 调整通道数
            if i != 0: 
                # 后续特征图的尺寸需对齐第一层特征图，故 第一层 不需要额外的调整通道数
                self.downs.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels,base_channel,1,1,0),
                        nn.LeakyReLU(0.02,True),
                        nn.Conv2d(base_channel, base_channel,1,1,0),
                        nn.LeakyReLU(0.02,True),
                    )
                )

            in_channels = out_channels
            out_channels *= 2 # double the channels
        
        if self.dip_type == 'single':
            # fuse all the features 
            in_channels = base_channel * self.num_layers
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2,1,1,0),
                nn.LeakyReLU(0.02,True),
                nn.Conv2d(in_channels // 2, base_channel,1,1,0),
                nn.LeakyReLU(0.02,True),
                nn.Conv2d(base_channel, base_channel,1,1,0),
            )

    def forward(self,x:Tensor) -> Tuple[List[Tensor],Union[List[Tensor],Tensor]]:
        """

        Args:
            x (Tensor): B 3 H W 

        Returns:
            List[Tensor]: for following params predictors
            Union[List[Tensor],Tensor]:  for following gate module
                - self.dip_type == 'single': Tensor B base_channel H W
                - self.dip_type == 'multi': List[Tensor] B base_channel H W
        """

        if self.use_pre_decoder:
            x = self.pre_decoder(x) # b, base_channels//2, h//2, w//2

        features = []
        mid_feat = []

        for i in range(self.num_layers):
            x = self.convs[i](x)
            mid_feat.append(x)

            if i != 0:
                down_x = self.downs[i-1](x)
                base_shape = features[0].size()[2:]
                down_x = F.interpolate(
                            down_x,
                            size=base_shape,
                            mode='bilinear',
                            align_corners=False
                        )
                features.append(down_x) # b,base_channels,h//4,w//4
            else:
                features.append(x)
        
        latent_out = []
        for i in range(self.num_layers):
            out = self.adconvs[i](mid_feat[i])
            latent_out.append(out)

        if self.dip_type == 'single':
            features = self.fuse(torch.cat(features,dim=1)) 
        
        return latent_out,features 