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
from .modules.afpn import * 
from .modules.starConv import * 
from .modules.basic import * 

__all__ = [
    'VisionEncoder'
]

class VisionEncoder(nn.Module):
    def __init__(
            self,
            type:str,
            act_type:str,
            use_pre_decoder:bool,
            use_afpn:bool,
            num_layers:int = 5,
            base_channel:int = 64,
            encoder_output_dim:int = 256,

            **kwargs
        ):
        super().__init__()


        assert type in ["vanilla", "star"]
        assert act_type in ['lrelu','telu']

        if use_afpn and num_layers <= 2:
            raise ValueError("num_layers must be greater than 2 when use_afpn is True")

        self.use_afpn = use_afpn
        self.num_layers = num_layers
        self.encoder_output_dim = encoder_output_dim
        self.base_channel = base_channel

        self.convs    = nn.ModuleList() 
        self.adconvs  = nn.ModuleList() 

        if use_pre_decoder:
            # 额外 增加两层 卷积 使其特征变化更加自然，同时进行下采样，降低特征图分辨率
            self.pre_decoder = nn.Sequential(
                BasicConv(3, base_channel // 4, kernel_size=3, stride=2, pad=1), # 不 进行下采样
                BasicConv(base_channel // 4, base_channel // 2, kernel_size=3, stride=2, pad=1)
            )
            in_channel = base_channel // 2 
        else:    
            self.pre_decoder = None 
            in_channel = 3 

        out_channel = base_channel
        in_channels_lst = [] 

        for i in range(num_layers):
            if type == 'vanilla':
                self.convs.append(
                    BasicConv(in_channel,out_channel,kernel_size=3,stride=2,pad=1) # 通道调整 + 下采样
                )
            elif type == 'star':
                self.convs.append(
                    StarConv(in_channel, out_channel,kernel_size=3,stride=2,pad=1) # 通道调整 + 下采样
                )
            if not self.use_afpn:
                self.adconvs.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(out_channel,out_channel, kernel_size=1, stride=1, padding=0),nn.SELU(),
                        nn.Conv2d(out_channel,encoder_output_dim,kernel_size=1,stride=1,padding=0)
                    )
                )
            else:
                self.adconvs.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        # nn.Conv2d(encoder_output_dim,encoder_output_dim, kernel_size=1, stride=1, padding=0),nn.SELU(),
                        # nn.Conv2d(encoder_output_dim,encoder_output_dim,kernel_size=1,stride=1,padding=0)
                    )
                )

            #---------------------------------#
            # From GDIP-YOLO: 
            #       The number of channels in each layer is double the previous, 
            #       starting from 64 in the first layer and 1024 in the final layer.
            #---------------------------------#
            in_channel = out_channel
            out_channel = out_channel * 2 
            in_channels_lst.append(in_channel)

        if self.use_afpn:
            self.afpn = AFPN(in_channels_lst, encoder_output_dim)

    def forward(self,x):

        outputs = []
        mid_feats = []

        if self.pre_decoder is not None:
            x = self.pre_decoder(x)

        for i in range(self.num_layers):
            x = self.convs[i](x)
            mid_feats.append(x)

        # If using AFPN, replace raw feats with fused ones.
        if self.use_afpn:
            mid_feats = self.afpn(mid_feats)

        for i in range(self.num_layers):
            out = self.adconvs[i](mid_feats[i])
            outputs.append(out)

        return outputs
