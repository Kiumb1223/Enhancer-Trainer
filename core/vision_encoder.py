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
from .modules.activation import * 

__all__ = [
    'VisionEncoder'
]

class VisionEncoder(nn.Module):
    def __init__(
            self,
            type:str,
            act_type:str,
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
        self.maxpools = nn.ModuleList()
        self.adpools  = nn.ModuleList()
        self.linears  = nn.ModuleList()

        in_channel = 3 
        out_channel = base_channel
        in_channels_lst = [] 

        for i in range(num_layers):
            if type == 'vanilla':
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1,padding=1),
                        # nn.LeakyReLU(0.1, inplace=True),
                        gen_activation(act_type)(0.1,inplace=True)
                    ) 
                )
            elif type == 'star':
                self.convs.append(
                    StarConv(in_channel, out_channel)
                )
            if i != num_layers - 1: 
                self.maxpools.append(
                    # 严格下采样
                    nn.AvgPool2d(kernel_size=2,stride=2)
                )

            self.adpools.append(
                nn.AdaptiveAvgPool2d((1,1))
            )
            self.linears.append(
                nn.Sequential(
                    nn.Linear(
                        encoder_output_dim if use_afpn else out_channel,
                        encoder_output_dim
                    ),
                    # nn.LeakyReLU(0.1, inplace=True),
                    gen_activation(act_type)(0.1,inplace=True)
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

        for i in range(self.num_layers):
            x = self.convs[i](x)
            mid_feats.append(x)
            if i != self.num_layers - 1:
                x = self.maxpools[i](x)
            else:
                x = x 

        # If using AFPN, replace raw feats with fused ones.
        if self.use_afpn:
            mid_feats = self.afpn(mid_feats)

        for i in range(self.num_layers):
            x_adp = self.adpools[i](mid_feats[i])
            x_vec = x_adp.view(x_adp.shape[0], -1)

            outputs.append(self.linears[i](x_vec)) 

        return outputs
