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
from torchvision import models 

__all__ = [
    'VisionEncoder'
]

from torchvision.models import \
    ResNet50_Weights, ResNet101_Weights,\
    ResNet152_Weights, ResNeXt50_32X4D_Weights,\
    ResNeXt101_32X8D_Weights

weights_map = {
    'resnet50': ResNet50_Weights.IMAGENET1K_V1,
    'resnet101': ResNet101_Weights.IMAGENET1K_V1,
    'resnet152': ResNet152_Weights.IMAGENET1K_V1,
    'resnext50_32x4d': ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
    'resnext101_32x8d': ResNeXt101_32X8D_Weights.IMAGENET1K_V1
}

class VisionEncoder(nn.Module):
    def __init__(
            self,
            arch:str,
            dip_type:str,
            output_dim_lst:list = [64,64,64,64],
            **kwargs
        ):
        super().__init__()

        assert arch in weights_map.keys(), 'Not supported arch.'
        backbone = models.__dict__[arch](weights=weights_map[arch])
        # del backbone.fc  # unnecessary

        self.backbone = backbone 
        
        # 该backbone 最多提取 4 个 特征图
        # 假设 输入 input size - [1,3,418,418]
        # layer1 : [1,256,105,105]
        # layer2 : [1,512,53,53]
        # layer3 : [1,1024,27,27]
        # layer4 : [1,2048,14,14]
        self.latent_convs  = nn.ModuleList() 
        self.featmap_convs = nn.ModuleList() 

        down_channels = [256,512,1024,2048]
        assert len(output_dim_lst) == 4, 'output_dim_lst must have 4 elements'

        for i,down_channel in enumerate(down_channels):
            self.latent_convs.append(
                nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(down_channel,down_channel, kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(0.02,True),
                        nn.Conv2d(down_channel,output_dim_lst[i],kernel_size=1,stride=1,padding=0),
                        nn.LeakyReLU(0.02,True),
                    )
                )
            self.featmap_convs.append(
                nn.Sequential(
                        nn.Conv2d(down_channel,down_channel, kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(0.02,True),
                        nn.Conv2d(down_channel,32,kernel_size=1,stride=1,padding=0),
                        nn.LeakyReLU(0.02,True),
                    )
                )

        self.dip_type = dip_type
        if self.dip_type == 'single':
            self.latent_single_conv  = nn.Sequential(
                    nn.Conv2d(sum(output_dim_lst),sum(output_dim_lst) // 2,kernel_size=1,stride=1,padding=0),
                    nn.LeakyReLU(0.02,True),
                    nn.Conv2d(sum(output_dim_lst) // 2,output_dim_lst[i],kernel_size=1,stride=1,padding=0),
                    nn.LeakyReLU(0.02,True),
            )
            self.featmap_single_conv = nn.Sequential(
                    nn.Conv2d(32 * 4,32 * 2,kernel_size=1,stride=1,padding=0),
                    nn.LeakyReLU(0.02,True),
                    nn.Conv2d(32 * 2,32,kernel_size=1,stride=1,padding=0),
                    nn.LeakyReLU(0.02,True),
            )

    def forward(self,x:Tensor) -> Tuple[List[Tensor],List[Tensor]]:
        """

        Args:
            x (Tensor): B 3 H W 

        Returns:
            List[Tensor]: 用于后续各个Filter的参数预测器
            List[Tensor]: 额外返回 原本的 feature map , 方便后续可以随时调用

        """

        # suppose x : [1,3,418,418]
        layer0 = self.backbone.conv1(x)        # [1, 64, 209, 209]
        layer0 = self.backbone.bn1(layer0)     # [1, 64, 209, 209]
        layer0 = self.backbone.relu(layer0)    # [1, 64, 209, 209]
        layer0 = self.backbone.maxpool(layer0) # [1, 64, 105, 105]

        layer1 = self.backbone.layer1(layer0)  # [1, 256, 105, 105]
        layer2 = self.backbone.layer2(layer1)  # [1, 512, 53 , 53 ]
        layer3 = self.backbone.layer3(layer2)  # [1, 1024, 27, 27 ]
        layer4 = self.backbone.layer4(layer3)  # [1, 2048, 14, 14 ]

        latent_out_lst = []   # Global latent vectors for filter's parameter predictor
        featmap_lst    = [] # Raw multi-scale feature maps from the backbone
        

        b,c,h,w = layer1.shape 
        for i,layer in enumerate([layer1,layer2,layer3,layer4]):
            latent = self.latent_convs[i](layer)
            latent_out_lst.append(latent) # fix output channels
            
            feat_map = self.featmap_convs[i](layer) # [1,32, x,x]
            feat_map = F.interpolate(
                feat_map,
                size=(h,w),
                mode='bilinear',
                align_corners=False
            )
            featmap_lst.append(feat_map) # [1,32,105,105]

        if self.dip_type == 'single':
            latent_out = torch.cat(latent_out_lst,dim=1)
            latent_out = self.latent_single_conv(latent_out)
            latent_out_lst = [latent_out]

            featmap = torch.cat(featmap_lst,dim=1)
            featmap = self.featmap_single_conv(featmap)
            featmap_lst = [featmap]


        return latent_out_lst,featmap_lst