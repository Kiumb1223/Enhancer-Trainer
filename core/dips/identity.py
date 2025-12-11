#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     identity.py
@Time     :     2025/12/08 17:15:46
@Author   :     Louis Swift
@Desc     :     
'''


import torch.nn as nn 

__all__ = [
    'Identity'
]

class Identity(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()

    def forward(self, x,latent_out):
        return x