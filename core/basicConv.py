#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     basic.py
@Time     :     2025/11/28 17:25:15
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn 
from collections import OrderedDict

__all__ = [
    'BasicConv', 
    'gen_activation' # can be deprecated 
]

def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if pad is None:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size, stride, pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("lrelu", nn.LeakyReLU(0.1,inplace=True)),
    ]))


def gen_activation(type:str):
    if type == 'lrelu':
        return nn.LeakyReLU(0.1,True)
    elif type == 'selu':
        return nn.SELU(True)
    elif type == 'relu':
        return nn.ReLU(True)
    elif type == 'telu':
        return TeLU()
    else:
        raise NotImplementedError(f'activation {type} not implemented')



class TeLU(nn.Module):
    """Reference : https://arxiv.org/pdf/2412.20269"""
    def __init__(self,*args,**kwargs): # filter out the relevant params
        super().__init__()
    
    def forward(self,x):
        return x * torch.tanh( torch.exp(x) )
