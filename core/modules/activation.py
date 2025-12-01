#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     activation.py
@Time     :     2025/11/28 17:25:15
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn 


def gen_activation(type:str):
    if type == 'lrelu':
        return nn.LeakyReLU
    elif type == 'telu':
        return TeLU
    else:
        raise NotImplementedError(f'activation {type} not implemented')



class TeLU(nn.Module):
    """Reference : https://arxiv.org/pdf/2412.20269"""
    def __init__(self,*args,**kwargs): # filter out the relevant params
        super().__init__()
    
    def forward(self,x):
        return x * torch.tanh( torch.exp(x) )
