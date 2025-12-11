#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     __init__.py
@Time     :     2025/11/21 21:14:23
@Author   :     Louis Swift
@Desc     :     
'''

from .enhancer import * 
from .loss import   *


__all__ = [
    'build_enhancer',
    'build_dip_loss'
]

def build_enhancer(cfg_enhancer):
    return Enhancer(**cfg_enhancer)

def build_dip_loss(cfg_loss):
    return Criterion(**cfg_loss)