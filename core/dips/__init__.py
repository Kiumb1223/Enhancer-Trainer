#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     __init__.py
@Time     :     2025/12/08 17:04:06
@Author   :     Louis Swift
@Desc     :     创建 Filter && Gate
'''

# 1. gate module 
from .gate import *

# 2. ISP modules
from .bezier import * 
from .kernel import * 
from .defog  import * 
from .sharpen import *
from .identity import * 

from omegaconf import OmegaConf
from typing import Optional

__all__ = [
    'build_filters',
    'build_gate',
]

filters_map = {
    'bezier': Bezier,
    'defog': Defog,
    'kernel': Kernel,
    'sharpen': Sharpen,
    'identity': Identity,
}

def build_filters(
        filter_name,
        filter_config:Optional[OmegaConf],
        **kwargs
    ):

    assert filter_name in filters_map.keys(), f'filter name {filter_name} not in {filters_map.keys()}.'
    
    if filter_name == 'identity':
        filter_config = {}

    return filters_map[filter_name](**filter_config)

def build_gate(
        gate_config:OmegaConf,
        **kwargs
):
    assert gate_config is not None, 'gate config is None.'
    
    return GateModule(**gate_config)