#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     __init__.py
@Time     :     2025/12/08 17:04:06
@Author   :     Louis Swift
@Desc     :     
'''


from .bezier import * 
from .kernel import * 
from .defog  import * 
from .sharpen import *
from .identity import * 

__all__ = [
    'build_filters'
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
        filter_config,
        **kwargs
    ):

    assert filter_name in filters_map.keys(), f'filter name {filter_name} not in {filters_map.keys()}.'
    
    if filter_name == 'identity':
        filter_config = {}

    assert filter_config is not None, 'filter config is None.'
    
    return filters_map[filter_name](**filter_config)
