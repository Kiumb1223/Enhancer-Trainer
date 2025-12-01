#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Extracted from https://github.com/serend1p1ty/core-pytorch-utils, with minor modification
and much thanks for their brilliant works~
'''

import os
import sys
import yaml 
import torch
import random
import numpy as np
# from loguru import logger 
import logging
from typing import Optional
from tabulate import tabulate
from omegaconf import OmegaConf
from collections import defaultdict
from easydict import EasyDict as edict


logger = logging.getLogger(__name__)

__all__ = [
    "collect_env",
    "get_model_info",
    "get_model_configuration",
    "get_exp_info",
    "set_random_seed",
    "symlink",
]

def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The value of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - PyTorch: PyTorch version.
        - TorchVision (optional): TorchVision version.
        - Pytorch-geometric (optional): Pytorch-geometric version.

    Returns:
        str: A string describing the running environment.
    """
    env_info = []
    env_info.append(("sys.platform", sys.platform))
    env_info.append(("Python", sys.version.replace("\n", "")))
    env_info.append(("Numpy", np.__version__))

    cuda_available = torch.cuda.is_available()
    env_info.append(("CUDA available", cuda_available))

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info.append(("GPU " + ",".join(device_ids), name))

    env_info.append(("PyTorch", torch.__version__))

    try:
        import torchvision
        env_info.append(("TorchVision", torchvision.__version__))
    except ModuleNotFoundError:
        pass

    return tabulate(env_info,tablefmt="grid")

def get_model_info(model:torch.nn.Module) -> str:
    '''get model information'''
    model_info = []
    total_params = 0

    # 遍历模型的每个子模块并计算其参数数量
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        model_info.append([name, f'{module_params / 1e6:.2f}M'])
        total_params += module_params

    # 添加总参数数量
    model_info.append(['Total Parameters', f'{total_params / 1e6:.2f}M'])

    return tabulate(model_info,tablefmt="grid")



def get_exp_info(cfg:OmegaConf) -> str:
    '''get all experimental information for better debugging'''
    return OmegaConf.to_yaml(cfg)

def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): If None or negative, use a generated seed.
        deterministic (bool): If True, set the deterministic option for CUDNN backend.
    """
    if seed is None or seed < 0:
        new_seed = np.random.randint(2**32)
        logger.info(f"Got invalid seed: {seed}, will use the randomly generated seed: {new_seed}")
        seed = new_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("The CUDNN is set to deterministic. This will increase reproducibility, "
                    "but may slow down your training considerably.")


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    """Create a symlink, dst -> src.

    Args:
        src (str): Path to source.
        dst (str): Path to target.
        overwrite (bool): If True, remove existed target. Defaults to True.
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)

