#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     dataset.py
@Time     :     2025/12/01 15:05:33
@Author   :     Louis Swift
@Desc     :     

        The structure of CustomDataset:
            data_dir 
                | --- JPEGImages   # save the original images 
                |
                | --- dark # save the dark images
                |
                | --- fog  # save the fog images
'''

import os 
import cv2 
import glob 
import random 
import logging 
import numpy as np 
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class VOCDataset(Dataset):
    def __init__(self,
            txt_path: str, # the directory of dataset 
            input_shape: list,
            phase:str= 'train',
            **kwargs
        ):

        super().__init__()
        
        assert phase in ['train','val'], 'data_type must be in [train, val]'

        with open(txt_path,'r') as f:
            self.data_lst = f.readlines()
            
        logging.info(f'load {len(self.data_lst)} images from {txt_path}.')

        if phase == 'train':
            self.geo_transform = A.Compose([
                    A.RandomResizedCrop(size=input_shape, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=5,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                ],
                additional_targets={'gt':'image'}  
            )
        else:
            self.geo_transform = A.Compose([
                    A.Resize(height=input_shape[0],width=input_shape[1], p=1.0),
                ],
                additional_targets={'gt':'image'}
            )
            
        self.norm_transform = A.Compose([
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                A.ToFloat(), # the same as  'image / 255'
                ToTensorV2(),
            ],
            additional_targets={'gt':'image'}  
        )

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        gt_path = self.data_lst[idx].strip() # 去除 \n
        
        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        gt = np.asarray(gt,dtype=np.uint8)

        mode = random.choice(['dark','fog'])
        damaged_img_path = gt_path.replace('JPEGImages',mode)

        damaged = cv2.imread(damaged_img_path) 
        damaged = cv2.cvtColor(damaged,cv2.COLOR_BGR2RGB)
        damaged = np.asarray(damaged,dtype=np.uint8)

        # -------------------------
        # 1. 几何增强
        augmented = self.geo_transform(image=damaged, gt=gt)

        damaged = augmented['image']
        gt = augmented['gt']
        
        # -------------------------
        # 2. 归一化 + tensor
        out = self.norm_transform(image=damaged, gt=gt)

        damaged = out['image']
        gt = out['gt']

        return damaged, gt



class CddDataset(Dataset):
    """
        The Structure of CddDataset:
            data_dir
                | --- clear          # save the clear images
                | --- haze           # save the haze images
                | --- low            # save the low images
                | --- low + haze     # save the low + haze images
    """
    def __init__(self,
            data_dir,
            modes:list,
            input_shape: list,
            phase = 'Train',
            **kwargs
        ):
        super().__init__()

        self.data_lst = glob.glob(os.path.join(data_dir,'clear','*.png'))

        logging.info(f'load {len(self.data_lst)} images from {data_dir}.')

        self.modes = modes
        self.input_shape = input_shape

        if phase == 'train':
            self.geo_transform = A.Compose([
                    A.RandomResizedCrop(size=input_shape, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=5,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                ],
                additional_targets={'gt':'image'}  
            )
        else:
            self.geo_transform = A.Compose([
                    A.Resize(height=input_shape[0],width=input_shape[1], p=1.0),
                ],
                additional_targets={'gt':'image'}
            )
            
        self.norm_transform = A.Compose([
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                A.ToFloat(), # the same as  'image / 255'
                ToTensorV2(),
            ],
            additional_targets={'gt':'image'}  
        )

    def __len__(self):
        return len(self.data_lst)
    

    def __getitem__(self, idx):
        gt_path = self.data_lst[idx].strip() # 去除 \n

        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        gt = np.asarray(gt,dtype=np.uint8)

        mode = random.choice(self.modes)
        damaged_img_path = gt_path.replace('clear',mode)

        damaged = cv2.imread(damaged_img_path)
        damaged = cv2.cvtColor(damaged,cv2.COLOR_BGR2RGB)
        damaged = np.asarray(damaged,dtype=np.uint8)

        # -------------------------
        # 1. 几何增强
        augmented = self.geo_transform(image=damaged, gt=gt)

        damaged = augmented['image']
        gt = augmented['gt']

        # -------------------------
        # 2. 归一化 + tensor
        out = self.norm_transform(image=damaged, gt=gt)

        damaged = out['image']
        gt = out['gt']

        return damaged, gt
    