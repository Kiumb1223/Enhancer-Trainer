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

import cv2 
import random 
import logging 
import numpy as np 
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self,
            txt_path: str, # the directory of dataset 
            input_shape: list,
            data_type:str= 'train',
            **kwargs
        ):

        super().__init__()
        
        assert data_type in ['train','val'], 'data_type must be in [train, val]'

        self.data_type = data_type


        with open(txt_path,'r') as f:
            self.data_lst = f.readlines()
            
        logging.info(f'load {len(self.data_lst)} images from {txt_path}.')

        if self.data_type == 'train':
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
                    A.Resize(size=input_shape, p=1.0),
                ],
                additional_targets={'gt':'image'}
            )
            
        self.norm_transform = A.Compose([
                A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0, p=1.0),
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

