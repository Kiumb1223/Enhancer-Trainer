#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     loss.py
@Time     :     2025/12/01 14:30:54
@Author   :     Louis Swift
@Desc     :     

            MSE loss 
            perceptual loss 
            SSIM loss 
'''

import torch 
import logging 
import torchvision 

import torch.nn as nn 
from torch import Tensor
import kornia.color as color
import torch.nn.functional as F 
from pytorch_msssim import ms_ssim

logger = logging.getLogger(__name__)


__all__ = [
    'Criterion'
]

"""
# --------------------------------------------------------------------------------------------------------------------------
# Extracted Lama https://github.com/advimman/lama/tree/main

def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        logger.warn(f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}")

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class PerceptualLoss(nn.Module):
    def __init__(self, normalize_inputs=True):
        super(PerceptualLoss, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD

        vgg = torchvision.models.vgg19().features
        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')

        # we expect input and target to be in [0, 1] range
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target

        for layer in self.vgg[:30]:

            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                loss = F.mse_loss(features_input, features_target, reduction='none')

                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:],
                                             mode='bilinear', align_corners=False)
                    loss = loss * (1 - cur_mask)

                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)

        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):
        check_and_warn_input_range(input, 0, 1, 'PerceptualLoss input in get_global_features')

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input

        features_input = self.vgg(features_input)
        return features_input

# --------------------------------------------------------------------------------------------------------------------------



class Criterion(nn.Module):
    def __init__(self,
            lam_mse:float,         
            lam_ssim:float,         
            lam_color:float,         
            lam_perceptual:float,         
        ):
        super().__init__()

        self.lam_mse = lam_mse
        self.lam_ssim = lam_ssim
        self.lam_color = lam_color
        self.lam_perceptual = lam_perceptual

        self.perceptual_loss = PerceptualLoss() 
        
    def ssim_loss(self,pred:Tensor,gt:Tensor):
        return 1 - ms_ssim(pred,gt,data_range=1,size_average=True).mean()
    
    def mse_loss(self,pred:Tensor,gt:Tensor):
        return F.l1_loss(pred,gt)
    
    def color_loss(self,pred:Tensor,gt:Tensor,weight_l=1.,weight_ab=1.):
        lab_pred = color.rgb_to_lab(pred)
        lab_gt = color.rgb_to_lab(gt)

        # Normalize L to [0, 1], a/b to [-1, 1]
        lab_pred_norm = torch.cat([
            lab_pred[:, 0:1] / 100.0,       # L ∈ [0, 1]
            lab_pred[:, 1:2] / 128.0,       # a ∈ [-1, 1]
            lab_pred[:, 2:3] / 128.0        # b ∈ [-1, 1]
        ], dim=1)

        lab_gt_norm = torch.cat([
            lab_gt[:, 0:1] / 100.0,
            lab_gt[:, 1:2] / 128.0,
            lab_gt[:, 2:3] / 128.0
        ], dim=1)

        l_diff = F.l1_loss(lab_pred_norm[:, 0:1], lab_gt_norm[:, 0:1])
        ab_diff = F.l1_loss(lab_pred_norm[:, 1:], lab_gt_norm[:, 1:])

        return weight_l * l_diff + weight_ab * ab_diff
    
    def forward(self,pred:Tensor,gt:Tensor):
        mse = self.mse_loss(pred,gt)
        ssim = self.ssim_loss(pred,gt)
        color = self.color_loss(pred,gt)
        perceptual = self.perceptual_loss(pred,gt).sum()


        total_loss = self.lam_mse * mse + \
                    self.lam_ssim * ssim + \
                    self.lam_color * color + \
                    self.lam_perceptual * perceptual
        
        return total_loss, \
                {
                    'mse':mse,
                    'ssim':ssim,
                    'color':color,
                    'perceptual':perceptual,
                    'total':total_loss,
                }
"""


import torch
import torchvision
from math import exp
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class ContrastLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()

        self.alpha = alpha
        self.l1 = nn.L1Loss()

        # 使用 VGG16 提取特征
        vgg = vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        self.model = vgg[:16].eval()

        for p in self.model.parameters():
            p.requires_grad = False

    def extract(self, x):
        feats = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if i in ["3", "8", "15"]:   # relu1_2, relu2_2, relu3_3
                feats.append(x)
        return feats

    def forward(self, out, gt, damaged):
        f_out = self.extract(out)
        f_gt = self.extract(gt)
        f_neg = self.extract(damaged)

        loss = 0
        for o, g, n in zip(f_out, f_gt, f_neg):
            pos = self.l1(o, g.detach())         # out 靠近 gt
            neg = self.l1(o, n.detach())         # out 远离 damaged
            loss += pos - self.alpha * neg

        return loss

class Criterion(nn.Module):
    def __init__(
            self,
            lam_l1:float,
            lam_msssim:float,
            lam_con:float,
            **kwargs
        ):
        super().__init__()
        self.con_loss = ContrastLoss()
        self.weight_sl1 = lam_l1
        self.weight_msssim = lam_msssim
        self.weight_com = lam_con
    def ssim_loss(self,pred:Tensor,gt:Tensor):
        return 1 - ms_ssim(pred,gt,data_range=1,size_average=True).mean()
    
    def forward(self, pred,gt,damaged):
        smooth_loss_l1 = F.smooth_l1_loss(pred,gt)
        msssim_loss = self.ssim_loss(pred,gt)
        c_loss = self.con_loss(pred,gt,damaged)

        total_loss = (
            self.weight_sl1 * smooth_loss_l1 +
            self.weight_msssim * msssim_loss +
            self.weight_com * c_loss
        )

        loss_dict = {
            'l1':smooth_loss_l1,
            'msssim':msssim_loss,
            'contrast':c_loss,
            'total':total_loss,
        }
        return total_loss, loss_dict