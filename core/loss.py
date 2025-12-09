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

import logging 
import torchvision 

import torch.nn as nn 
from torch import Tensor
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
from kornia.filters import sobel

# 数据预处理的时候 就简单 /255 进行归一化
# 所以 再喂入 VGG 网络 之前，进一步标准化
MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
STD  = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

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

        self.register_buffer('MEAN',MEAN)
        self.register_buffer('STD',STD)

    def extract(self, x):
        feats = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if i in ["3", "8", "15"]:   # relu1_2, relu2_2, relu3_3
                feats.append(x)
        return feats

    def do_norm_input(self,img_norm):
        return (img_norm - self.MEAN) / self.STD

    def forward(self, out, gt, damaged):
        
        # 1. normalize the input 
        out     = self.do_norm_input(out)
        gt      = self.do_norm_input(gt)
        damaged = self.do_norm_input(damaged)
        # 2. extract the feats
        f_out = self.extract(out)
        f_gt  = self.extract(gt)
        f_neg = self.extract(damaged)

        # 3. calc the loss 
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
            lam_text:float,
            lam_color:float,
            lam_con:float,
            **kwargs
        ):
        super().__init__()
        self.con_loss = ContrastLoss()

        self.weight_sl1    = lam_l1
        self.weight_msssim = lam_msssim
        self.weight_text   = lam_text
        self.weight_color  = lam_color
        self.weight_con    = lam_con


    def ssim_loss(self,pred:Tensor,gt:Tensor):
        return 1 - ms_ssim(pred,gt,data_range=1,size_average=True).mean()

    def texture_loss(self,pred:Tensor,gt:Tensor):
        edge_pred = sobel(pred)
        edge_gt   = sobel(gt)

        loss = F.l1_loss(edge_pred,edge_gt)
        return loss 
    
    def color_angle_loss(self,pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
        B, C, H, W = pred.shape
        assert C == 3

        # (B, C, H*W)
        a = pred.view(B, C, -1)
        b = gt.view(B, C, -1)

        up = (a * b).sum(dim=2)  # (B, C)
        a_norm = a.pow(2).sum(dim=2).sqrt().clamp_min(eps)  # (B, C)
        b_norm = b.pow(2).sum(dim=2).sqrt().clamp_min(eps)  # (B, C)

        cos = (up / (a_norm * b_norm)).clamp(-1.0, 1.0)  # (B, C)
        theta = torch.acos(cos)  # (B, C)

        loss = theta.mean()  # 对 B 和 C 求平均
        return loss
    def color_loss(self, pred: Tensor, gt: Tensor ,eps = 1e-8):
        """
        颜色一致性损失（基于 RGB 向量夹角）

        pred: (B, 3, H, W) 增强图
        gt  : (B, 3, H, W) 真值图
        返回: 标量损失
        """
        # 展平到 (B, 3, N)，N = H * W，方便按像素计算向量相似度
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1)  # (B, 3, N)
        gt_flat   = gt.view(B, C, -1)    # (B, 3, N)

        # 计算每个像素的 RGB 向量 L2 范数 (B, 1, N)
        pred_norm = torch.norm(pred_flat, dim=1, keepdim=True)  # (B, 1, N)
        gt_norm   = torch.norm(gt_flat,   dim=1, keepdim=True)  # (B, 1, N)

        # 避免除零
        pred_norm = pred_norm.clamp_min(eps)
        gt_norm   = gt_norm.clamp_min(eps)

        # 单位向量
        pred_unit = pred_flat / pred_norm  # (B, 3, N)
        gt_unit   = gt_flat   / gt_norm    # (B, 3, N)

        # 余弦相似度：对通道维度求内积
        cos_sim = (pred_unit * gt_unit).sum(dim=1)  # (B, N)

        # 由于数值精度问题，cos_sim 可能略超出 [-1, 1]，做一下截断
        cos_sim = cos_sim.clamp(-1.0, 1.0)

        # 颜色损失：1 - cos，cos 越接近 1，损失越小
        loss = (1.0 - cos_sim).mean()  # 标量

        return loss
            

    def forward(self, pred,gt,damaged):
        
        smooth_loss_l1 = F.smooth_l1_loss(pred,gt)
        msssim_loss    = self.ssim_loss(pred,gt)
        text_loss      = self.texture_loss(pred,gt)
        color_loss     = self.color_angle_loss(pred,gt)
        con_loss       = self.con_loss(pred,gt,damaged)

        total_loss = (
            self.weight_sl1    * smooth_loss_l1 +
            self.weight_msssim * msssim_loss    +
            self.weight_text   * text_loss      +
            self.weight_color  * color_loss     +
            self.weight_con    * con_loss
        )

        loss_dict = {
            'l1':smooth_loss_l1,
            'msssim':msssim_loss,
            'color':color_loss,
            'texture':text_loss,
            'contrast':con_loss,
            'dip':total_loss,
        }
        return total_loss, loss_dict