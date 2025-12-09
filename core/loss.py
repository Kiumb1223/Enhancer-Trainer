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


import torch
import torchvision
from math import exp
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from kornia.filters import sobel

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

        # 1. extract the feats
        f_out = self.extract(out)
        f_gt  = self.extract(gt)
        f_neg = self.extract(damaged)

        # 2. calc the loss 
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
        color_loss     = self.color_loss(pred,gt)
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
    


# class Criterion(nn.Module):
#     def __init__(
#             self,
#             lam_l1:float,
#             lam_msssim:float,
#             lam_text:float,
#             lam_color:float,
#             lam_con:float,
#             **kwargs
#         ):
#         super().__init__()
#         self.con_loss = ContrastLoss()

#         self.weight_sl1    = lam_l1
#         self.weight_msssim = lam_msssim
#         self.weight_text   = lam_text
#         self.weight_color  = lam_color
#         self.weight_con    = lam_con


#     def ssim_loss(self,pred:Tensor,gt:Tensor):
#         return 1 - ms_ssim(pred,gt,data_range=1,size_average=True).mean()

#     def texture_loss(self,pred:Tensor,gt:Tensor):
#         edge_pred = sobel(pred)
#         edge_gt   = sobel(gt)

#         loss = F.l1_loss(edge_pred,edge_gt)
#         return loss 
    
#     def color_loss(self, pred: Tensor, gt: Tensor ,eps = 1e-8):
#         """
#         颜色一致性损失（基于 RGB 向量夹角）

#         pred: (B, 3, H, W) 增强图
#         gt  : (B, 3, H, W) 真值图
#         返回: 标量损失
#         """
#         # 展平到 (B, 3, N)，N = H * W，方便按像素计算向量相似度
#         B, C, H, W = pred.shape
#         pred_flat = pred.view(B, C, -1)  # (B, 3, N)
#         gt_flat   = gt.view(B, C, -1)    # (B, 3, N)

#         # 计算每个像素的 RGB 向量 L2 范数 (B, 1, N)
#         pred_norm = torch.norm(pred_flat, dim=1, keepdim=True)  # (B, 1, N)
#         gt_norm   = torch.norm(gt_flat,   dim=1, keepdim=True)  # (B, 1, N)

#         # 避免除零
#         pred_norm = pred_norm.clamp_min(eps)
#         gt_norm   = gt_norm.clamp_min(eps)

#         # 单位向量
#         pred_unit = pred_flat / pred_norm  # (B, 3, N)
#         gt_unit   = gt_flat   / gt_norm    # (B, 3, N)

#         # 余弦相似度：对通道维度求内积
#         cos_sim = (pred_unit * gt_unit).sum(dim=1)  # (B, N)

#         # 由于数值精度问题，cos_sim 可能略超出 [-1, 1]，做一下截断
#         cos_sim = cos_sim.clamp(-1.0, 1.0)

#         # 颜色损失：1 - cos，cos 越接近 1，损失越小
#         loss = (1.0 - cos_sim).mean()  # 标量

#         return loss
            

    def forward(self, pred_lst,gt,damaged):

        smooth_loss_l1 = 0
        msssim_loss    = 0
        text_loss      = 0
        color_loss     = 0
        con_loss       = 0
        for i in range(len(pred_lst)):
            smooth_loss_l1 += F.smooth_l1_loss(pred_lst[i],gt)
            msssim_loss    += self.ssim_loss(pred_lst[i],gt)
            text_loss      += self.texture_loss(pred_lst[i],gt)
            color_loss     += self.color_loss(pred_lst[i],gt)
            con_loss       += self.con_loss(pred_lst[i],gt,damaged)

        # smooth_loss_l1 = F.smooth_l1_loss(pred,gt)
        # msssim_loss    = self.ssim_loss(pred,gt)
        # text_loss      = self.texture_loss(pred,gt)
        # color_loss     = self.color_loss(pred,gt)
        # con_loss       = self.con_loss(pred,gt,damaged)

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
    
