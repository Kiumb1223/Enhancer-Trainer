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

from typing import List

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

class PerceptualLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()

        self.alpha = alpha
        self.l1 = nn.L1Loss()

        # 使用 VGG16 提取特征
        vgg = vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        self.model = vgg[:16].eval()

        for p in self.model.parameters():
            p.requires_grad = False

        MEAN = torch.as_tensor([0.485, 0.456, 0.406],dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        STD  = torch.as_tensor([0.229, 0.224, 0.225],dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        self.register_buffer('mean', MEAN)
        self.register_buffer('std', STD)

    def extract(self, x):
        feats = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if i in ["3", "8", "15"]:   # relu1_2, relu2_2, relu3_3
                feats.append(x)
        return feats
    
    def forward(self, out, gt):

        # 1. extract the feats
        f_out = self.extract(out)
        f_gt  = self.extract(gt)

        # 2. calc the loss 
        loss = 0
        for o, g in zip(f_out, f_gt):
            pos = self.l1(o, g.detach())         # out 靠近 gt
            # neg = self.l1(o, n.detach())         # out 远离 damaged
            # loss += pos - self.alpha * neg
            loss += pos 

        return loss

class Criterion(nn.Module):
    def __init__(
            self,
            bt_multi_dips:bool, 
            lam_l1:float,
            lam_msssim:float,
            lam_color:float,
            lam_per:float,
            lam_gate_ent:float,
            lam_gate_rgb:float,
            **kwargs
        ):
        super().__init__()

        self.bt_multi_dips = bt_multi_dips

        self.per_loss = PerceptualLoss()

        self.weight_sl1    = lam_l1
        self.weight_msssim = lam_msssim
        self.weight_color  = lam_color
        self.weight_per    = lam_per

        # for gate regularization
        self.weight_gate_ent = lam_gate_ent
        self.weight_gate_rgb = lam_gate_rgb


    def ssim_loss(self,pred:Tensor,gt:Tensor):
        return 1 - ms_ssim(pred,gt,data_range=1,size_average=True).mean()

   
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
        loss = (1.0 - cos).mean()
        return loss

    def gate_entropy_regularization(self,g:Tensor):
        """
        g: [B, N, 3]  softmax-normalized gate
        lambda_ent: weight of entropy regularization

        Returns:
            scalar loss
        """
        eps = 1e-8
        # per-sample entropy over (N,3), then average over batch
        ent = - (g * (g.clamp_min(eps).log())).sum(dim=(1, 2)).mean()
        return ent
    
    def gate_rgb_consistency_regularization(self,g:Tensor):
        """
        g: [B, N, 3]
        lambda_rgb: weight of RGB consistency regularization

        Returns:
            scalar loss
        """
        g_R = g[..., 0]  # [B, N]
        g_G = g[..., 1]
        g_B = g[..., 2]

        diff_RG = (g_R - g_G).abs()
        diff_GB = (g_G - g_B).abs()

        rgb_cons = (diff_RG + diff_GB).mean()  # average over batch and filters
        return rgb_cons

    def forward(
            self,
            pred_lst:List[Tensor],
            gate_lst:List[Tensor],
            gt:Tensor,
            damaged:Tensor
        ):

        smooth_loss_l1 = 0
        msssim_loss    = 0
        color_loss     = 0
        per_loss       = 0

        # for gate regularization
        gate_ent_loss  = 0 
        gate_rgb_loss  = 0

        if self.bt_multi_dips:
            for i,(pred,gate) in enumerate(zip(pred_lst,gate_lst)):
                smooth_loss_l1 += F.smooth_l1_loss(pred,gt)
                msssim_loss    += self.ssim_loss(pred,gt)
                color_loss     += self.color_angle_loss(pred,gt)
                per_loss       += self.per_loss(pred,gt)

                gate = gate.squeeze(-1).squeeze(-1)
                gate_ent_loss   += self.gate_entropy_regularization(gate)
                gate_rgb_loss  += self.gate_rgb_consistency_regularization(gate)
        else:
            # only calc the last output of DIP
            pred = pred_lst[-1]
            smooth_loss_l1 += F.smooth_l1_loss(pred,gt)
            msssim_loss    += self.ssim_loss(pred,gt)
            color_loss     += self.color_angle_loss(pred,gt)
            per_loss       += self.per_loss(pred,gt)

            for gate in gate_lst:
                gate = gate.squeeze(-1).squeeze(-1)
                gate_ent_loss   += self.gate_entropy_regularization(gate)
                gate_rgb_loss  += self.gate_rgb_consistency_regularization(gate)


        total_loss = (
            self.weight_sl1      * smooth_loss_l1 +
            self.weight_msssim   * msssim_loss    +
            self.weight_color    * color_loss     +
            self.weight_per      * per_loss       +
            self.weight_gate_ent * gate_ent_loss  +
            self.weight_gate_rgb * gate_rgb_loss  
        )

        loss_dict = {
            'l1':smooth_loss_l1,
            'msssim':msssim_loss,
            'color':color_loss,
            'perceptual':per_loss,
            'gate_entropy':gate_ent_loss,
            'gate_rgb':gate_rgb_loss,
            'dip':total_loss,
        }
        return total_loss, loss_dict
    
