#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     bezier.py
@Time     :     2025/12/08 15:16:53
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn 
from torch import Tensor

__all__ = [
    'Bezier'
]

class Bezier(nn.Module):
    def __init__(
            self,
            L:int,
            num_of_params:int,
            encoder_output_dim:int,
            **kwargs
        ):
        super().__init__()

        self.params_predictor = nn.Sequential(
            nn.Conv2d(encoder_output_dim,encoder_output_dim//2,3,2,1),
            nn.LeakyReLU(0.02,True),
            nn.Conv2d(encoder_output_dim//2,num_of_params,3,2,1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_of_params,num_of_params,1,1,0),
            nn.Tanh() # [-1,1]
        )

        # initialization weight
        self.params_predictor[-2].weight.data.zero_()
        self.params_predictor[-2].bias.data.zero_()


        self.register_buffer("q_buf", torch.linspace(0,1,L+1).view(1,1,L+1))
        self.register_buffer("B1_buf", 3*(1-self.q_buf)**2 * self.q_buf)
        self.register_buffer("B2_buf", 3*(1-self.q_buf) * self.q_buf**2)
        self.register_buffer("B3_buf", self.q_buf**3)

    def forward(self, x:Tensor,latent_out:Tensor) -> Tensor:

        B, C, H, W = x.shape
        assert C == 3, "BPW 目前只支持 3 通道 RGB"

        # ---------- 1. 预测参数 r1, theta1, r2, theta2 ----------
        params = self.params_predictor(latent_out).squeeze() # [B, 12], in [-1,1]
        params = params.view(B, 3, 4)                        # [B, 3, 4]
        r1, theta1, r2, theta2 = params.unbind(dim=-1)       # 各 [B,3]

        # ---------- 2. 控制点 p1, p2 ----------
        p1_radius = (r1 + 1.0) * 0.5                         # [B,3], ∈ [0,1]
        p2_radius = (r2 + 1.0) * 0.5

        # 论文中两个角度都是 (theta+1)*π/4
        ang1 = (theta1 + 1.0) * torch.pi / 4.0               # [B,3]
        ang2 = (theta2 + 1.0) * torch.pi / 4.0

        p1_x = p1_radius * torch.cos(ang1)                   # [B,3]
        p1_y = p1_radius * torch.sin(ang1)
        p2_x = 1.0 - p2_radius * torch.cos(ang2)
        p2_y = 1.0 - p2_radius * torch.sin(ang2)

        # 扩一维用于与 [1,1,L+1] 的基函数广播
        p1_x = p1_x.unsqueeze(-1)                            # [B,3,1]
        p1_y = p1_y.unsqueeze(-1)
        p2_x = p2_x.unsqueeze(-1)
        p2_y = p2_y.unsqueeze(-1)

        # ---------- 3. 使用预计算好的 Bezier 基函数 ----------
        B1 = self.B1_buf    # [1,1,L+1]
        B2 = self.B2_buf
        B3 = self.B3_buf

        # P0=(0,0), P3=(1,1)
        # Cx,Cy: [B,3,L+1]
        Cx = B1 * p1_x + B2 * p2_x + B3
        Cy = B1 * p1_y + B2 * p2_y + B3

        # ---------- 4. 分段线性近似 ----------
        dCx = Cx[..., 1:] - Cx[..., :-1]             # [B,3,L]
        dCy = Cy[..., 1:] - Cy[..., :-1]             # [B,3,L]
        CPi = Cx[..., :-1]                           # 左端点 [B,3,L]

        CPi = CPi.unsqueeze(-1).unsqueeze(-1)        # [B,3,L,1,1]
        dCx = dCx.unsqueeze(-1).unsqueeze(-1)        # [B,3,L,1,1]
        dCy = dCy.unsqueeze(-1).unsqueeze(-1)        # [B,3,L,1,1]

        eps = 1e-6
        dCx_pos = dCx.clamp_min(eps)                 # 避免除 0

        # 输入像素: [B,3,1,H,W]
        Pi = x.unsqueeze(2)                          # [B,3,1,H,W]

        # 分两步 clip: 先 min=0，再 max=dCx_pos
        diff = (Pi - CPi).clamp_min(0.0)             # >=0
        t = torch.minimum(diff, dCx_pos)             # <= ΔCP_i   [B,3,L,H,W]

        # slope = ΔCP_o / ΔCP_i
        slope = dCy / dCx_pos                        # [B,3,L,1,1]
        slope = slope.squeeze(-1).squeeze(-1)        # [B,3,L]

        # out_curve = Σ t * slope over L
        out = torch.einsum("bclhw,bcl->bchw", t, slope)  # [B,3,H,W]
        # out = (out - out.min()) / (out.max() - out.min())
        # out = torch.clamp(out,min=0.0,max=1.0)
        # out = x + 0.2 * (out - x)
        
        return out
