#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     gdip.py
@Time     :     2025/11/24 20:04:23
@Author   :     TZM12138 @ https://github.com/TZM12138  && Louis Swift 
@Desc     :     

            Mainly reference: https://github.com/Gatedip/GDIP-Yolo
            Thanks a lot for their brilliant works.

'''


import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from kornia.filters import guided_blur,gaussian_blur2d

__all__ = [
    'GatedDIP',
    'ProgressiveGatedDIP',
]

class GatedDIP(torch.nn.Module):
    def __init__(
            self,
            kernel_size = 7,
            num_of_gates = 2, # the number of filters
            encoder_output_dim = 256,
            **kwargs
        ):
        super().__init__()

        # Gating Module
        self.gate_module = nn.Sequential(
                nn.Conv2d(encoder_output_dim,encoder_output_dim,1,1,0),nn.SELU(),
                nn.Conv2d(encoder_output_dim,num_of_gates,kernel_size=1,stride=1,padding=0),
                nn.Tanh()
            )

        # Filter Modules
        # 1. bezier Filter
        self.bezier_module = nn.Sequential(
                nn.Conv2d(encoder_output_dim,encoder_output_dim,1,1,0),nn.SELU(),
                nn.Conv2d(encoder_output_dim,12,kernel_size=1,stride=1,padding=0),
                nn.Tanh()
            )
        L = 8 
        self.register_buffer("q_buf", torch.linspace(0,1,L+1).view(1,1,L+1))
        self.register_buffer("B1_buf", 3*(1-self.q_buf)**2 * self.q_buf)
        self.register_buffer("B2_buf", 3*(1-self.q_buf) * self.q_buf**2)
        self.register_buffer("B3_buf", self.q_buf**3)

        # 3. kernel Filter
        self.kernel_size = kernel_size 
        self.kernel_module = nn.Sequential(
                nn.Conv2d(encoder_output_dim,encoder_output_dim,1,1,0),nn.SELU(),
                nn.Conv2d(encoder_output_dim,6*self.kernel_size**2,kernel_size=1,stride=1,padding=0),
        )

    
    def identity(self,x : torch.tensor,identity_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            identity_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        x = x*identity_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return x
    

    def bezier(self, x: torch.Tensor, latent_out: torch.Tensor, bezier_gate: torch.Tensor) -> torch.Tensor:
        """
        改进版 BPW:
        - 严格按 ERUP-YOLO 的 Bezier + 分段线性近似
        - 使用缓存好的 Bezier 基函数
        - 残差式输出: out = x + gate * (curve(x) - x)，避免过曝
        """
        B, C, H, W = x.shape
        assert C == 3, "BPW 目前只支持 3 通道 RGB"

        # x = x.clamp(0.0, 1.0)

        # ---------- 1. 预测参数 r1, theta1, r2, theta2 ----------
        params = self.bezier_module(latent_out).squeeze() # [B, 12], in [-1,1]
        params = params.view(B, 3, 4)                # [B, 3, 4]
        r1, theta1, r2, theta2 = params.unbind(dim=-1)   # 各 [B,3]

        # ---------- 2. 控制点 p1, p2 ----------
        p1_radius = (r1 + 1.0) * 0.5                 # [B,3], ∈ [0,1]
        p2_radius = (r2 + 1.0) * 0.5

        # 论文中两个角度都是 (theta+1)*π/4
        ang1 = (theta1 + 1.0) * torch.pi / 4.0       # [B,3]
        ang2 = (theta2 + 1.0) * torch.pi / 4.0

        p1_x = p1_radius * torch.cos(ang1)           # [B,3]
        p1_y = p1_radius * torch.sin(ang1)
        p2_x = 1.0 - p2_radius * torch.cos(ang2)
        p2_y = 1.0 - p2_radius * torch.sin(ang2)

        # 扩一维用于与 [1,1,L+1] 的基函数广播
        p1_x = p1_x.unsqueeze(-1)                    # [B,3,1]
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
        out_curve = torch.einsum("bclhw,bcl->bchw", t, slope)  # [B,3,H,W]

        # ---------- 5. 残差式叠加 + gate ----------
        # instead of: out = out_curve * gate
        # 我们用: out = x + gate * (out_curve - x)
        # 这样当 gate 较小/初始时几乎是 identity，不容易把亮度拉爆
        gate = bezier_gate.view(B, 1, 1, 1)
        out = gate * (out_curve - x) + x 
        out = out.clamp(0.0, 1.0)
        return out
    


    def kernel(self, x, latent_out, kernel_gate):
        B, C, H, W = x.shape
        assert C == 3, "Only 3-channel supported"

        params = self.kernel_module(latent_out).squeeze()  # [B, 2*3*k*k] in [-1,1]
        params = params.view(B, 2, C, self.kernel_size, self.kernel_size)

        K1 = params[:, 0]  # [B, 3, k, k]
        K2 = params[:, 1]  # [B, 3, k, k]
        K1 = K1 / (K1.abs().sum(dim=[2,3], keepdim=True) + 1e-6)
        K2 = K2 / (K2.abs().sum(dim=[2,3], keepdim=True) + 1e-6)

        # x = torch.clamp(x, 0.0, 1.0)

        # --- Vectorized convolution without loop ---
        padding = self.kernel_size // 2

        # Reshape input: [B, 3, H, W] -> [1, B*3, H, W]
        x_flat = x.view(1, B * C, H, W)

        # Reshape kernels: [B, 3, k, k] -> [B*3, 1, k, k]
        K1_flat = K1.reshape(B * C, 1, self.kernel_size, self.kernel_size)
        K2_flat = K2.reshape(B * C, 1, self.kernel_size, self.kernel_size)

        # Grouped convolution: each of the B*3 "channels" uses its own 1x1 kernel
        conv1_flat = F.conv2d(x_flat, K1_flat, padding=padding, groups=B * C)
        conv2_flat = F.conv2d(x_flat, K2_flat, padding=padding, groups=B * C)

        # Reshape back: [1, B*3, H, W] -> [B, 3, H, W]
        conv1 = conv1_flat.view(B, C, H, W)
        conv2 = conv2_flat.view(B, C, H, W)

        # Final output
        output = x * conv1 + conv2 + x
        output = torch.clamp(output, 0.0, 1.0)
        output = output * kernel_gate.view(B, 1, 1, 1)

        return output

    def forward(self,x,feat_proj):

        gate = self.gate_module(feat_proj).squeeze()
        # gate = self.tanh_range(gate,0.01,1.0)
        gate = torch.clamp(gate,0.01,1.0)

        # identity_out = self.identity(x,gate[:,0])
        bezier_out = self.bezier(x,feat_proj,gate[:,0])
        kernel_out = self.kernel(x,feat_proj,gate[:,1])
        x = bezier_out + kernel_out
        x = torch.clamp(x, 0.0, 1.0)

        """
        x = self.denoise(x,feat_proj)
        bezier_out  = self.bezier(x,feat_proj,gate[:,1])
        defog_out   = self.defog(x,feat_proj,gate[:,2])
        sharpen_out = self.sharpening(x,feat_proj,gate[:,3])
        x = identity_out + bezier_out + defog_out + sharpen_out

        x = torch.clamp(x, 0.0, 1.0)
        """
        
        return x,gate


# class ProgressiveGatedDIP(torch.nn.Module):
#     def __init__(
#             self,
#             kernel_size=7,
#             num_of_gates=2,
#             encoder_output_dim=256,
#             use_residual=True,  # 新增：是否使用残差
#             residual_scale=0.3,  # 新增：残差缩放因子
#             **kwargs
#         ):
#         super().__init__()
        
#         self.use_residual = use_residual
#         self.residual_scale = residual_scale
        
#         # Gating Module
#         self.gate_module = nn.Sequential(
#             nn.Conv2d(encoder_output_dim, encoder_output_dim, 1, 1, 0),
#             nn.SELU(),
#             nn.Conv2d(encoder_output_dim, num_of_gates, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()  # 改为Sigmoid，确保范围在[0,1]
#         )
        
#         # Residual Scaling Module (新增)
#         self.residual_scale_module = nn.Sequential(
#             nn.Conv2d(encoder_output_dim, encoder_output_dim // 4, 1),
#             nn.SELU(),
#             nn.Conv2d(encoder_output_dim // 4, num_of_gates, 1),
#             nn.Sigmoid()
#         )
        
#         # Bezier 模块保持不变
#         self.bezier_module = nn.Sequential(
#             nn.Conv2d(encoder_output_dim, encoder_output_dim, 1, 1, 0),
#             nn.SELU(),
#             nn.Conv2d(encoder_output_dim, 12, kernel_size=1, stride=1, padding=0),
#             nn.Tanh()
#         )
        
#         L = 8
#         self.register_buffer("q_buf", torch.linspace(0, 1, L+1).view(1, 1, L+1))
#         self.register_buffer("B1_buf", 3*(1-self.q_buf)**2 * self.q_buf)
#         self.register_buffer("B2_buf", 3*(1-self.q_buf) * self.q_buf**2)
#         self.register_buffer("B3_buf", self.q_buf**3)
        
#         # Kernel 模块
#         self.kernel_size = kernel_size
#         self.kernel_module = nn.Sequential(
#             nn.Conv2d(encoder_output_dim, encoder_output_dim, 1, 1, 0),
#             nn.SELU(),
#             nn.Conv2d(encoder_output_dim, 2*3*self.kernel_size**2, kernel_size=1, stride=1, padding=0),
#         )

#     def progressive_bezier(self, x: torch.Tensor, latent_out: torch.Tensor, 
#                           bezier_gate: torch.Tensor, base_x: torch.Tensor = None) -> torch.Tensor:
#         """渐进式Bezier曲线增强"""
#         B, C, H, W = x.shape
        
#         # 1. 计算Bezier变换
#         out_curve = self._compute_bezier_curve(x, latent_out)
        
#         # 2. 渐进式增强策略
#         if self.use_residual and base_x is not None:
#             # 策略A：基于原始图像的残差增强
#             residual = out_curve - base_x  # 相对于原始图像的变换
#             weighted_residual = self.residual_scale * residual
#             out = x + bezier_gate.view(B, 1, 1, 1) * weighted_residual
#         else:
#             # 策略B：自适应增强
#             # 根据当前图像质量决定增强强度
#             brightness = x.mean(dim=1, keepdim=True)
#             overexposed_mask = (brightness > 0.8).float()
            
#             # 对过曝区域降低增强强度
#             adaptive_gate = bezier_gate.view(B, 1, 1, 1) * (1 - 0.5 * overexposed_mask)
#             out = adaptive_gate * out_curve + (1 - adaptive_gate) * x
        
#         return out.clamp(0.0, 1.0)

#     def progressive_kernel(self, x: torch.Tensor, latent_out: torch.Tensor, 
#                           kernel_gate: torch.Tensor, base_x: torch.Tensor = None) -> torch.Tensor:
#         """渐进式卷积增强"""
#         B, C, H, W = x.shape
        
#         # 获取卷积核参数
#         params = self.kernel_module(latent_out).squeeze()
#         params = params.view(B, 2, C, self.kernel_size, self.kernel_size)
        
#         K1 = params[:, 0]  # [B, 3, k, k]
#         K2 = params[:, 1]
#         K1 = K1 / (K1.abs().sum(dim=[2,3], keepdim=True) + 1e-6)
#         K2 = K2 / (K2.abs().sum(dim=[2,3], keepdim=True) + 1e-6)
        
#         padding = self.kernel_size // 2
        
#         # 并行卷积
#         x_flat = x.view(1, B*C, H, W)
#         K1_flat = K1.reshape(B*C, 1, self.kernel_size, self.kernel_size)
#         K2_flat = K2.reshape(B*C, 1, self.kernel_size, self.kernel_size)
        
#         conv1_flat = F.conv2d(x_flat, K1_flat, padding=padding, groups=B*C)
#         conv2_flat = F.conv2d(x_flat, K2_flat, padding=padding, groups=B*C)
        
#         conv1 = conv1_flat.view(B, C, H, W)
#         conv2 = conv2_flat.view(B, C, H, W)
        
#         # 渐进式增强
#         if self.use_residual:
#             # 残差增强：只增强细节，不改变整体结构
#             detail = conv2 - x  # 提取高频细节
#             output = x + kernel_gate.view(B, 1, 1, 1) * detail
#         else:
#             # 原始增强
#             output = x * conv1 + conv2
        
#         return output.clamp(0.0, 1.0)

#     def forward(self, x, feat_proj, base_x=None):
#         """
#         Args:
#             x: 当前增强后的图像
#             feat_proj: 特征
#             base_x: 原始输入图像（用于残差学习）
#         """
#         B, C, H, W = x.shape
        
#         # 获取门控
#         gate = self.gate_module(feat_proj).squeeze()
        
#         # 如果需要，获取残差缩放系数
#         if self.use_residual:
#             residual_scales = self.residual_scale_module(feat_proj).squeeze()
#             gate = gate * (0.1 + 0.9 * residual_scales)  # 动态调整门控范围
        
#         # 渐进式增强
#         bezier_out = self.progressive_bezier(x, feat_proj, gate[:, 0], base_x)
#         kernel_out = self.progressive_kernel(x, feat_proj, gate[:, 1], base_x)
        
#         # 自适应融合
#         brightness = x.mean(dim=1, keepdim=True)
#         dark_mask = (brightness < 0.3).float()
#         bright_mask = (brightness > 0.7).float()
        
#         # 对暗区域增加Bezier增强，对亮区域减少
#         bezier_weight = (0.5 + 0.3 * dark_mask - 0.3 * bright_mask)
#         kernel_weight = 1.0 - bezier_weight
        
#         x = bezier_weight * bezier_out + kernel_weight * kernel_out
#         x = torch.clamp(x, 0.0, 1.0)
        
#         return x, gate