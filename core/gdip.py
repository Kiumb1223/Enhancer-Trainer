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
import torchvision 
import torch.nn.functional as F 

__all__ = [
    'GatedDIP'
]

class GatedDIP(torch.nn.Module):
    def __init__(
            self,
            num_of_gates = 3, # the number of filters
            kernel_size: int = 9 ,
            encoder_output_dim = 256,
            **kwargs
        ):
        super().__init__()

        # Gating Module
        self.gate_module = nn.Sequential(
                nn.Linear(encoder_output_dim,num_of_gates,bias=True),
                nn.Softmax(dim=-1)
            )

        # Filter Module
        self.bezier_module =nn.Sequential(
                nn.Linear(encoder_output_dim,12,bias=True)
            )

        self.kernel_size = kernel_size
        self.kernel_module = nn.Sequential(
                nn.Linear(encoder_output_dim,6*self.kernel_size**2,bias=True)
            )

    def tanh01(self,x : torch.tensor):
        """Shifts tanh from the [-1, 1] range to the [0, 1 range] and returns it for the given input. 

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Constrained tanh
        """
        return torch.tanh(x)*0.5+0.5

    def tanh_range(self,x : torch.tensor,left : float,right : float):
        """Returns tanh constrained to a particular range

        Args:
            x (torch.tensor): Input tensor
            left (float): Left bound 
            right (float): Right bound

        Returns:
            torch.tensor: Constrained tanh
        """
        return self.tanh01(x)*(right-left)+ left
    
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
    
    def bezier(self, x: torch.Tensor, latent_out: torch.Tensor, bezier_gate: torch.Tensor):
        B, C, H, W = x.shape
        assert C == 3, "Input must be 3-channel"
        device = x.device

        # Get parameters
        params = self.bezier_module(latent_out)  # [B, 12]
        params = torch.tanh(params)              # [-1, 1]
        params = params.view(B, 3, 4)            # [B, 3, 4]

        r1, theta1, r2, theta2 = params.unbind(dim=-1)  # each [B, 3]

        # Convert to control points in [0,1]
        ang1 = (theta1 + 1) * torch.pi / 4
        ang2 = (theta2 + 1) * torch.pi / 4

        p1_x = (r1 + 1) * 0.5 * torch.cos(ang1)
        p1_y = (r1 + 1) * 0.5 * torch.sin(ang1)
        p2_x = 1 - (r2 + 1) * 0.5 * torch.cos(ang2)
        p2_y = 1 - (r2 + 1) * 0.5 * torch.sin(ang2)

        # [B, 3, 4]: each channel has (p1x, p2x, p1y, p2y)
        control_points = torch.stack([p1_x, p2_x, p1_y, p2_y], dim=-1)  # [B, 3, 4]

        # Sample Bezier curve at 9 points (q=0 to 1)
        q = torch.linspace(0, 1, 9, device=device)  # [9]
        B1 = 3 * q * (1 - q)**2    # [9]
        B2 = 3 * q**2 * (1 - q)    # [9]
        B3 = q**3                  # [9]

        # Expand for broadcasting: [1, 1, 9]
        B1 = B1.view(1, 1, 9)
        B2 = B2.view(1, 1, 9)
        B3 = B3.view(1, 1, 9)

        # Compute Cx, Cy for each channel: [B, 3, 9]
        Cx = B1 * control_points[..., 0:1] + B2 * control_points[..., 1:2] + B3  # p1x, p2x
        Cy = B1 * control_points[..., 2:3] + B2 * control_points[..., 3:4] + B3  # p1y, p2y

        # Differences between consecutive points: [B, 3, 8]
        Dx = Cx[..., 1:] - Cx[..., :-1]
        Dy = Cy[..., 1:] - Cy[..., :-1]

        # Reshape for broadcasting with image
        Cx0 = Cx[..., :-1].view(B, 3, 8, 1, 1)   # [B, 3, 8, 1, 1]
        Dx  = Dx.view(B, 3, 8, 1, 1)
        Dy  = Dy.view(B, 3, 8, 1, 1)

        x_in = x.view(B, 3, 1, H, W)  # [B, 3, 1, H, W]

        # Clamp input offset within each segment
        tone_step = torch.clamp(x_in - Cx0, min=torch.zeros_like(Dx), max=Dx)

        # Compute slope and accumulate
        slope = Dy / (Dx + 1e-30)
        output = (tone_step * slope).sum(dim=2)  # [B, 3, H, W]

        # Apply gate
        output = output * bezier_gate.view(B, 1, 1, 1)

        # Optional: clamp final output to [0,1] if needed
        output = torch.clamp(output, 0.0, 1.0)

        return output
    
    def kernel(self, x, latent_out, kernel_gate):
        B, C, H, W = x.shape
        assert C == 3, "Only 3-channel supported"

        params = self.kernel_module(latent_out)  # [B, 2*3*k*k]
        params = params.view(B, 2, C, self.kernel_size, self.kernel_size)
        params = torch.tanh(params)

        K1 = params[:, 0]  # [B, 3, k, k]
        K2 = params[:, 1]  # [B, 3, k, k]

        x = torch.clamp(x, 0.0, 1.0)

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
    def forward(self,x,linear_proj):

        gate = self.tanh_range(self.gate_module(linear_proj),0.01,1.0)

        # identity_out = self.identity(x,gate[:,0])
        bezier_out = self.bezier(x,linear_proj,gate[:,0])
        kernel_out = self.kernel(x,linear_proj,gate[:,1])

        x = bezier_out + kernel_out
        x = torch.clamp(x,0.0,1.0)

        return x,gate
