#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     afpn.py
@Time     :     2025/11/28 17:51:26
@Author   :     Louis Swift
@Desc     :     
            参考论文：https://arxiv.org/pdf/2306.15988
            标题：Asymptotic Feature Pyramid Network for Labeling Pixels and Regions（2024-1区）
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import BasicConv

__all__ = [
    'AFPN'
]



class BasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        super().__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv(in_ch, out_ch, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        )
    def forward(self, x):
        return self.net(x)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.net = BasicConv(in_ch, out_ch, scale_factor, stride=scale_factor, pad=0)
    def forward(self, x):
        return self.net(x)


# ------------------ Dynamic ASFF ------------------
class DynamicASFF(nn.Module):
    def __init__(self, inter_dim, num_levels):
        super().__init__()
        compress_c = max(8, inter_dim // 16)
        self.weight_convs = nn.ModuleList([
            BasicConv(inter_dim, compress_c, 1) for _ in range(num_levels)
        ])
        self.fuse_conv = nn.Conv2d(compress_c * num_levels, num_levels, 1)
        self.post = BasicConv(inter_dim, inter_dim, 3)

    def forward(self, inputs):
        # inputs: List[Tensor], each [N, C, H_i, W_i]
        weights = torch.cat([conv(x) for conv, x in zip(self.weight_convs, inputs)], dim=1)
        weights = F.softmax(self.fuse_conv(weights), dim=1)  # [N, L, H, W]
        fused = sum(inputs[i] * weights[:, i:i+1] for i in range(len(inputs)))
        return self.post(fused)


# ------------------ BlockBody (通用版) ------------------
class BlockBody(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.num = len(channels)
        assert self.num >= 2, "At least 2 input scales required"

        # Pre-conv to unify channel
        self.pre_convs = nn.ModuleList([
            BasicConv(c, c, 1) for c in channels
        ])

        # Build upsample/downsample ops per pair (i, j)
        self.upsample_ops = nn.ModuleList()
        self.downsample_ops = nn.ModuleList()
        for i in range(self.num):
            up_row = nn.ModuleList()
            down_row = nn.ModuleList()
            for j in range(self.num):
                if i == j:
                    up_row.append(None)
                    down_row.append(None)
                elif i < j:
                    scale = 2 ** (j - i)
                    up_row.append(Upsample(channels[j], channels[i], scale))
                    down_row.append(None)
                else:
                    scale = 2 ** (i - j)
                    up_row.append(None)
                    down_row.append(Downsample(channels[j], channels[i], scale))
            self.upsample_ops.append(up_row)
            self.downsample_ops.append(down_row)

        # Stages: stage k fuses first (k+2) levels? No — stage s fuses (s+2) levels
        # We'll do: stage 0 → fuse 2 levels, stage 1 → fuse 3, ..., last → fuse all
        self.stage_res_blocks = nn.ModuleList()
        self.stage_asffs = nn.ModuleList()

        for k in range(2, self.num + 1):  # k = number of levels to fuse
            res_blocks = nn.ModuleList([
                nn.Sequential(*[BasicBlock(channels[i], channels[i]) for _ in range(2)])
                for i in range(k)
            ])
            asffs = nn.ModuleList([
                DynamicASFF(channels[i], k) for i in range(k)
            ])
            self.stage_res_blocks.append(res_blocks)
            self.stage_asffs.append(asffs)

    def forward(self, x):
        feats = [conv(f) for conv, f in zip(self.pre_convs, x)]  # [C1, C2, ..., Cn]

        # Progressive fusion
        for stage_idx, k in enumerate(range(2, self.num + 1)):
            # Align features for first k scales
            aligned_list_per_level = []
            for i in range(k):
                inputs_for_level_i = []
                for j in range(k):
                    if i == j:
                        inputs_for_level_i.append(feats[j])
                    elif i < j:
                        inputs_for_level_i.append(self.upsample_ops[i][j](feats[j]))
                    else:
                        inputs_for_level_i.append(self.downsample_ops[i][j](feats[j]))
                aligned_list_per_level.append(inputs_for_level_i)

            # Fuse via ASFF (pass LIST, not *args)
            new_feats = [
                self.stage_asffs[stage_idx][i](aligned_list_per_level[i])
                for i in range(k)
            ]

            # Refine
            refined = [
                self.stage_res_blocks[stage_idx][i](new_feats[i])
                for i in range(k)
            ]

            # Update first k features
            feats[:k] = refined

        return feats


# ------------------ AFPN Head ------------------
class AFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        reduced = [c // 8 for c in in_channels]
        self.compress = nn.ModuleList([
            BasicConv(c_in, c_red, 1) for c_in, c_red in zip(in_channels, reduced)
        ]) # compress to 1/8 channels
        self.body = BlockBody(reduced)
        self.project = nn.ModuleList([
            BasicConv(c_red, out_channels, 1) for c_red in reduced
        ])
        self.extra_pool = nn.MaxPool2d(kernel_size=1, stride=2)

    def forward(self, x):
        compressed = [conv(f) for conv, f in zip(self.compress, x)]
        fused = self.body(compressed)
        outputs = [proj(f) for proj, f in zip(self.project, fused)]
        p6 = self.extra_pool(outputs[-1])
        return (*outputs, p6)


# ------------------ Test ------------------
if __name__ == "__main__":
    print("Test with 3 scales:")
    model = AFPN([256, 512, 1024], out_channels=256)
    inputs = (
        torch.randn(1, 256, 80, 80),
        torch.randn(1, 512, 40, 40),
        torch.randn(1, 1024, 20, 20),
    )
    outs = model(inputs)
    for i, o in enumerate(outs):
        print(f"P{i+2}: {o.shape}")

    print("\n Test with 4 scales:")
    model4 = AFPN([64, 128, 256, 512])
    inputs4 = (
        torch.randn(1, 64, 160, 160),
        torch.randn(1, 128, 80, 80),
        torch.randn(1, 256, 40, 40),
        torch.randn(1, 512, 20, 20),
    )
    outs4 = model4(inputs4)
    for i, o in enumerate(outs4):
        print(f"P{i+2}: {o.shape}")