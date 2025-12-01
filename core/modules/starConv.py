#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     starConv.py
@Time     :     2025/11/28 17:28:53
@Author   :     Louis Swift
@Desc     :     
        论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Rewrite_the_Stars_CVPR_2024_paper.pdf
        论文题目：Rewrite the Stars（CVPR 2024）
'''

import torch
import torch.nn as nn
from timm.models.layers import DropPath

__all__ = [ 
    'StarConv'
]

class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """构造一个包含卷积与批归一化的基本层。"""
        super().__init__()
        # 若未指定填充，自动设置为卷积核尺寸的一半；若k为列表，则对每个元素进行相同计算。
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        # 初始化一个不使用偏置的二维卷积层
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        # 初始化批归一化层以稳定训练过程
        self.bn = nn.BatchNorm2d(c2)
        # 采用SiLU激活函数（注意：当前前向传播版本未使用激活）
        self.act = nn.SiLU()
    def forward(self, x):
        """对输入张量执行卷积和批归一化操作。"""
        return self.bn(self.conv(x))  # 仅执行卷积和BN，不包括激活。   

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        """
        Star_Block模块通过深度卷积与1x1卷积扩展通道，
        结合ReLU6非线性激活和残差连接，来实现特征的重分配。
        """
        super().__init__()
        # 采用7x7深度卷积，其中groups设为dim实现逐通道卷积
        self.dwconv = ConvBN(dim, dim, 7, g=dim)
        # 两个1x1卷积分别用于将特征扩展至mlp_ratio倍的通道数
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        # 通过1x1卷积整合扩展后的特征
        self.g = ConvBN(mlp_ratio * dim, dim, 1)
        # 使用另一个7x7深度卷积实现进一步的特征重分配
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        # ReLU6激活函数限制输出范围
        self.act = nn.ReLU6()
        # 引入DropPath机制来随机丢弃部分路径以缓解过拟合；当drop_path=0时，该层为恒等映射
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """前向传播：提取特征后进行非线性变换，最后通过残差连接融合输入。"""
        input = x  # 保存原始输入用于残差连接
        x = self.dwconv(x)  # 通过7x7深度卷积提取局部特征
        x1, x2 = self.f1(x), self.f2(x)  # 通过两个1x1卷积扩展特征通道
        x = self.act(x1) * x2  # 将激活后的第一分支与第二分支逐元素相乘   
        x = self.g(x)  # 使用1x1卷积融合特征
        x = self.dwconv2(x)  # 再次应用7x7深度卷积细化特征
        x = input + self.drop_path(x)  # 添加残差连接并应用DropPath机制
        return x


class StarConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.star_block = Star_Block(out_channel,drop_path=0.1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.star_block(x)
        return x


if __name__ == '__main__':
    block = Star_Block(dim=32)  # 构造Star_Block模块，特征维度为32
    input = torch.rand(1, 32, 64, 64)  # 生成随机输入数据，尺寸为1x32x64x64
    output = block(input)  # 前向传播得到输出
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)  # 输出形状应与输入一致   
