#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     defog.py
@Time     :     2025/12/08 15:16:49
@Author   :     Louis Swift
@Desc     :     
'''


import torch 
from torch import nn
from torch import Tensor 
import torch.nn.functional as F 
from kornia.filters import guided_blur

__all__ = [
   'Defog'
]

class Defog(nn.Module):
    def __init__(
            self,
            num_of_params: int,
            use_guided_filter: bool,
            value_range: list,
            encoder_output_dim: int,
            **kwargs
        ):
        super().__init__()

        self.use_guided_filter = use_guided_filter

        self.params_predictor = nn.Sequential(
            nn.Conv2d(encoder_output_dim,encoder_output_dim//2,3,2,1),
            nn.LeakyReLU(0.02,True),
            nn.Conv2d(encoder_output_dim//2,num_of_params,3,2,1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_of_params,num_of_params,1,1,0),
            nn.Tanh() # [-1,1]
        )
        self.value_range = value_range # [0.1,1]

    def tanh_range(self,params):
        return params * (self.value_range[1] - self.value_range[0]) / 2 + (self.value_range[1] + self.value_range[0]) / 2
    
    def darkChannel_batch(self,batch_im, r = 3):

      if r == 0 : # 无窗口
        return torch.min(batch_im,dim=1)[0]

      # 有窗口
      # 求每个像素点在所有通道上的最小值，以找到暗通道
      DarkChann = torch.min(batch_im, dim=1, keepdim=True)[0]

      # 使用最大池化的负数来模拟腐蚀操作，以实现窗口中的最小值
      # 先对输入数据取负，然后应用最大池化操作，最后再对结果取负
      DarkChann = -F.max_pool2d(-DarkChann, kernel_size=2 * r + 1, stride=1, padding=r, return_indices=False)

      return DarkChann
    
    def estimateA_batch(self,batch_im,batch_DarkChann):
      n,c,h,w = batch_im.shape
      length  = h * w 
      num     = max (int(length * 0.001),1)
      batch_DarkChannVec = batch_DarkChann.reshape(n,-1)

      batch_imageVec     = batch_im.permute(0,2,3,1).reshape(n,length,c) # 重新排列维度并reshape

      indices = batch_DarkChannVec.argsort(dim=1,descending = True)      # 降序
      indices = indices[:,:num]

      batch_sumA = torch.zeros(n,1,c,device=batch_im.device)

      for i in range(n):
          for ind in range(num):
              batch_sumA[i] += batch_imageVec[i,indices[i,ind]]

      batch_A = batch_sumA / num

      return batch_A.squeeze(1)  # (n,c)
    
    def estimateT_batch(self,batch_im,batch_A,param):
      n,c,h,w        = batch_im.shape
      batch_A_expand = batch_A.view(n,c,1,1).expand_as(batch_im)    
      batch_T_tmp    = self.darkChannel_batch( batch_im / batch_A_expand)  # batch_IcA

      batch_T        = 1 - param * batch_T_tmp

      return batch_T.repeat(1, 3, 1, 1)       # size - [batch,channel, height,weight]
    
    def forward(self, x: Tensor,latent_out: Tensor) -> Tensor:

        params = self.params_predictor(latent_out) # [B, 1,1,1], in [-1,1]
        params = self.tanh_range(params) 

        batch_DarkChann = self.darkChannel_batch(x,r=0)
        batch_A         = self.estimateA_batch(x,batch_DarkChann)
        batch_T         = self.estimateT_batch(x,batch_A,params)

        # 导向滤波
        if self.use_guided_filter:
            batch_im_gray   = torch.min(x,1,keepdim=True)[0]
            batch_T     = guided_blur(batch_im_gray,batch_T,kernel_size=5,eps=1e-4)
        
        batch_T_clamped = torch.clamp(batch_T, min=0.01) 

        out = (x - batch_A.unsqueeze(2).unsqueeze(3)) / batch_T_clamped + batch_A.unsqueeze(2).unsqueeze(3)

        out = (out - out.min()) / (out.max() - out.min())
        
        return out 