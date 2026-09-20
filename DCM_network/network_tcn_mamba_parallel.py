#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCN-Mamba并行网络 + GCN融合模块
在每个TCN层旁边并行一个Mamba层，512通道分成256+256分别处理后concat
添加GCN融合模块用于FPN融合过程
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
import copy
import random
import math
import numpy as np

# 导入mamba-ssm库
try:
    from mamba_ssm import Mamba
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    _MAMBA_SSM_AVAILABLE = False
    print("Warning: mamba-ssm not available, please install: pip install mamba-ssm")

# 导入GCN融合模块
from temporal_gcn import GCNFusionModule


class VideoNas(nn.Module):
    def __init__(self, args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, num_i=6, num_v=10,
                 num_t=15):
        super(VideoNas, self).__init__()
        # 使用TCN-Mamba并行结构
        self.PG = ParallelTCNMamba(num_layers_PG, num_f_maps, dim, num_classes)

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_out_i = nn.Conv1d(num_f_maps, num_i, 1)
        self.conv_out_v = nn.Conv1d(num_f_maps, num_v, 1)
        self.conv_out_t = nn.Conv1d(num_f_maps, num_t, 1)
        self.args = args
        self.Rs = nn.ModuleList(
            [copy.deepcopy(RefinementParallel(args, num_layers_R, num_f_maps, num_classes, num_classes, self.conv_out)) 
             for s in range(num_R)])
        self.use_fpn = args.fpn
        self.use_output = args.output
        self.use_feature = args.feature
        self.use_trans = args.trans
        self.fuse_w = float(getattr(args, 'fuse_w', 0.0))
        self.fuse_src = int(getattr(args, 'fuse_src', 3))
        self.fuse_ref = int(getattr(args, 'fuse_ref', 0))
        
        if args.fpn:
            # 使用带GCN融合的FPN
            self.fpn = FPNWithGCN(num_f_maps, num_i=num_i, num_v=num_v, num_t=num_t, num_triplet=num_classes)

    def forward(self, x, ismask):
        out_list = []
        out_list_i = []
        out_list_v = []
        out_list_t = []
        f_list = []
        x = x.permute(0, 2, 1)
        
        if self.args.mask and ismask:
            num_patches = x.flatten().shape[0]
            num_mask = int(num_patches * 0.75)
            mask = torch.concat((torch.zeros(num_patches - num_mask), torch.ones(num_mask)))
            mask = mask[torch.randperm(mask.nelement())]
            mask = mask.view(x.shape)
            f, out1 = self.PG(x, mask)
        else:
            f, out1 = self.PG(x)

        f_list.append(f)
        if not self.use_fpn:
            out_list.append(out1)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)
            
        if self.use_fpn:
            # FPN + GCN融合，返回每个融合步骤的预测
            out_list, out_list_i, out_list_v, out_list_t = self.fpn(f_list)
            if self.fuse_w != 0.0:
                for _lst in (out_list, out_list_i, out_list_v, out_list_t):
                    _lst.append(_lst[self.fuse_src] + self.fuse_w * _lst[self.fuse_ref])
                
        return out_list, out_list_i, out_list_v, out_list_t, f_list, f_list


class FPNWithGCN(nn.Module):
    """
    FPN + GCN融合模块
    在FPN的每一步融合中使用GCN优化融合过程
    """
    def __init__(self, num_f_maps, num_i=6, num_v=10, num_t=15, num_triplet=100):
        super(FPNWithGCN, self).__init__()
        
        # FPN的lateral层
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        
        # 最后一个阶段（f4）的简单分类头
        self.conv_out_f4 = nn.Conv1d(num_f_maps, num_triplet, 1)
        self.conv_out_i_f4 = nn.Conv1d(num_f_maps, num_i, 1)
        self.conv_out_v_f4 = nn.Conv1d(num_f_maps, num_v, 1)
        self.conv_out_t_f4 = nn.Conv1d(num_f_maps, num_t, 1)
        
        # 3个GCN融合模块（f3+f4, f2+p3, f1+p2）
        self.gcn_fusion_modules = nn.ModuleList([
            GCNFusionModule(
                in_channels=num_f_maps,
                d_node=num_f_maps,  # 512 -> 512 (保持不降维)
                num_gcn_layers=2,
                num_i=num_i,
                num_v=num_v,
                num_t=num_t,
                num_triplet=num_triplet
            )
            for _ in range(3)
        ])

    def _upsample_add(self, x, y):
        _, _, W = y.size()
        return F.interpolate(x, size=W, mode='linear') + y

    def forward(self, f_list):
        out_list = []
        out_list_i = []
        out_list_v = []
        out_list_t = []
        
        c1, c2, c3, c4 = f_list[0], f_list[1], f_list[2], f_list[3]
        
        # 第0步：f4直接输出（简单分类头）
        out_list.append(self.conv_out_f4(c4))
        out_list_i.append(self.conv_out_i_f4(c4))
        out_list_v.append(self.conv_out_v_f4(c4))
        out_list_t.append(self.conv_out_t_f4(c4))
        
        # 第1步融合：f3 + f4 → p3
        c4_upsampled = self._upsample_add(c4, self.latlayer1(c3))
        out_i, out_v, out_t, out_triplet, triplet_feat_p3 = self.gcn_fusion_modules[0](c3, c4_upsampled)
        out_list.append(out_triplet)
        out_list_i.append(out_i)
        out_list_v.append(out_v)
        out_list_t.append(out_t)
        
        # 第2步融合：f2 + p3 → p2
        p3_upsampled = self._upsample_add(triplet_feat_p3, self.latlayer2(c2))
        out_i, out_v, out_t, out_triplet, triplet_feat_p2 = self.gcn_fusion_modules[1](c2, p3_upsampled)
        out_list.append(out_triplet)
        out_list_i.append(out_i)
        out_list_v.append(out_v)
        out_list_t.append(out_t)
        
        # 第3步融合：f1 + p2 → p1
        p2_upsampled = self._upsample_add(triplet_feat_p2, self.latlayer3(c1))
        out_i, out_v, out_t, out_triplet, _ = self.gcn_fusion_modules[2](c1, p2_upsampled)
        out_list.append(out_triplet)
        out_list_i.append(out_i)
        out_list_v.append(out_v)
        out_list_t.append(out_t)
        
        return out_list, out_list_i, out_list_v, out_list_t


class FPN(nn.Module):
    """原始FPN（不使用GCN）"""
    def __init__(self, num_f_maps):
        super(FPN, self).__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, W = y.size()
        return F.interpolate(x, size=W, mode='linear') + y

    def forward(self, out_list):
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1, p2, p3, p4]


class ParallelTCNMamba(nn.Module):
    """
    TCN-Mamba并行结构
    768通道分成384+384，分别输入TCN和Mamba，然后concat
    """
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(ParallelTCNMamba, self).__init__()
        # 使用传入的层数（支持消融实验）
        print(f"ParallelTCNMamba: {num_layers} layers, channels: {num_f_maps}")
        
        self.num_f_maps = num_f_maps
        self.half_channels = num_f_maps // 2  # 384 (when num_f_maps=768)
        
        # 输入投影：dim -> num_f_maps
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        
        # 并行层：TCN和Mamba各处理256通道
        self.parallel_layers = nn.ModuleList([
            ParallelTCNMambaLayer(2 ** i, self.half_channels, self.half_channels) 
            for i in range(num_layers)
        ])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.channel_dropout = nn.Dropout2d()
        self.num_classes = num_classes

    def forward(self, x, labels=None, mask=None, test=False):
        if mask is not None:
            x = x * mask

        x = x.unsqueeze(3)
        x = self.channel_dropout(x)
        x = x.squeeze(3)

        # 输入投影
        out = self.conv_1x1(x)  # (B, 768, L)
        
        # 通过并行层
        for layer in self.parallel_layers:
            out = layer(out)  # (B, 768, L)

        x = self.conv_out(out)  # (B, num_classes, L)

        return out, x


class RefinementParallel(nn.Module):
    """
    Refinement模块，使用TCN-Mamba并行结构
    """
    def __init__(self, args, num_layers, num_f_maps, dim, num_classes, conv_out):
        super(RefinementParallel, self).__init__()
        
        # 使用传入的层数（支持消融实验）
        print(f"RefinementParallel: {num_layers} layers, channels: {num_f_maps}")
        
        self.half_channels = num_f_maps // 2  # 384 (when num_f_maps=768)
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        
        # 并行层
        self.parallel_layers = nn.ModuleList([
            ParallelTCNMambaLayer(2 ** i, self.half_channels, self.half_channels) 
            for i in range(num_layers)
        ])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.max_pool_1x1 = nn.AvgPool1d(kernel_size=7, stride=3)
        self.use_output = args.output
        self.hier = args.hier

    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x
            
        for layer in self.parallel_layers:
            out = layer(out)
            
        if self.hier:
            f = self.max_pool_1x1(out)
        else:
            f = out
        out = self.conv_out(f)

        return f, out


class ParallelTCNMambaLayer(nn.Module):
    """
    单个TCN-Mamba并行层
    输入768通道 -> 分成384+384 -> TCN和Mamba并行处理 -> concat回768通道
    """
    def __init__(self, dilation, in_channels, out_channels):
        super(ParallelTCNMambaLayer, self).__init__()
        
        self.in_channels = in_channels * 2  # 768
        self.out_channels = out_channels * 2  # 768
        self.half_channels = in_channels  # 384
        
        # TCN分支 (处理384通道)
        self.tcn_branch = DilatedResidualLayer(dilation, in_channels, out_channels)
        
        # Mamba分支 (处理384通道)
        if _MAMBA_SSM_AVAILABLE:
            self.mamba_branch = MambaLayer(in_channels, out_channels)
        else:
            # 如果Mamba不可用，使用简单的卷积层代替
            print("Warning: Using Conv1d instead of Mamba")
            self.mamba_branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Dropout()
            )

    def forward(self, x):
        """
        x: (B, 768, L)
        """
        # 分割通道：前384给TCN，后384给Mamba
        x_tcn = x[:, :self.half_channels, :]  # (B, 384, L)
        x_mamba = x[:, self.half_channels:, :]  # (B, 384, L)
        
        # TCN分支处理
        out_tcn = self.tcn_branch(x_tcn)  # (B, 384, L)
        
        # Mamba分支处理
        out_mamba = self.mamba_branch(x_mamba)  # (B, 384, L)
        
        # Concat回768通道
        out = torch.cat([out_tcn, out_mamba], dim=1)  # (B, 768, L)
        
        return out


class DilatedResidualLayer(nn.Module):
    """
    标准的TCN残差层
    """
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class MambaLayer(nn.Module):
    """
    Mamba层，包含LayerNorm和残差连接
    """
    def __init__(self, d_model, out_channels, d_state=16, d_conv=4, expand=2):
        super(MambaLayer, self).__init__()
        
        self.d_model = d_model
        self.out_channels = out_channels
        
        # LayerNorm (需要在(B, L, D)格式上操作)
        self.norm = nn.LayerNorm(d_model)
        
        # Mamba核心
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 如果输入输出通道数不同，需要投影
        self.proj = None
        if d_model != out_channels:
            self.proj = nn.Linear(d_model, out_channels)

    def forward(self, x):
        """
        x: (B, C, L) - Conv1d格式
        """
        # 转换为(B, L, C)格式
        x = x.permute(0, 2, 1)  # (B, L, C)
        
        residual = x
        
        # LayerNorm
        x = self.norm(x)
        
        # Mamba
        x = self.mamba(x)
        
        # 残差连接
        x = residual + x
        
        # 投影（如果需要）
        if self.proj is not None:
            x = self.proj(x)
        
        # 转换回(B, C, L)格式
        x = x.permute(0, 2, 1)  # (B, C, L)
        
        return x
