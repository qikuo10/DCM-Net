#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序图卷积网络模块 (Temporal GCN) - 适配768通道版本
用于建模Instrument-Verb-Target-Triplet的三元组关系
"""

import torch
from torch import nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    图卷积层
    实现: X' = σ(A @ X @ W)
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, adj):
        """
        前向传播
        
        参数:
            x: (B, N, D) 节点特征矩阵
            adj: (N, N) 邻接矩阵
        
        返回:
            x: (B, N, D') 更新后的节点特征
        """
        B, N, D = x.shape
        
        # 图卷积: X' = A @ X
        x = torch.matmul(adj.unsqueeze(0), x)  # (B, N, D)
        
        # 线性变换 + BN + ReLU
        x = x.reshape(B * N, D)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.reshape(B, N, -1)
        
        return x


class GCNFusionModule(nn.Module):
    """
    GCN融合模块：在FPN融合过程中使用GCN优化融合
    
    简化设计（768通道版本）：
    - 12个语义节点：[I1, V1, T1, Tri1] + [I2, V2, T2, Tri2] + [I_fused, V_fused, T_fused, Tri_fused]
    - 节点维度保持768维，不降维
    - 通过图卷积学习并优化融合过程
    - 输出4个分类头 + Triplet特征传递给下一层
    """
    
    def __init__(self, in_channels=768, d_node=768,
                 num_gcn_layers=2, num_i=6, num_v=10, num_t=15, num_triplet=100):
        super(GCNFusionModule, self).__init__()
        
        self.in_channels = in_channels
        self.d_node = d_node
        self.num_gcn_layers = num_gcn_layers
        
        # 节点特征提取器 - 保持768维
        self.node_extractors = nn.ModuleDict({
            'instrument': nn.Sequential(
                nn.Conv1d(in_channels, d_node, 1),
                nn.BatchNorm1d(d_node),
                nn.ReLU()
            ),
            'verb': nn.Sequential(
                nn.Conv1d(in_channels, d_node, 1),
                nn.BatchNorm1d(d_node),
                nn.ReLU()
            ),
            'target': nn.Sequential(
                nn.Conv1d(in_channels, d_node, 1),
                nn.BatchNorm1d(d_node),
                nn.ReLU()
            ),
            'triplet': nn.Sequential(
                nn.Conv1d(in_channels, d_node, 1),
                nn.BatchNorm1d(d_node),
                nn.ReLU()
            ),
        })
        
        # 图卷积层 - 12个语义节点（不使用时序节点）
        num_nodes = 12
        self.adj_matrix = nn.Parameter(
            self._init_adjacency_matrix(num_nodes),
            requires_grad=True
        )
        
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(d_node, d_node) 
            for _ in range(num_gcn_layers)
        ])
        
        # 分类头 - 从768维到类别数（只对融合节点输出）
        self.classifier_i = nn.Sequential(
            nn.Conv1d(d_node, d_node // 2, 1),
            nn.BatchNorm1d(d_node // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_node // 2, num_i, 1)
        )
        
        self.classifier_v = nn.Sequential(
            nn.Conv1d(d_node, d_node // 2, 1),
            nn.BatchNorm1d(d_node // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_node // 2, num_v, 1)
        )
        
        self.classifier_t = nn.Sequential(
            nn.Conv1d(d_node, d_node // 2, 1),
            nn.BatchNorm1d(d_node // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_node // 2, num_t, 1)
        )
        
        self.classifier_triplet = nn.Sequential(
            nn.Conv1d(d_node, d_node // 2, 1),
            nn.BatchNorm1d(d_node // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_node // 2, num_triplet, 1)
        )
        
        # Triplet特征投影 - 保持768维
        self.triplet_proj = nn.Sequential(
            nn.Conv1d(d_node, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
    
    def _init_adjacency_matrix(self, num_nodes):
        """
        初始化邻接矩阵
        12个语义节点：[I1,V1,T1,Tri1, I2,V2,T2,Tri2, I_fused,V_fused,T_fused,Tri_fused]
        """
        adj = torch.zeros(num_nodes, num_nodes)
        
        # 每组内部全连接（3组，每组4个节点）
        for group_start in [0, 4, 8]:
            adj[group_start:group_start+4, group_start:group_start+4] = 1.0
        
        # 相同类型节点之间连接（I1-I2-I_fused, V1-V2-V_fused等）
        for offset in range(4):
            adj[offset, 4+offset] = 0.8
            adj[offset, 8+offset] = 0.8
            adj[4+offset, offset] = 0.8
            adj[4+offset, 8+offset] = 0.8
            adj[8+offset, offset] = 0.8
            adj[8+offset, 4+offset] = 0.8
        
        # 自连接
        adj = adj + torch.eye(num_nodes)
        
        # 度归一化
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        degree_matrix = torch.diag(degree_inv_sqrt)
        adj = degree_matrix @ adj @ degree_matrix
        
        return adj
    
    def forward(self, f_low, f_high):
        """
        前向传播（改进版：保留时序信息）
        
        输入:
            f_low: (B, 768, L) 低层特征（如f1）
            f_high: (B, 768, L) 高层特征（如f2，已经upsample到与f_low相同尺寸）
        
        返回:
            out_i, out_v, out_t, out_triplet: 融合节点的分类预测
            triplet_feat: (B, 768, L) Triplet节点特征，传递给下一层
        """
        B, C, L = f_low.shape
        
        # 简单融合（FPN原始方式）
        f_fused = f_low + f_high
        
        # 提取3组节点特征（f_low, f_high, f_fused）
        # 第1组：f_low的节点 (B, 768, L)
        feat_i1 = self.node_extractors['instrument'](f_low)
        feat_v1 = self.node_extractors['verb'](f_low)
        feat_t1 = self.node_extractors['target'](f_low)
        feat_tri1 = self.node_extractors['triplet'](f_low)
        
        # 第2组：f_high的节点
        feat_i2 = self.node_extractors['instrument'](f_high)
        feat_v2 = self.node_extractors['verb'](f_high)
        feat_t2 = self.node_extractors['target'](f_high)
        feat_tri2 = self.node_extractors['triplet'](f_high)
        
        # 第3组：f_fused的节点
        feat_i_fused = self.node_extractors['instrument'](f_fused)
        feat_v_fused = self.node_extractors['verb'](f_fused)
        feat_t_fused = self.node_extractors['target'](f_fused)
        feat_tri_fused = self.node_extractors['triplet'](f_fused)
        
        # 将所有节点特征堆叠 (B, 12, 768, L)
        all_node_feats = torch.stack([
            feat_i1, feat_v1, feat_t1, feat_tri1,
            feat_i2, feat_v2, feat_t2, feat_tri2,
            feat_i_fused, feat_v_fused, feat_t_fused, feat_tri_fused,
        ], dim=1)
        
        # 转换为 (B, L, 12, 768) 以便对每个时间步应用GCN
        all_node_feats = all_node_feats.permute(0, 3, 1, 2)  # (B, L, 12, 768)
        
        # 对每个时间步应用图卷积
        B, L, N, D = all_node_feats.shape
        all_node_feats = all_node_feats.reshape(B * L, N, D)  # (B*L, 12, 768)
        
        for gcn_layer in self.gcn_layers:
            all_node_feats = gcn_layer(all_node_feats, self.adj_matrix)
        
        all_node_feats = all_node_feats.reshape(B, L, N, D)  # (B, L, 12, 768)
        
        # 转换回 (B, 12, 768, L)
        all_node_feats = all_node_feats.permute(0, 2, 3, 1)  # (B, 12, 768, L)
        
        # 提取更新后的融合节点特征（索引8-11）
        updated_i_fused = all_node_feats[:, 8, :, :]  # (B, 768, L)
        updated_v_fused = all_node_feats[:, 9, :, :]
        updated_t_fused = all_node_feats[:, 10, :, :]
        updated_tri_fused = all_node_feats[:, 11, :, :]
        
        # 残差连接
        updated_i_fused = updated_i_fused + feat_i_fused
        updated_v_fused = updated_v_fused + feat_v_fused
        updated_t_fused = updated_t_fused + feat_t_fused
        updated_tri_fused = updated_tri_fused + feat_tri_fused
        
        # 通过分类头得到预测
        out_i = self.classifier_i(updated_i_fused)
        out_v = self.classifier_v(updated_v_fused)
        out_t = self.classifier_t(updated_t_fused)
        out_triplet = self.classifier_triplet(updated_tri_fused)
        
        # Triplet特征投影回768维，传递给下一层
        triplet_feat = self.triplet_proj(updated_tri_fused)
        
        return out_i, out_v, out_t, out_triplet, triplet_feat


if __name__ == "__main__":
    # 测试代码
    print("Testing GCNFusionModule (768 channels, 768 node dims)...")
    
    # 创建模块
    gcn_module = GCNFusionModule(
        in_channels=768,
        d_node=768,
        num_gcn_layers=2,
        num_i=6,
        num_v=10,
        num_t=15,
        num_triplet=100
    ).cuda()
    
    # 测试输入
    B, C, L = 2, 768, 1000
    f_low = torch.randn(B, C, L).cuda()
    f_high = torch.randn(B, C, L).cuda()
    
    print(f"\nInput f_low: {f_low.shape}")
    print(f"Input f_high: {f_high.shape}")
    
    out_i, out_v, out_t, out_triplet, triplet_feat = gcn_module(f_low, f_high)
    
    print(f"Output I: {out_i.shape}")
    print(f"Output V: {out_v.shape}")
    print(f"Output T: {out_t.shape}")
    print(f"Output Triplet: {out_triplet.shape}")
    print(f"Triplet Feature: {triplet_feat.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in gcn_module.parameters())
    trainable_params = sum(p.numel() for p in gcn_module.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    print("\nTest passed!")
