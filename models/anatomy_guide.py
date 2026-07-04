# 文件位置：models/anatomy_guide.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class AnatomyPriorGuideGeneration(nn.Module):
    def __init__(self, k_channels=18, guide_map_path=None):
        """
        Stage 1: 可学习的解剖先验生成模块
        :param k_channels: 解剖向导图的通道数 (Braak 核心脑区数，本例为 18)
        :param guide_map_path: 你用 Julich 图谱生成的 npy 张量路径
        """
        super().__init__()
        if guide_map_path is None:
            guide_map_path = "data/anatomy_prior/braak_guide_map.npy"
        
        # ==========================================
        # 1. 载入静态的医学真理 (Static Prior)
        # ==========================================
        # 形状为 (18, D, H, W)
        raw_atlas = np.load(guide_map_path) 
        raw_atlas_tensor = torch.from_numpy(raw_atlas).float()
        
        # 增加一个 Batch 维度，变成 (1, 18, D, H, W)
        # register_buffer 的作用是把它存入 GPU 内存，但不会被优化器更新 (不需要算梯度)
        self.register_buffer('atlas_prior', raw_atlas_tensor.unsqueeze(0)) 
        
        # ==========================================
        # 2. 定义动态的通道加权 (Dynamic Weights)
        # ==========================================
        # 初始化为 0，经过 Sigmoid 后权重都是 0.5，保证网络在冷启动时的公平性
        # nn.Parameter 告诉 PyTorch 这是一个需要反向传播更新的参数
        self.channel_weights = nn.Parameter(torch.zeros(k_channels))
        
    # 优化后的 Stage 1 前向传播
    def forward(self, mri_image):
        """
        :param mri_image: 当前输入的 MRI 批次 (B, 1, D, H, W)
        :return: adaptive_guide_batched: 适配Batch的动态解剖向导图 (B, k_channels, D, H, W)
        """
        # 网络自己去读取当前的 Batch Size 是多少
        B = mri_image.shape[0] 
        # 获取当前MRI的空间维度 (D, H, W)
        mri_spatial_size = mri_image.shape[2:]

        # ==========================================
        # 核心修改：将静态图谱插值到MRI的空间维度
        # ==========================================
        atlas_prior_resampled = F.interpolate(
            self.atlas_prior,                # 原始图谱 (1, 18, D_atlas, H_atlas, W_atlas)
            size=mri_spatial_size,           # 匹配MRI的空间尺寸
            mode='trilinear',                # 和其他模块保持一致的3D插值方式
            align_corners=False              # 避免空间对齐误差
        )
        
        # 计算带权重的向导图（使用插值后的图谱）
        w_k = torch.sigmoid(self.channel_weights).view(1, -1, 1, 1, 1)
        adaptive_guide = w_k * atlas_prior_resampled
    
        # 调试输出
        # print(f"图谱值范围: {atlas_prior_resampled.min().item():.3f}-{atlas_prior_resampled.max().item():.3f}")
        # print(f"权重值范围: {w_k.min().item():.3f}-{w_k.max().item():.3f}")
        # print(f"输出值范围: {adaptive_guide.min().item():.3f}-{adaptive_guide.max().item():.3f}")
    
        # 适配当前Batch大小
        adaptive_guide_batched = adaptive_guide.expand(B, -1, -1, -1, -1)
    
        return adaptive_guide_batched
        # 校验空间维度匹配 (D, H, W)
        # if self.atlas_prior.shape[2:] != mri_image.shape[2:]:
        #     raise ValueError(
        #         f"解剖图谱空间维度 {self.atlas_prior.shape[2:]} 与MRI {mri_image.shape[2:]} 不匹配！"
        # )
        
        # # 计算带权重的向导图
        # w_k = torch.sigmoid(self.channel_weights).view(1, -1, 1, 1, 1)
        # adaptive_guide = w_k * self.atlas_prior
        
        # # 完美自适应当前 Batch 大小
        # adaptive_guide_batched = adaptive_guide.expand(B, -1, -1, -1, -1)
        
        # return adaptive_guide_batched