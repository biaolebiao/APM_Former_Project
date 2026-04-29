# 文件位置：models/alignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomyGuidedAlignment(nn.Module):
    def __init__(self, mri_channels, guide_channels=18,displacement_scale=0.1):
        """
        Stage 2: 解剖引导的可变形特征对齐模块 (Anatomy-Guided Alignment)
        
        :param mri_channels: 输入的 MRI 特征图通道数 (例如从 Swin UNETR 提取出的通道数)
        :param guide_channels: 解剖向导图的通道数 (本例中 Braak 核心脑区 K=18)
        """
        super().__init__()
        self.guide_channels = guide_channels
        self.displacement_scale = displacement_scale
        
        # 拼接后的总通道数 (MRI特征 + 解剖先验)
        in_channels = mri_channels + guide_channels
        # 1. 共享特征提取主干
        self.shared_net = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. 偏移量预测头 (输出 3 个通道: Z, Y, X 的形变)
        self.offset_head = nn.Conv3d(in_channels // 4, 3, kernel_size=3, padding=1)
        
        # 3. 调制掩码预测头 (输出 1 个通道: 0~1的注意力权重)
        self.mask_head = nn.Conv3d(in_channels // 4, 1, kernel_size=3, padding=1)
        
        # 【极其关键的初始化神技】
        # 强制将偏移头的权重和偏置初始化为 0，确保初始形变为 0
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        
        # 掩码头的权重初始化为 0，偏置初始化为 0
        # 这样初始状态下 raw_mask 为 0，经过 sigmoid 后变为 0.5
        # 意味着初始状态下所有特征被一视同仁地保留 50%
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    def forward(self, mri_features, anatomy_guide_map):
        """
        前向传播
        :param mri_features:      Swin Transformer 提取的图像特征, 形状 (B, C, D', H', W')
        :param anatomy_guide_map: Stage 1 生成的动态解剖向导图,   形状 (B, 18, D, H, W)
        :return:                  被形变拉扯对齐后的特征,          形状 (B, C, D', H', W')
        """
        # 校验通道数
        if anatomy_guide_map.shape[1] != self.guide_channels:
            raise ValueError(f"解剖向导图通道数 {anatomy_guide_map.shape[1]} 与预期 {self.guide_channels} 不匹配！")
        # 校验空间维度（至少维度数一致，3D）
        if len(mri_features.shape) != 5 or len(anatomy_guide_map.shape) != 5:
            raise ValueError("输入必须是5维张量 (B, C, D, H, W)！")
        # 获取当前特征图的空间尺寸
        B, C, D_feat, H_feat, W_feat = mri_features.shape
        
        # ==========================================
        # 1. 空间分辨率对齐 (Downsample)
        # ==========================================
        # 解剖向导图通常是高分辨率的，而 MRI 特征经过网络池化变小了。
        # 将高分辨率的解剖向导图，精确缩小到与当前 MRI 特征图一样的分辨率。
        guide_down = F.interpolate(
            anatomy_guide_map, 
            size=(D_feat, H_feat, W_feat), 
            mode='trilinear', 
            align_corners=False
        )
        
        # ==========================================
        # 2. 知识与数据融合 (Concat Fusion)
        # ==========================================
        # 把医学真理(向导图)和当前病理特征(MRI特征)在通道维度叠在一起
        # 形状变为 (B, C+18, D', H', W')
        fused_features = torch.cat([mri_features, guide_down], dim=1)
        
        # 3. 提取共享特征并双头预测 (Offsets & Mask)
        # ==========================================
        shared_feat = self.shared_net(fused_features)
        
        # 预测形变场 (B, 3, D', H', W')
        raw_displacement = self.offset_head(shared_feat)
        displacement_field = torch.tanh(raw_displacement) * self.displacement_scale
        
        # 预测调制掩码 (B, 1, D', H', W')
        raw_mask = self.mask_head(shared_feat)
        # 使用 sigmoid 将掩码约束到 [0, 1] 之间
        modulation_mask = torch.sigmoid(raw_mask)
        
        # ==========================================
        # 4. 构建标准的三维坐标网格 (Standard Grid)
        # ==========================================
        # PyTorch 的 grid_sample 要求的空间坐标范围是 [-1, 1]
        # 我们需要生成一个规规矩矩的“金属网格”
        vectors = [
            torch.linspace(-1, 1, D_feat, device=mri_features.device),
            torch.linspace(-1, 1, H_feat, device=mri_features.device),
            torch.linspace(-1, 1, W_feat, device=mri_features.device)
        ]
        # meshgrid 生成 3D 坐标系
        grid_d, grid_h, grid_w = torch.meshgrid(vectors, indexing='ij')
        
        # 拼接网格并扩展到 Batch 大小
        # 输出通道为 3，分别代表在 D(Z轴), H(Y轴), W(X轴) 方向上的连续位移量（对应meshgrid顺序）
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0) # (1, D', H', W', 3)
        
        base_grid = base_grid.expand(B, -1, -1, -1, -1)                        # (B, D', H', W', 3)
        
        # ==========================================
        # 5. 生成变形网格 (Deformed Grid)
        # ==========================================
        # 将位移场的通道维移到最后: (B, 3, D', H', W') -> (B, D', H', W', 3)
        displacement_field = displacement_field.permute(0, 2, 3, 4, 1)
        print(f"raw_displacement 的形状: {raw_displacement.shape}")
        print(f"base_grid 的形状: {base_grid.shape}")
        print(f"displacement_field 的形状: {displacement_field.shape}")
        # 标准网格 + 网络预测的拉力位移 = 变形后的弹性网格
        deformed_grid = base_grid + displacement_field

        B, C, D_feat, H_feat, W_feat = mri_features.shape
        vectors = [
            torch.linspace(-1, 1, D_feat, device=mri_features.device),
            torch.linspace(-1, 1, H_feat, device=mri_features.device),
            torch.linspace(-1, 1, W_feat, device=mri_features.device)
        ]
        grid_d, grid_h, grid_w = torch.meshgrid(vectors, indexing='ij')
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        displacement_field = displacement_field.permute(0, 2, 3, 4, 1)
        deformed_grid = base_grid + displacement_field
        
        # ==========================================
        # 6. 医学解剖对齐重采样 (Deformable Resampling)
        # ==========================================
        # 用变形后的网格去原始特征图里“捞”像素，强制纠正脑萎缩带来的偏移
        aligned_features = F.grid_sample(
            mri_features, 
            deformed_grid, 
            mode='bilinear',       # 线性插值，保证梯度完美反向传播给 Stage 1
            padding_mode='border', # 超出边界的采样点直接取边缘值
            align_corners=False
        )

        modulated_features = aligned_features * modulation_mask
        
        return modulated_features