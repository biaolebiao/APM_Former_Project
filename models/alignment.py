# 文件位置：models/alignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomyGuidedAlignment(nn.Module):
    def __init__(self, mri_channels, guide_channels=18, displacement_scale=0.1):
        super().__init__()
        self.guide_channels = guide_channels
        self.displacement_scale = displacement_scale
        
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
        
        # 2. 偏移量预测头 (强制输出 3 通道)
        self.offset_head = nn.Conv3d(in_channels // 4, 3, kernel_size=3, padding=1)
        
        # 3. 调制掩码预测头 (强制输出 1 通道)
        self.mask_head = nn.Conv3d(in_channels // 4, 1, kernel_size=3, padding=1)
        
        # 0初始化，防止冷启动崩溃
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    def forward(self, mri_features, anatomy_guide_map):
        # 空间分辨率对齐
        guide_down = F.interpolate(anatomy_guide_map, size=mri_features.shape[2:], mode='trilinear', align_corners=False)
        fused_features = torch.cat([mri_features, guide_down], dim=1)
        
        # 提取共享特征
        shared_feat = self.shared_net(fused_features)
        
        # 预测形变场并激活
        raw_displacement = self.offset_head(shared_feat)
        displacement_field = torch.tanh(raw_displacement) * self.displacement_scale
        
        # 预测调制掩码并约束到 0~1
        raw_mask = self.mask_head(shared_feat)
        modulation_mask = torch.sigmoid(raw_mask) 
        
        # 构建基础网格
        B, C, D_feat, H_feat, W_feat = mri_features.shape
        vectors = [
            torch.linspace(-1, 1, D_feat, device=mri_features.device),
            torch.linspace(-1, 1, H_feat, device=mri_features.device),
            torch.linspace(-1, 1, W_feat, device=mri_features.device)
        ]
        grid_d, grid_h, grid_w = torch.meshgrid(vectors, indexing='ij')
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        # 叠加偏移量
        displacement_field = displacement_field.permute(0, 2, 3, 4, 1)
        deformed_grid = base_grid + displacement_field
        
        # 医学解剖对齐重采样
        aligned_features = F.grid_sample(
            mri_features, 
            deformed_grid, 
            mode='bilinear',       
            padding_mode='border', 
            align_corners=False
        )
        
        
        # 🌟 修改这里：增加残差连接 (Residual Connection)
        # 即使 modulation_mask 初期没学好，原始特征 aligned_features 也能无损传导梯度
        modulated_features = aligned_features + aligned_features * modulation_mask
        
        return modulated_features,displacement_field