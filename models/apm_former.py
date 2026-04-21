# 文件位置：models/apm_former.py
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import torch.nn.functional as F

from .anatomy_guide import AnatomyPriorGuideGeneration
from .alignment import AnatomyGuidedAlignment

class APM_Former_ImageOnly(nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, num_classes=2, feature_size=48, guide_channels=18, pretrained_swin_path=None,dropout1=0.3, dropout2=0.2, fc_hidden=64):
        super().__init__()
        self.guide_channels = guide_channels
        
        # 🚨 核心修复1：关闭 use_checkpoint（解决CheckpointError）
        self.swin_backbone = SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=False,  # 必须关闭！！！
        )
        
        if pretrained_swin_path is not None:
            checkpoint = torch.load(pretrained_swin_path, map_location='cpu')
            self.swin_backbone.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"加载SwinUNETR预训练权重：{pretrained_swin_path}")
        
        self.anatomy_guide_gen = AnatomyPriorGuideGeneration(k_channels=guide_channels)
        
        # 特征通道配置
        feat_shallow_channels = feature_size 
        feat_mid_channels = feature_size * 2
        feat_deep_channels = feature_size * 4
        mri_feature_channels = feat_shallow_channels + feat_mid_channels + feat_deep_channels
        
        self.anatomy_alignment = AnatomyGuidedAlignment(
            mri_channels=mri_feature_channels, 
            guide_channels=guide_channels
        )
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(mri_feature_channels, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(fc_hidden, num_classes)
        )
        
        self.attention_conv = nn.Conv3d(guide_channels, 1, kernel_size=1, padding=0)
        
    def forward(self, mri_image):
        # 1. 解剖先验
        guide_map = self.anatomy_guide_gen(mri_image) 
        
        # 2. Swin特征提取
        hidden_states = self.swin_backbone.swinViT(mri_image)
        feat_shallow = hidden_states[0]
        feat_mid = hidden_states[1]
        feat_deep = hidden_states[2]
        feat_shallow_down = F.adaptive_avg_pool3d(feat_shallow, output_size=feat_mid.shape[2:])
        feat_deep_up = F.interpolate(feat_deep, size=feat_mid.shape[2:], mode='trilinear', align_corners=False)
        swin_feature = torch.cat([feat_shallow_down, feat_mid, feat_deep_up], dim=1)
        
        aligned_features = self.anatomy_alignment(swin_feature, guide_map)
        
        # 3. 空间注意力
        guide_down = F.interpolate(guide_map, size=aligned_features.shape[2:], mode='trilinear', align_corners=False)
        spatial_attention = torch.sigmoid(self.attention_conv(guide_down))
        focused_features = aligned_features * spatial_attention
        
        # 4. 分类
        pooled_features = self.global_pool(focused_features)
        flattened_features = torch.flatten(pooled_features, 1)
        logits = self.classifier(flattened_features)
        
        return logits, aligned_features, spatial_attention
    
    def train(self, mode=True):
        super().train(mode)