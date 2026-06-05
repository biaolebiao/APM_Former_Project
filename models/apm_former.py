# 文件位置：models/apm_former.py
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import torch.nn.functional as F

from .anatomy_guide import AnatomyPriorGuideGeneration
from .alignment import AnatomyGuidedAlignment

class APM_Former_ImageOnly(nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, num_classes=2, feature_size=24, guide_channels=18, pretrained_swin_path=None,dropout1=0.3, dropout2=0.2, fc_hidden=64,
                 use_anatomy_prior=True,
                 use_dcn_alignment=True
                 ):
        super().__init__()
        self.use_anatomy_prior = use_anatomy_prior
        self.use_dcn_alignment = use_dcn_alignment

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
        # mri_feature_channels = feat_mid_channels + feat_deep_channels

        self.shallow_downsample = nn.Sequential(
            # kernel_size=3, stride=2, padding=1 可以完美将空间尺寸减半 (48 -> 24)
            nn.Conv3d(feat_shallow_channels, feat_shallow_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(feat_shallow_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 加上科学初始化代码，防止新卷积层开局输出巨大噪声
        nn.init.kaiming_normal_(self.shallow_downsample[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.shallow_downsample[0].bias)

        
        #  新增核心：特征平滑与降维层 (将 168 维降到 48 维) 
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(mri_feature_channels, feat_mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feat_mid_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.anatomy_alignment = AnatomyGuidedAlignment(
            mri_channels=feat_mid_channels, # 👇 修改：这里原来是 mri_feature_channels，现在改为降维后的 feat_mid_channels
            guide_channels=guide_channels
        )
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(feat_mid_channels, fc_hidden), # 👇 修改：这里原来是 mri_feature_channels，现在改为降维后的 feat_mid_channels
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(fc_hidden, num_classes)
        )
        
        self.attention_conv = nn.Conv3d(guide_channels, 1, kernel_size=1, padding=0)
        
    def forward(self, mri_image):
        # 1. Swin特征提取 (这是所有消融实验都要用到的 Baseline)
        hidden_states = self.swin_backbone.swinViT(mri_image)
        feat_shallow = hidden_states[0]
        feat_mid = hidden_states[1]
        feat_deep = hidden_states[2]
        
        # 使用可学习的卷积进行特征浓缩
        feat_shallow_down = self.shallow_downsample(feat_shallow)
        feat_deep_up = F.interpolate(feat_deep, size=feat_mid.shape[2:], mode='trilinear', align_corners=False)
        # swin_feature = torch.cat([feat_shallow_down, feat_mid, feat_deep_up], dim=1)
        # 👇 修改：拼接后立刻进行特征平滑和降维
        swin_feature_raw = torch.cat([feat_shallow_down, feat_mid, feat_deep_up], dim=1)
        swin_feature = self.feature_fusion(swin_feature_raw)
        
        # 🔴 关键点：设定默认的占位返回值
        # 为了保证无论走哪个 if 分支，最后 return 的 4 个变量都存在，防止外面的代码解包报错
        aligned_features = swin_feature
        spatial_attention = torch.ones_like(swin_feature[:, 0:1, ...]) # 默认全是1，乘了等于没乘
        displacement_field = torch.zeros(swin_feature.shape[0], 3, *swin_feature.shape[2:], device=swin_feature.device) # 默认形变场为0

        # 2. 核心消融逻辑分支
        if self.use_anatomy_prior and self.use_dcn_alignment:
            # 【情形 D/C：完全体 / 只有DCN】先验图谱 + DCN形变对齐
            guide_map = self.anatomy_guide_gen(mri_image) 
            aligned_features, displacement_field = self.anatomy_alignment(swin_feature, guide_map)
            
            # 空间注意力
            guide_down = F.interpolate(guide_map, size=aligned_features.shape[2:], mode='trilinear', align_corners=False)
            spatial_attention = torch.sigmoid(self.attention_conv(guide_down))
            focused_features = aligned_features * spatial_attention
            
        elif self.use_anatomy_prior and not self.use_dcn_alignment:
            # 【情形 B：仅先验】生成了先验图谱，但只做注意力加权，不做DCN形变
            guide_map = self.anatomy_guide_gen(mri_image)
            
            # 空间注意力 (直接作用在未对齐的 swin_feature 上)
            guide_down = F.interpolate(guide_map, size=swin_feature.shape[2:], mode='trilinear', align_corners=False)
            spatial_attention = torch.sigmoid(self.attention_conv(guide_down))
            focused_features = swin_feature * spatial_attention
            
        else:
            # 【情形 A：纯Baseline】什么都不加，直接拿 Swin 特征去分类
            focused_features = swin_feature

        # 3. 分类头 (永远都会执行)
        pooled_features = self.global_pool(focused_features)
        flattened_features = torch.flatten(pooled_features, 1)
        logits = self.classifier(flattened_features)
        
        # 无论经过哪个分支，都会返回这4个变量，你的 train_epoch 和 val_epoch 完全不需要改代码
        return logits, aligned_features, spatial_attention, displacement_field
    
    def train(self, mode=True):
        super().train(mode)