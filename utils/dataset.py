# 文件位置：utils/dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CropForegroundd,
    Resized,
    ToTensord,
    RandAffineD
)
from utils.logging_utils import get_logger  # 新增：统一日志

logger = get_logger(__name__)

class ADNIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        ADNI 纯影像版数据集加载器
        :param csv_file: 包含了 Subject_ID, Label, File_Path 的 CSV 文件
        :param transform: MONAI 的数据增强/预处理 pipeline
        """
        self.clinical_df = pd.read_csv(csv_file)
        self.transform = transform
        self.data_list = []
        
        for index, row in self.clinical_df.iterrows():
            label = int(row['Label'])
            img_path = row['File_Path']
            
            # 校验文件存在性
            if pd.notna(img_path) and os.path.exists(img_path):
                self.data_list.append({"image": img_path, "label": label})
            else:
                logger.warning(f"⚠️ 找不到文件 {img_path}，已跳过。")
        
        # 新增：校验数据集是否为空
        if len(self.data_list) == 0:
            raise ValueError(f"数据集为空！请检查 CSV 文件: {csv_file}")
        logger.info(f"✅ 成功加载 {len(self.data_list)} 个样本")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        if self.transform:
            item = self.transform(item)
            
        image_tensor = item["image"]
        label_tensor = torch.tensor(item["label"], dtype=torch.long)
        
        return image_tensor, label_tensor

# ==========================================
# 获取 DataLoader 的便捷函数（修正版）
# ==========================================
def get_adni_dataloaders(
    train_csv, 
    val_csv, 
    batch_size=4, 
    target_size=(96, 96, 96),
    num_workers=4
):
    """
    获取训练集和验证集的 DataLoader
    
    Args:
        train_csv: 训练集 CSV 路径
        val_csv: 验证集 CSV 路径
        batch_size: 批次大小（训练/验证统一）
        target_size: 目标空间尺寸
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader
    """
    # 训练集 transforms（带数据增强）
    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=target_size, mode='trilinear'),
        RandAffineD(
            keys=["image"], 
            prob=0.5, 
            rotate_range=(0.1, 0.1, 0.1), 
            translate_range=(5, 5, 5)
        ),
        ToTensord(keys=["image"])
    ])

    # 验证集 transforms（无数据增强）
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=target_size, mode='trilinear'),
        ToTensord(keys=["image"])
    ])

    # 构建数据集
    train_dataset = ADNIDataset(csv_file=train_csv, transform=train_transforms)
    val_dataset = ADNIDataset(csv_file=val_csv, transform=val_transforms)

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,      
        num_workers=num_workers,     
        pin_memory=True    
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # 修正：不要用1，和训练集保持一致
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    logger.info(f"✅ DataLoader 构建完成！")
    logger.info(f"训练集批次: {len(train_loader)}，验证集批次: {len(val_loader)}")
    
    return train_loader, val_loader  # 修正：同时返回两个