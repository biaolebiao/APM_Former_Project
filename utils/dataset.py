# 文件位置：utils/dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    CropForegroundd, Resized, ToTensord, RandAffineD, RandFlipd
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class ADNIDataset(Dataset):
    # ... (保持之前的 ADNIDataset 类不变) ...
    def __init__(self, csv_file, transform=None):
        self.clinical_df = pd.read_csv(csv_file)
        self.transform = transform
        self.data_list = []
        for index, row in self.clinical_df.iterrows():
            label = int(row['Label'])
            img_path = row['File_Path']
            if pd.notna(img_path) and os.path.exists(img_path):
                self.data_list.append({"image": img_path, "label": label})
            else:
                logger.warning(f"⚠️ 找不到文件 {img_path}，已跳过。")
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

# 🌟 核心修改：增加 use_sampler 参数，让外部决定用哪种策略
def get_adni_dataloaders(
    train_csv, 
    val_csv, 
    batch_size=4, 
    target_size=(96, 96, 96),
    num_workers=4,
    use_sampler=False  # 👈 核心开关，默认为 False
):
    # 验证集永远保持纯净
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=target_size, mode='trilinear'),
        ToTensord(keys=["image"])
    ])
    val_dataset = ADNIDataset(csv_file=val_csv, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 根据传入的 use_sampler 参数决定训练集策略
    if use_sampler:
        logger.info("🧪 启用【1:1动态加权采样 + 翻转增强】策略")
        train_transforms = Compose([
            LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image"], source_key="image"),
            Resized(keys=["image"], spatial_size=target_size, mode='trilinear'),
            RandAffineD(keys=["image"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), translate_range=(5, 5, 5)),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            ToTensord(keys=["image"])
        ])
        train_dataset = ADNIDataset(csv_file=train_csv, transform=train_transforms)

        labels = [item["label"] for item in train_dataset.data_list]
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = np.array([class_weights[t] for t in labels])
        
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, 
            sampler=sampler, num_workers=num_workers, pin_memory=True
        )
    else:
        logger.info("🧪 启用【原生数据分布 (Shuffle) + 基础增强】策略")
        train_transforms = Compose([
            LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image"], source_key="image"),
            Resized(keys=["image"], spatial_size=target_size, mode='trilinear'),
            RandAffineD(keys=["image"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), translate_range=(5, 5, 5)),
            ToTensord(keys=["image"])
        ])
        train_dataset = ADNIDataset(csv_file=train_csv, transform=train_transforms)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )

    logger.info(f"训练集批次: {len(train_loader)}，验证集批次: {len(val_loader)}")
    return train_loader, val_loader

def get_test_dataloader(
    test_csv, 
    batch_size=4, 
    target_size=(96, 96, 96),
    num_workers=4
):
    """
    专门用于生成测试集的 DataLoader。
    测试集的数据预处理必须与验证集保持绝对一致（纯净，无随机增强）。
    """
    logger.info(f"🧪 正在初始化测试集 DataLoader (来源: {test_csv})")
    
    # 预处理流程 (Transforms) 必须和 val_transforms 一模一样
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=target_size, mode='trilinear'),
        ToTensord(keys=["image"])
    ])
    
    # 实例化 Dataset
    test_dataset = ADNIDataset(csv_file=test_csv, transform=test_transforms)
    
    # 实例化 DataLoader (不需要 shuffle)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    logger.info(f"✅ 测试集加载完成，共 {len(test_loader)} 个批次。")
    return test_loader