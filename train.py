import os
# 告诉系统，只让 PyTorch 看到 1 号显卡（注意：这行必须在 import torch 之前执行）
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch
import torch.nn as nn
import torch.optim as optim

# 导入你写好的数据加载器和主网络
from utils.dataset import get_adni_dataloaders
from models import APM_Former_ImageOnly
import matplotlib.pyplot as plt

print("所有库导入成功！")

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

epochs = 50
batch_size = 4        # 如果显存不够，可以改成 2 或 1
learning_rate = 1e-4
best_val_acc = 0.0    

# 创建保存权重的文件夹
os.makedirs("checkpoints", exist_ok=True)
print("基础配置完成！")

print("正在加载数据集...")

# 加载训练集
train_loader = get_adni_dataloaders(
    csv_path="data/clinical/train.csv",  
    img_dir="data/preprocessed/sMCI_processed", 
    batch_size=batch_size
)

# 加载验证集
val_loader = get_adni_dataloaders(
    csv_path="data/clinical/val.csv",    
    img_dir="data/preprocessed/sMCI_processed",
    batch_size=1  
)

print(f"✅ 数据加载完毕！训练集批次数量: {len(train_loader)}, 验证集批次数量: {len(val_loader)}")

# --- Jupyter 专属探针：偷偷看一眼第一个 Batch 的数据形状 ---
sample_images, sample_labels = next(iter(train_loader))
print(f"📸 检查张量形状 -> 图像: {sample_images.shape}, 标签: {sample_labels.shape}")


from monai.losses import FocalLoss
print("正在初始化 APM-Former 网络...")
model = APM_Former_ImageOnly(
    img_size=(96, 96, 96), 
    in_channels=1, 
    num_classes=2,   
    feature_size=48, 
    guide_channels=18
).to(device)

# 自动计算类别权重 (sMCI有526例，pMCI有298例)
# num_sMCI = 526
# num_pMCI = 298
# weight_pMCI = num_sMCI / num_pMCI  # 约 1.765

# class 0 权重为 1.0; class 1 权重为 1.765
weights = torch.tensor([1.0, 1.765]).to(device) 
criterion = FocalLoss(weight=weights, gamma=2.0, to_onehot_y=True)

learning_rate = 5e-5  
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
print("✅ 网络与优化器准备就绪！")


train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []
print("🚀 启动训练...")

for epoch in range(epochs):
    model.train() 
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 梯度清零 -> 前向传播 -> 计算 Loss -> 反向传播 -> 更新参数
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 统计指标
        total_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # 每隔几个 batch 打印一次进度
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_acc = 100 * correct_train / total_train
    avg_train_loss = total_train_loss / len(train_loader)
    
    # ---------------- 验证集评估 ----------------
    model.eval() 
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    val_acc = 100 * correct_val / total_val
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\n=== Epoch {epoch+1} 总结 ===")
    print(f"训练集 Loss：{avg_train_loss:.4f}, 训练集 Acc: {train_acc:.2f}%, 验证集 Loss: {avg_val_loss:.4f}, 验证集 Acc: {val_acc:.2f}%")
    
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    
    # 保存最佳权重
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = f"checkpoints/best_apm_former.pth"
        torch.save(model.state_dict(), save_path)
        print(f"🏆 发现新纪录！已将最佳权重保存至: {save_path} (Acc: {best_val_acc:.2f}%)")
    print("-" * 50)



    plt.rcParams.update({'font.size': 12})

# 创建一个 1行2列 的画布，尺寸为 14x5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ==========================================
# 图 1：Loss 下降曲线
# ==========================================
# x 轴就是 Epoch 的数量
epochs_range = range(1, len(train_loss_history) + 1)

ax1.plot(epochs_range, train_loss_history, label='Training Loss', color='#1f77b4', linewidth=2)
ax1.plot(epochs_range, val_loss_history, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')
ax1.set_title('Training and Validation Loss', fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.7) # 加个虚线网格，显得更专业

# ==========================================
# 图 2：Accuracy 上升曲线
# ==========================================
ax2.plot(epochs_range, train_acc_history, label='Training Accuracy', color='#2ca02c', linewidth=2)
ax2.plot(epochs_range, val_acc_history, label='Validation Accuracy', color='#d62728', linewidth=2, linestyle='--')
ax2.set_title('Training and Validation Accuracy', fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.7)

# 调整子图之间的间距
plt.tight_layout()

# 保存高清图片到本地，准备贴进 Word 论文里
plt.savefig("checkpoints/training_curves.png", dpi=300, bbox_inches='tight')

# 在 Jupyter 里展示出来
plt.show()