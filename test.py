import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# 导入你写好的模块
from models.apm_former import APM_Former_ImageOnly
from utils.dataset import get_test_dataloader
from utils.config import (
    TEST_CSV, 
    TRAIN_IMG_SIZE, 
    NUM_CLASSES, 
    FEATURE_SIZE, 
    GUIDE_CHANNELS
)

def evaluate_best_model(model, dataloader, checkpoint_path, device):
    """
    加载最佳权重并在测试集上计算完整的评估指标
    """
    print(f"========== 正在加载最佳权重 ==========")
    print(f"权重路径: {checkpoint_path}")
    
    # 1. 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("========== 开始推理评估 ==========")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 接收 4 个返回值（包含了 TV Loss 对应的 displacement_field）
            logits, _, _, _ = model(images)
            
            # 获取属于类别 1（AD 或 pMCI）的概率，用于计算 AUC
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 2. 计算混淆矩阵 (假设 0: 阴性/CN/sMCI, 1: 阳性/AD/pMCI)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # 3. 计算核心指标
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 打印指标
    print("\n========== 测试集最佳评估结果 ==========")
    print(f"Accuracy (准确率)    : {accuracy * 100:.2f}%")
    print(f"AUC (曲线下面积)     : {auc * 100:.2f}%")
    print(f"Sensitivity (灵敏度) : {sensitivity * 100:.2f}%  (TP: {tp}, FN: {fn})")
    print(f"Specificity (特异性) : {specificity * 100:.2f}%  (TN: {tn}, FP: {fp})")
    print(f"Precision (精确率)   : {precision * 100:.2f}%  (TP: {tp}, FP: {fp})")

    # 4. 绘制混淆矩阵热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0 (CN/sMCI)', 'Class 1 (AD/pMCI)'], 
                yticklabels=['Class 0 (CN/sMCI)', 'Class 1 (AD/pMCI)'],
                annot_kws={"size": 14})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    # 保存图片到本地（可选）
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 1. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 初始化模型
    model = APM_Former_ImageOnly(
        img_size=TRAIN_IMG_SIZE,
        in_channels=1,
        num_classes=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        guide_channels=GUIDE_CHANNELS,
        pretrained_swin_path=None  # 测试阶段不需要预训练权重
    ).to(device)

    # 3. 加载测试集 DataLoader
    test_loader = get_test_dataloader(
        test_csv=TEST_CSV, 
        batch_size=4, 
        target_size=TRAIN_IMG_SIZE
    )

    # 4. 设定最佳权重路径
    best_checkpoint_path = "checkpoints/best_adcn_model_8312.pth" 

    # 5. 运行评估
    evaluate_best_model(model, test_loader, best_checkpoint_path, device)

if __name__ == "__main__":
    main()