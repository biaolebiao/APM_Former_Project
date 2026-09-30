import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 导入你的模型和配置
from models.apm_former import APM_Former_ImageOnly
from utils.dataset import get_test_dataloader
from utils.config import TEST_CSV, TRAIN_IMG_SIZE, NUM_CLASSES

class GradCAM3D:
    """
    针对 3D 医疗影像的 Grad-CAM 实现
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output[0] if isinstance(output, tuple) else output

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        model_outputs = self.model(input_tensor)
        logits = model_outputs[0]
        
        if target_class is None:
            target_class = logits[0].argmax().item()

        self.model.zero_grad()
        target_score = logits[0, target_class]
        
        activations = self.activations
        
        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=activations,
            grad_outputs=torch.ones_like(target_score),
            retain_graph=True,
            create_graph=False
        )[0]

        gradients_np = grads.cpu().data.numpy()[0]
        activations_np = activations.cpu().data.numpy()[0]

        weights = np.mean(gradients_np, axis=(1, 2, 3))

        cam = np.zeros(activations_np.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations_np[i]

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam, target_class, model_outputs


def visualize_grid_gradcam(model, dataloader, device, target_layer, slice_axis=2, slices=[35, 48, 60]):
    # 设置你在验证集表现最好的动态阈值
    DYNAMIC_THRESHOLD = 0.40 
    
    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    grad_cam = GradCAM3D(model, target_layer)
    save_dir = "visualizations_candidates"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🔍 开始全量扫描测试集 (阈值: {DYNAMIC_THRESHOLD})... 结果将保存在 {save_dir} 文件夹中")
    
    patient_count = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images_dev = images.to(device)
        
        with torch.no_grad():
            outputs = model(images_dev)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs_pmci = F.softmax(logits, dim=1)[:, 1].cpu()
            preds = (probs_pmci > DYNAMIC_THRESHOLD).long()
            
        for i in range(len(labels)):
            patient_count += 1
            true_label = labels[i].item()
            pred_label = preds[i].item()
            prob = probs_pmci[i].item()
            
            # 只处理预测完全正确的样本 (TN 或 TP)
            if true_label == pred_label:
                img_tensor = images[i:i+1].clone().to(device)
                img_tensor.requires_grad = True
                orig_size = img_tensor.shape[2:]
                
                # 生成 CAM
                cam_raw, _, _ = grad_cam.generate(img_tensor)
                cam_tensor = torch.from_numpy(cam_raw).unsqueeze(0).unsqueeze(0)
                cam_resized = F.interpolate(cam_tensor, size=orig_size, mode='trilinear', align_corners=False)
                cam_3d = cam_resized.squeeze().numpy()
                img_np = img_tensor[0, 0].detach().cpu().numpy()
                
                # 开始绘图 (1行 x 3列，针对单个病人)
                bg_color = '#0D1424'
                text_color = '#E5E7EB'
                fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 3.5), facecolor=bg_color)
                
                patient_type = "pMCI" if true_label == 1 else "sMCI"
                
                for j, s_idx in enumerate(slices):
                    # 提取并旋转切片
                    if slice_axis == 0:
                        s_img, s_cam = img_np[s_idx, :, :], cam_3d[s_idx, :, :]
                    elif slice_axis == 1:
                        s_img, s_cam = img_np[:, s_idx, :], cam_3d[:, s_idx, :]
                    else:
                        s_img, s_cam = img_np[:, :, s_idx], cam_3d[:, :, s_idx]
                        
                    s_img = np.rot90(s_img, k=-1)
                    s_cam = np.rot90(s_cam, k=-1)
                    
                    axes[j].imshow(s_img, cmap='gray', interpolation='lanczos')
                    axes[j].imshow(s_cam, cmap='jet', alpha=0.55, interpolation='lanczos', vmin=0, vmax=1)
                    axes[j].axis('off')
                    
                    axes[j].set_title(f'Slice {s_idx}', color=text_color, fontsize=16, pad=10)
                    
                    if j == 0:
                        axes[j].text(-0.15, 0.5, f"True {patient_type}\n(Pred: {patient_type})", 
                                     color=text_color, fontsize=14, va='center', ha='center', rotation=90, transform=axes[j].transAxes)
                
                plt.tight_layout()
                file_name = f"Patient_{patient_count:03d}_{patient_type}.png"
                plt.savefig(os.path.join(save_dir, file_name), dpi=200, bbox_inches='tight', facecolor=bg_color)
                plt.close(fig) # 关闭画布防止内存泄漏
                print(f"✅ 已保存: {file_name}")

    print(f"🎉 扫描结束！请前往 {save_dir} 文件夹挑选完美的对比图。")
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 为了确保能抓到一个 sMCI 和一个 pMCI，可以稍微把 batch_size 设大一点（比如 8）
    test_loader = get_test_dataloader(TEST_CSV, batch_size=8, target_size=TRAIN_IMG_SIZE)

    model = APM_Former_ImageOnly(
        img_size=TRAIN_IMG_SIZE,
        in_channels=1,
        num_classes=NUM_CLASSES,
        feature_size=24,
        guide_channels=18,
        use_anatomy_prior=True, 
        use_dcn_alignment=True
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/best.pth", map_location=device, weights_only=True))

    target_layer = model.fusion_norm_act

    # 设定你要观察的切片索引 (假设你的输入深度是 96，这里取 35, 48, 60 三层)
    # 你可以根据实际脑部核心区域（如海马体所在层）自行修改这些数值
    target_slices = [35, 48, 60] 

    visualize_grid_gradcam(model, test_loader, device, target_layer, slice_axis=2, slices=target_slices)