import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

# 导入你的模型和配置
from models.apm_former import APM_Former_ImageOnly
from utils.dataset import get_test_dataloader
from utils.config import TEST_CSV, TRAIN_IMG_SIZE, NUM_CLASSES

def visualize_model_outputs_premium(model, dataloader, device, slice_axis=2, slice_idx=None):
    model.eval()
    
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        logits, aligned_features, spatial_attention, displacement_field = model(images)
        
        orig_size = images.shape[2:] 
        spatial_attention = F.interpolate(spatial_attention, size=orig_size, mode='trilinear', align_corners=False)
        displacement_field = F.interpolate(displacement_field, size=orig_size, mode='trilinear', align_corners=False)
    
    img = images[0, 0].cpu().numpy()
    att = spatial_attention[0, 0].cpu().numpy()
    disp = displacement_field[0].cpu().numpy() 
    
    if slice_idx is None:
        slice_idx = img.shape[slice_axis] // 2
        
    if slice_axis == 0:
        img_slice, att_slice = img[slice_idx, :, :], att[slice_idx, :, :]
        dy_slice, dx_slice = disp[1, slice_idx, :, :], disp[2, slice_idx, :, :]
    elif slice_axis == 1: 
        img_slice, att_slice = img[:, slice_idx, :], att[:, slice_idx, :]
        dy_slice, dx_slice = disp[0, slice_idx, :, :], disp[2, slice_idx, :, :]
    else:                 
        img_slice, att_slice = img[:, :, slice_idx], att[:, :, slice_idx]
        dy_slice, dx_slice = disp[0, :, :, slice_idx], disp[1, :, :, slice_idx]

    # 🌟 修复：去除强制挖空的 masked_where，只做归一化
    # 这样底层就会是很均匀的蓝色，高亮区是红黄，完美对标 Grad-CAM
    att_slice = (att_slice - att_slice.min()) / (att_slice.max() - att_slice.min() + 1e-8)
    
    h, w = img_slice.shape
    box_size = 40  
    y1, y2 = h//2 - box_size//2, h//2 + box_size//2
    x1, x2 = w//2 - box_size//2, w//2 + box_size//2

    step = 8 
    grid_y, grid_x = np.mgrid[y1:y2:step, x1:x2:step]
    dy_sampled = dy_slice[y1:y2:step, x1:x2:step]
    dx_sampled = dx_slice[y1:y2:step, x1:x2:step]

    # ================= 开始绘图 =================
    bg_color = '#0D1424'
    text_color = '#E5E7EB'
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=bg_color)
    label_str = 'pMCI' if labels[0].item() == 1 else 'sMCI'
    
    for ax in axes:
        ax.set_facecolor(bg_color)
        ax.axis('off')

    # 图 1: 原始 MRI
    axes[0].imshow(img_slice, cmap='gray', interpolation='lanczos')
    axes[0].set_title(f"{label_str} - Original MRI", color=text_color, fontsize=15, pad=15)

    # 图 2: 纯正平滑注意力热力图 (去掉 mask 后，灰块彻底消失)
    axes[1].imshow(img_slice, cmap='gray', interpolation='lanczos')
    im_att = axes[1].imshow(att_slice, cmap='jet', alpha=0.55, interpolation='lanczos') 
    axes[1].set_title(f"{label_str} - Anatomy Attention", color=text_color, fontsize=15, pad=15)
    
    cbar1 = plt.colorbar(im_att, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors=text_color, labelsize=10)
    cbar1.outline.set_edgecolor(text_color)

    # 图 3: DCN 形变场 (优化箭头形态，让它看起来更专业)
    axes[2].imshow(img_slice, cmap='gray', interpolation='lanczos')
    
    rect = patches.Rectangle((x1, y1), box_size, box_size, linewidth=1.5, edgecolor='yellow', facecolor='none', linestyle='--')
    axes[2].add_patch(rect)
    
    # 调整箭头：变细长一点，取消粗大感
    axes[2].quiver(grid_x, grid_y, dx_sampled, -dy_sampled, color='#FF3333', 
                   scale=3.5, alpha=1.0, width=0.007, headwidth=4, headlength=5)
    axes[2].set_title(f"{label_str} - Deformable Field (ROI)", color=text_color, fontsize=15, pad=15)

    plt.tight_layout()
    
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'visual_evidence_axis{slice_axis}_slice{slice_idx}_final.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    print(f"✅ 图片已成功保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_dataloader(TEST_CSV, batch_size=4, target_size=TRAIN_IMG_SIZE)

    # 实例化完全体模型
    model = APM_Former_ImageOnly(
        img_size=TRAIN_IMG_SIZE,
        in_channels=1,
        num_classes=NUM_CLASSES,
        feature_size=24,
        guide_channels=18,
        use_anatomy_prior=True, 
        use_dcn_alignment=True
    ).to(device)

    # 🌟 修复警告：加入 weights_only=True
    model.load_state_dict(torch.load("checkpoints/best_model3_APM_Former.pth", map_location=device, weights_only=True))

    # 运行画图
    visualize_model_outputs_premium(model, test_loader, device, slice_axis=2)