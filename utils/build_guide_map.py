# 文件位置：build_guide_map.py
import ants
import numpy as np
import os
from config import GUIDE_MAP_CONFIG
from logging_utils import get_logger  # 新增：统一日志

logger = get_logger(__name__)

def build_anatomy_guide_map(pm_file_paths, reference_mri_path, output_path):
    """
    将 Julich-Brain 的多个概率图谱合并为 K 通道的 NumPy 张量，并保存为文件
    
    Args:
        pm_file_paths: 概率图谱文件路径列表
        reference_mri_path: 参考 MRI 路径（用于空间对齐）
        output_path: 输出 .npy 文件路径
    
    Returns:
        guide_map_tensor: 生成的解剖向导图张量 (C, D, H, W)
    
    Raises:
        FileNotFoundError: 关键文件不存在
        ValueError: 图像形状/数值不合法
        RuntimeError: 图像读取/重采样失败
    """
    # ==========================================
    # 第一层校验：文件存在性
    # ==========================================
    logger.info(f"🔍 开始校验输入文件...")
    
    # 1. 校验参考 MRI
    if not os.path.exists(reference_mri_path):
        raise FileNotFoundError(f"参考 MRI 不存在: {reference_mri_path}")
    logger.info(f"✅ 参考 MRI 存在: {reference_mri_path}")
    
    # 2. 校验所有概率图谱文件
    missing_files = [f for f in pm_file_paths if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"以下概率图谱文件不存在: {missing_files}")
    logger.info(f"✅ 所有 {len(pm_file_paths)} 个概率图谱文件存在")
    
    # 3. 校验输出目录
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"✅ 输出目录就绪: {output_dir}")

    # ==========================================
    # 第二层校验：图像读取 & 形状一致性
    # ==========================================
    logger.info(f"🔍 开始加载图像并校验形状...")
    
    # 1. 加载参考 MRI 并获取空间信息
    try:
        ref_img = ants.image_read(reference_mri_path)
        ref_shape = ref_img.shape  # (H, W, D) 注意 ANTS 的维度顺序
        logger.info(f"✅ 参考 MRI 加载成功，形状: {ref_shape}")
    except Exception as e:
        raise RuntimeError(f"参考 MRI 读取失败: {str(e)}")

    channel_list = []
    for i, pm_path in enumerate(pm_file_paths):
        logger.info(f"正在处理通道 {i+1}/{len(pm_file_paths)}: {os.path.basename(pm_path)}")
        
        # 2. 校验概率图谱读取
        try:
            pm_img = ants.image_read(pm_path)
        except Exception as e:
            logger.error(f"❌ 通道 {i+1} 读取失败: {str(e)}，跳过该通道")
            continue
        
        # 3. 重采样到参考 MRI 空间
        try:
            pm_resampled = ants.resample_image_to_target(
                pm_img, ref_img, interp_type='linear'
            )
        except Exception as e:
            logger.error(f"❌ 通道 {i+1} 重采样失败: {str(e)}，跳过该通道")
            continue
        
        # 4. 校验重采样后的形状一致性
        if pm_resampled.shape != ref_shape:
            logger.warning(
                f"⚠️ 通道 {i+1} 重采样后形状 {pm_resampled.shape} "
                f"与参考形状 {ref_shape} 不一致，已跳过"
            )
            continue
        
        # 5. 校验数值范围（概率图谱应在 0~1 之间）
        pm_array = pm_resampled.numpy()
        if pm_array.min() < 0 or pm_array.max() > 1.01:  # 允许微小浮点误差
            logger.warning(
                f"⚠️ 通道 {i+1} 数值范围异常: [{pm_array.min():.4f}, {pm_array.max():.4f}] "
                f"(期望 0~1)，仍将保留"
            )
        
        channel_list.append(pm_array)
    
    # 6. 校验有效通道数
    if len(channel_list) == 0:
        raise ValueError("所有通道均处理失败，请检查输入文件！")
    if len(channel_list) < len(pm_file_paths):
        logger.warning(
            f"⚠️ 仅成功处理 {len(channel_list)}/{len(pm_file_paths)} 个通道"
        )

    # ==========================================
    # 第三层校验：张量生成 & 合法性
    # ==========================================
    logger.info(f"🔍 开始生成解剖向导图张量...")
    
    # 1. 拼接通道并调整维度 (K, H, W, D) -> (K, D, H, W)
    guide_map_tensor = np.stack(channel_list, axis=0).transpose(0, 3, 1, 2)
    
    # 2. 校验最终张量形状
    expected_shape = (len(channel_list), ref_shape[2], ref_shape[0], ref_shape[1])
    if guide_map_tensor.shape != expected_shape:
        raise ValueError(
            f"最终张量形状 {guide_map_tensor.shape} 与期望 {expected_shape} 不一致！"
        )
    
    # 3. 校验张量数值（无 NaN/Inf）
    if np.isnan(guide_map_tensor).any():
        logger.warning("⚠️ 张量中存在 NaN 值！")
    if np.isinf(guide_map_tensor).any():
        logger.warning("⚠️ 张量中存在 Inf 值！")

    # ==========================================
    # 保存 & 完成
    # ==========================================
    np.save(output_path, guide_map_tensor)
    logger.info(f"✅ 解剖向导图构建成功！")
    logger.info(f"   张量形状: {guide_map_tensor.shape}")
    logger.info(f"   数值范围: [{guide_map_tensor.min():.4f}, {guide_map_tensor.max():.4f}]")
    logger.info(f"   已保存至: {output_path}")
    
    return guide_map_tensor

# ==========================================
# 执行脚本（使用config.py中的配置）
# ==========================================
if __name__ == "__main__":
    try:
        # 从config.py导入配置
        base_dir = GUIDE_MAP_CONFIG["base_dir"]
        core_areas = GUIDE_MAP_CONFIG["core_areas"]
        REF_MRI = GUIDE_MAP_CONFIG["ref_mri"]
        OUTPUT_FILE = GUIDE_MAP_CONFIG["output_file"]
        
        # 生成概率图谱路径列表
        SELECTED_PMS = []
        for area in core_areas:
            lh_path = os.path.join(base_dir, area, f"{area}_lh_MNI152.nii.gz")
            rh_path = os.path.join(base_dir, area, f"{area}_rh_MNI152.nii.gz")
            SELECTED_PMS.append(lh_path)
            SELECTED_PMS.append(rh_path)
        
        # 打印路径示例（仅调试用）
        logger.info(f"生成的概率图谱路径示例：")
        if len(SELECTED_PMS) >= 2:
            logger.info(f"  {SELECTED_PMS[0]}")
            logger.info(f"  {SELECTED_PMS[1]}")
        logger.info(f"总共生成了 {len(SELECTED_PMS)} 个通道的路径")
        
        # 运行构建
        build_anatomy_guide_map(SELECTED_PMS, REF_MRI, OUTPUT_FILE)
    
    except Exception as e:
        logger.error(f"❌ 解剖向导图构建失败: {str(e)}", exc_info=True)
        raise