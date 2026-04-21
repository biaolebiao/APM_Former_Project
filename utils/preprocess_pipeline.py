import ants
import os
import logging
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import PREPROCESS_CONFIG

# ==============================
# 配置区域（从config.py导入）
# ==============================
# 日志配置（保持原有配置，专门用于预处理）
LOG_FILE = "preprocessing_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 是否启用并行处理（从config.py导入）
USE_PARALLEL = PREPROCESS_CONFIG["use_parallel"]
MAX_WORKERS = PREPROCESS_CONFIG["max_workers"]  # 仅当USE_PARALLEL=True时生效，None为自动检测


def preprocess_single_subject(
    input_nii_path, 
    mni_template_path, 
    mni_mask_path, 
    output_dir,
    overwrite=False
):
    """
    处理单个受试者的MRI数据（内部函数，供批量调用）
    """
    base_name = os.path.basename(input_nii_path).split('.nii')[0]
    
    # 构建输出路径（保持输入目录的子结构）
    # 例如输入: /input/sub1/file.nii -> 输出: /output/sub1/file_final.nii
    relative_path = os.path.relpath(os.path.dirname(input_nii_path), start=INPUT_ROOT_DIR)
    subject_output_dir = os.path.join(output_dir, relative_path)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # 最终输出文件路径
    final_out_path = os.path.join(subject_output_dir, f"{base_name}_final_preprocessed.nii.gz")
    
    # 检查是否已处理（避免重复）
    if os.path.exists(final_out_path) and not overwrite:
        logger.info(f"跳过已处理文件: {base_name}")
        return final_out_path

    try:
        logger.info(f"开始处理: {base_name}")
        
        # 1. 读取图像
        moving_img = ants.image_read(input_nii_path)
        fixed_template = ants.image_read(mni_template_path)
        template_mask = ants.image_read(mni_mask_path)

        # 2. N4偏置场校正
        n4_corrected_img = ants.n4_bias_field_correction(moving_img)
        n4_out_path = os.path.join(subject_output_dir, f"{base_name}_n4.nii.gz")
        ants.image_write(n4_corrected_img, n4_out_path)

        # 3. 空间配准（SyN算法）
        registration_result = ants.registration(
            fixed=fixed_template, 
            moving=n4_corrected_img, 
            type_of_transform='SyN'
        )
        registered_img = registration_result['warpedmovout']
        reg_out_path = os.path.join(subject_output_dir, f"{base_name}_registered.nii.gz")
        ants.image_write(registered_img, reg_out_path)

        # 4. 剥头骨（基于MNI模板Mask）
        brain_extracted_img = registered_img * template_mask
        # 新增：全局强度归一化（0-1）
        brain_extracted_img = (brain_extracted_img - brain_extracted_img.min()) / (brain_extracted_img.max() - brain_extracted_img.min())
        # 新增：高斯平滑去噪（可选，核大小根据需求调）
        brain_extracted_img = ants.smooth_image(brain_extracted_img, sigma=0.5)
        ants.image_write(brain_extracted_img, final_out_path)
        
        logger.info(f"处理完成: {base_name} -> 保存至 {final_out_path}")
        return final_out_path

    except Exception as e:
        logger.error(f"处理失败: {base_name}, 错误信息: {str(e)}", exc_info=True)
        return None


def batch_preprocess(
    input_root_dir,
    mni_template_path,
    mni_mask_path,
    output_dir,
    overwrite=False
):
    """
    批量预处理入口函数
    """
    global INPUT_ROOT_DIR
    INPUT_ROOT_DIR = input_root_dir  # 保存全局根路径，用于构建相对路径
    
    # 1. 递归查找所有.nii和.nii.gz文件
    logger.info(f"正在扫描输入目录: {input_root_dir}")
    nii_files = sorted(glob(os.path.join(input_root_dir, "**", "*.nii*"), recursive=True))
    # 过滤掉非标准文件（避免._开头的Mac临时文件等）
    nii_files = [f for f in nii_files if not os.path.basename(f).startswith("._")]
    
    if not nii_files:
        logger.error("未找到任何.nii或.nii.gz文件，请检查输入路径！")
        return
    
    logger.info(f"共找到 {len(nii_files)} 个待处理文件")

    # 2. 执行处理
    results = []
    if USE_PARALLEL:
        logger.info(f"启用并行处理，最大进程数: {MAX_WORKERS or '自动'}")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    preprocess_single_subject,
                    f, mni_template_path, mni_mask_path, output_dir, overwrite
                ): f for f in nii_files
            }
            # 用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(futures), desc="批量处理进度"):
                results.append(future.result())
    else:
        logger.info("使用顺序处理（如需加速请设置USE_PARALLEL=True）")
        for f in tqdm(nii_files, desc="批量处理进度"):
            res = preprocess_single_subject(
                f, mni_template_path, mni_mask_path, output_dir, overwrite
            )
            results.append(res)

    # 3. 统计结果
    success_count = sum(1 for r in results if r is not None)
    logger.info(f"批量处理结束！成功: {success_count}/{len(nii_files)}，详细日志见 {LOG_FILE}")


# ==========================================
# 执行入口（使用config.py中的配置）
# ==========================================
if __name__ == "__main__":
    # --------------------------
    # 从config.py导入路径配置
    # --------------------------
    INPUT_ROOT_DIR = PREPROCESS_CONFIG["input_root"]
    MNI_TEMPLATE = PREPROCESS_CONFIG["mni_template"]
    MNI_MASK = PREPROCESS_CONFIG["mni_mask"]
    OUTPUT_FOLDER = PREPROCESS_CONFIG["output_dir"]
    OVERWRITE = PREPROCESS_CONFIG["overwrite"]

    # --------------------------
    # 运行批量处理
    # --------------------------
    try:
        batch_preprocess(
            input_root_dir=INPUT_ROOT_DIR,
            mni_template_path=MNI_TEMPLATE,
            mni_mask_path=MNI_MASK,
            output_dir=OUTPUT_FOLDER,
            overwrite=OVERWRITE  # 从config.py导入
        )
    except Exception as e:
        logger.critical(f"批量处理主程序崩溃: {str(e)}", exc_info=True)