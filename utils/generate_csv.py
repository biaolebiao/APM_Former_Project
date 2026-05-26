# 文件位置：utils/generate_csv.py

import os
import glob
import pandas as pd
import re  # 引入正则表达式库
from config import DATA_SOURCES, FILE_SUFFIX, SUBJECT_ID_REGEX, ALL_DATA_CSV
from logging_utils import get_logger

logger = get_logger(__name__)

def generate_master_csv():
    """
    自动扫描预处理文件夹，生成包含 Subject_ID 和 Label 的总 CSV 文件
    """
    # ==========================================
    # 1. 配置你的数据路径和标签
    # ==========================================
    data_sources = DATA_SOURCES
    file_suffix = FILE_SUFFIX
    subject_id_regex = SUBJECT_ID_REGEX
    output_path = ALL_DATA_CSV
    
    records = []
    
    # ==========================================
    # 2. 遍历文件夹并精准提取 ID
    # ==========================================
    for source in data_sources:
        folder_path = source["dir"]
        label = source["label"]
        
        logger.info(f"正在扫描文件夹: {folder_path} (打标签为 {label})")
        
        search_pattern = os.path.join(folder_path, "**", f"*{file_suffix}")
        found_files = glob.glob(search_pattern, recursive=True)
        
        for file_path in found_files:
            filename = os.path.basename(file_path)
            
            # 🔥 核心改动：使用正则表达式精准提取 ADNI ID
            # \d{3} 代表 3个数字，_S_ 是固定字符，\d{4} 代表 4个数字
            match = re.search(subject_id_regex, filename)
            
            if match:
                subject_id = match.group(0) # 成功抓取，例如得到 "002_S_0729"
            else:
                logger.warning(f"无法从 {filename} 中提取标准 Subject_ID，已跳过。")
                continue
            
            records.append({
                "Subject_ID": subject_id,
                "Label": label,
                "File_Path": file_path
            })
            
    # ==========================================
    # 3. 转换为 DataFrame 并保存
    # ==========================================
    df = pd.DataFrame(records)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ 成功生成总表！共找到 {len(df)} 个样本。")
    logger.info(f"其中 sMCI/CN(0): {len(df[df['Label']==0])} 例")  #sMCI CN
    logger.info(f"其中 pMCI/AD(1): {len(df[df['Label']==1])} 例")  #pMCI AD
    logger.info(f"表格已保存至: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_master_csv()