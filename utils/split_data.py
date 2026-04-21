# utils/split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from config import (
    ALL_DATA_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV,
    SPLIT_SEED, TRAIN_RATIO, VAL_TEST_RATIO
)
from logging_utils import get_logger

logger = get_logger(__name__)

def split_dataset(
    input_csv: str = ALL_DATA_CSV,
    train_output: str = TRAIN_CSV,
    val_output: str = VAL_CSV,
    test_output: str = TEST_CSV,
    seed: int = SPLIT_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_test_ratio: float = VAL_TEST_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分层划分数据集（保证各集合Label比例一致）
    
    Args:
        input_csv: 总样本CSV路径
        train_output: 训练集输出路径
        val_output: 验证集输出路径
        test_output: 测试集输出路径
        seed: 随机种子（保证可复现）
        train_ratio: 训练集占比
        val_test_ratio: 剩余数据中验证集占比（测试集=1-val_test_ratio）
    
    Returns:
        train_df, val_df, test_df: 划分后的数据集
    
    Raises:
        FileNotFoundError: 输入CSV不存在
        ValueError: Label列缺失/样本数为0/划分后集合为空
    """
    # 1. 前置校验
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"输入文件不存在: {input_csv}，请先运行 generate_csv.py")
    
    # 2. 读取数据并校验
    df = pd.read_csv(input_csv)
    if len(df) == 0:
        raise ValueError("输入CSV无样本数据！")
    if "Label" not in df.columns:
        raise ValueError("输入CSV缺少 'Label' 列！")
    
    # 3. 分层划分
    train_df, temp_df = train_test_split(
        df, test_size=1-train_ratio, random_state=seed, stratify=df["Label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1-val_test_ratio, random_state=seed, stratify=temp_df["Label"]
    )
    
    # 4. 校验划分结果
    for name, data in zip(["训练集", "验证集", "测试集"], [train_df, val_df, test_df]):
        if len(data) == 0:
            raise ValueError(f"{name}样本数为0，请检查划分比例或原始数据！")
    
    # 5. 保存文件
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    # 6. 日志输出
    logger.info(f"✅ 数据集划分完成！")
    logger.info(f"训练集: {len(train_df)} 例（{len(train_df)/len(df)*100:.1f}%）")
    logger.info(f"验证集: {len(val_df)} 例（{len(val_df)/len(df)*100:.1f}%）")
    logger.info(f"测试集: {len(test_df)} 例（{len(test_df)/len(df)*100:.1f}%）")
    logger.info(f"训练集Label分布: {train_df['Label'].value_counts().to_dict()}")
    logger.info(f"验证集Label分布: {val_df['Label'].value_counts().to_dict()}")
    logger.info(f"测试集Label分布: {test_df['Label'].value_counts().to_dict()}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    try:
        split_dataset()
    except Exception as e:
        logger.error(f"数据集划分失败: {str(e)}", exc_info=True)
        raise  # 抛出异常，让脚本退出