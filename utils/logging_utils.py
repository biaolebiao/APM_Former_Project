# utils/logging_utils.py
import logging
from utils.config import LOG_CONFIG

def get_logger(name):
    """获取统一配置的logger"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_CONFIG["level"])
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 文件handler（按时间戳滚动，可选）
    file_handler = logging.FileHandler(LOG_CONFIG["file"], encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger