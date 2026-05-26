# config.py 统一配置文件
import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录（上一级目录）
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(BASE_DIR, "template")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 0.全局任务开关
CURRENT_TASK = "MCI" #可选值：AD_CN或MCI

# 1. generate_csv.py 配置
# 1. 根据任务自动切换数据源、CSV目录和模型命名
# ==========================================
if CURRENT_TASK == "MCI":
    # MCI 任务配置
    DATA_SOURCES = [
        {"dir": os.path.join(DATA_DIR, "preprocessed/sMCI_processed"), "label": 0},
        {"dir": os.path.join(DATA_DIR, "preprocessed/pMCI_processed"), "label": 1},
    ]
    CLINICAL_DIR = os.path.join(DATA_DIR, "clinical/MCI")
    SAVE_MODEL_PATH = "best_model_mci.pth"  # 专属模型名

elif CURRENT_TASK == "AD_CN":
    # AD/CN 任务配置 (CN为0，AD为1)
    DATA_SOURCES = [
        {"dir": os.path.join(DATA_DIR, "preprocessed/CN_processed"), "label": 0},
        {"dir": os.path.join(DATA_DIR, "preprocessed/AD_processed"), "label": 1},
    ]
    CLINICAL_DIR = os.path.join(DATA_DIR, "clinical/AD_CN")
    SAVE_MODEL_PATH = "best_model_ad_cn.pth" # 专属模型名
FILE_SUFFIX = "_final_preprocessed.nii.gz"
SUBJECT_ID_REGEX = r"\d{3}_S_\d{4}"  # 可按需修改

# 2. split_data.py 配置
os.makedirs(CLINICAL_DIR, exist_ok=True) # 自动创建任务对应的文件夹

ALL_DATA_CSV = os.path.join(CLINICAL_DIR, "all_data.csv")
TRAIN_CSV = os.path.join(CLINICAL_DIR, "train.csv")
VAL_CSV = os.path.join(CLINICAL_DIR, "val.csv")
TEST_CSV = os.path.join(CLINICAL_DIR, "test.csv")
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VAL_TEST_RATIO = 0.5  # 剩余20%中，验证集/测试集各占50%

# 3. preprocess_pipeline.py 配置
PREPROCESS_CONFIG = {
    "input_root": os.path.join(DATA_DIR, "raw/AD/nii"),
    "mni_template": os.path.join(TEMPLATE_DIR, "mni_icbm152_t1_tal_nlin_asym_55_ext.nii"),
    "mni_mask": os.path.join(TEMPLATE_DIR, "mni_icbm152_t1_tal_nlin_asym_55_ext_mask.nii"),
    "output_dir": os.path.join(DATA_DIR, "preprocessed/AD_processed"),
    "use_parallel": False,
    "max_workers": None,
    "overwrite": False,
}

# 4. build_guide_map.py 配置
GUIDE_MAP_CONFIG = {
    "base_dir": os.path.join(BASE_DIR, "Julich/BrainMap/probabilistic-maps_PMs_207-areas"),
    "core_areas": [
        "Area-EC", 
        "Hippocampus-CA1", "Hippocampus-CA2", "Hippocampus-CA3", "Hippocampus-DG", "Hippocampus-Subc",
        "Amygdala-LB", "Amygdala-CM", "Amygdala-SF"
    ],
    "ref_mri": os.path.join(DATA_DIR, "preprocessed/pMCI_processed/002_S_0729/002_S_0729_MPRAGE_SENSE_final_preprocessed.nii.gz"),
    "output_file": os.path.join(DATA_DIR, "anatomy_prior/braak_guide_map.npy"),
}

# 5.训练配置
BATCH_SIZE = 2               # 批次大小（显存小就设2/4）
NUM_WORKERS = 4              # 数据加载线程（Windows设0，Linux设4）
TRAIN_IMG_SIZE = (96, 96, 96)# 输入MRI尺寸
# SAVE_MODEL_PATH = "best_model.pth" # 最佳模型保存路径

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(LOG_DIR, "run.log"),
}