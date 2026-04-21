# 文件位置：models/__init__.py

# 从各个子文件中导入我们写好的类
from .anatomy_guide import AnatomyPriorGuideGeneration
from .alignment import AnatomyGuidedAlignment
from .apm_former import APM_Former_ImageOnly
# 下面这两个是我们接下来要写的：
# from .fusion import AnatomyConstrainedFusion
# from .apm_former import APM_Former

# __all__ 规定了当使用 `from models import *` 时，哪些类会被暴露出去
__all__ = [
    "AnatomyPriorGuideGeneration",
    "AnatomyGuidedAlignment",
    "APM_Former_ImageOnly",
    # "AnatomyConstrainedFusion",
    # "APM_Former"
]