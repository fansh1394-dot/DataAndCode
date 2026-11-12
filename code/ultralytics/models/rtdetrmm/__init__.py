# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""RTDETRMM 独立模型家族（不依附 RTDETR）。

说明：
- 该包提供 RTDETRMM 的 model/train/val/predict/visualize 等组件。
- 设计目标：RTDETRMM 不继承 RTDETR，不依赖 ultralytics.models.rtdetr.*。
"""

from .model import RTDETRMM
from .train_obb import RTDETRMMOBBTrainer
from .val_obb import RTDETRMMOBBValidator
from .mm_predictor_obb import RTDETRMMOBBPredictor

__all__ = ["RTDETRMM", "RTDETRMMOBBTrainer", "RTDETRMMOBBValidator", "RTDETRMMOBBPredictor"]
