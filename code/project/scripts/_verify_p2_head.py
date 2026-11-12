# -*- coding: utf-8 -*-
"""P2 头配置的 stride 回填校验。

为什么单独校验
--------------
tasks.py 在多头场景下用「实际前向特征图尺寸」反推每个分支头的 stride。
若某个头的 P2 特征没有被正确收集（例如 Concat 尺寸不匹配被静默 reshape），
stride 会退回兜底值 [8,16,32,64]，构建不报错但 TAL 正样本分配全错，
表现为 loss 正常、mAP 极低。这里直接打印，避免白跑 150 轮。
"""
import sys
from pathlib import Path

import yaml

ROOT = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

CFG = ROOT / "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat-p2.yaml"


if __name__ == "__main__":
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    m = DetectionModel(cfg, ch=6, nc=1, verbose=False)
    agg = m.model[-1]
    print(f"NL={agg.nl}")
    print(f"MODEL_STRIDE={m.stride.tolist()}")
    for name, head in agg._branch_modules.items():
        print(f"HEAD {name}: nl={head.nl} stride={head.stride.tolist()}")
    print(f"PARAMS_M={sum(p.numel() for p in m.parameters()) / 1e6:.4f}")
