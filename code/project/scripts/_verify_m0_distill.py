# -*- coding: utf-8 -*-
"""构建校验：确认 M0(concat) 三头配置可构建、可前向、且融合节点能被 RGB 屏蔽定位。"""
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

import torch

from ultralytics.nn.tasks import DetectionModel, yaml_model_load

CFG = "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml"

m = DetectionModel(yaml_model_load(CFG), ch=3, nc=1, verbose=False)
head = m.model[-1]
print(f"layers={len(m.model)} branch_order={head.branch_order} fuse_ir={head.fuse_ir_detect is not None}")
print(f"params={sum(p.numel() for p in m.parameters()) / 1e6:.4f} M")

# 融合节点定位（_mm_fusion_concat 由 parse_model 打标）
idx = [i for i, layer in enumerate(m.model) if getattr(layer, "_mm_fusion_concat", False)]
print(f"fusion_concat_idx={idx}")

m.eval()
with torch.no_grad():
    out = m(torch.randn(2, 6, 256, 256))
print(f"forward OK keys={list(out.keys()) if isinstance(out, dict) else type(out)}")
