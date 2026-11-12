# -*- coding: utf-8 -*-
"""定位：置零后仍无变化，究竟是 patch 未命中还是数据被缓存。

逐步打印 dataset 的类型、MRO、以及 patch 是否被杀。
"""
import os
import sys

import numpy as np
import yaml

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator  # noqa: E402

DERIVED = "_d2p1/eval_yamls/data_test_dual.yaml"
raw = yaml.safe_load(open(DERIVED, encoding="utf-8"))
data = check_det_dataset(DERIVED)
for k, v in raw.items():
    data.setdefault(k, v)

vargs = dict(task="detect", mode="val", data=DERIVED, split="val", imgsz=640,
             batch=4, workers=0, device="0", conf=0.001, iou=0.7, max_det=300,
             half=False, plots=False, save_json=False, save_txt=False,
             save_conf=False, verbose=False)

builder = MultiModalDetectionValidator(args=vargs)
builder.data = data
builder.stride = 32
ds = builder.build_dataset(data.get("val"), mode="val", batch=4)

print("dataset 类型:", type(ds).__name__)
print("MRO:", [c.__name__ for c in type(ds).__mro__][:6])
print("有 _load_multimodal_components:", hasattr(ds, "_load_multimodal_components"))
print("cache_ram:", getattr(ds, "cache_ram", None))
print("cache:", getattr(ds, "cache", None))

# 原始路径拿到的 6 通道图
lab = ds[0]
img = lab["img"]
print("\n[原样] img.type=", type(img).__name__, "shape=", tuple(img.shape))
arr = np.asarray(img)
print("  前3均值=%.2f 后3均值=%.2f" % (arr[:3].mean(), arr[3:].mean()))

# 直接调 _load_multimodal_components，看它是否就是拼接源头
rgb, x = ds._load_multimodal_components(0)
print("\n_load_multimodal_components(0):")
print("  rgb shape=", tuple(rgb.shape), "mean=%.2f" % rgb.mean())
print("  x   shape=", tuple(x.shape), "mean=%.2f" % x.mean())

# 检查 6 通道图是否等于两者直接拼接
cat = np.concatenate([rgb, x], axis=2)
print("  concat 与原样 img 形状一致:", cat.shape == arr.shape if hasattr(arr, "shape") else "N/A")
