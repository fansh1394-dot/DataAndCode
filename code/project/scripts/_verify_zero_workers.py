# -*- coding: utf-8 -*-
"""对比：置零 patch 在 workers=0 与 workers=4 下是否生效。

B 阶段实测三种置零的 fuse 指标完全相同（Δ=0），说明 patch 没生效。
本脚本直接在 batch 层面看通道值，定位是「patch 没被调用」还是
「patch 生效但被后续层覆盖」。
"""
import os
import sys

import torch
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

CALLS = {"n": 0}
orig = ds.__class__.__getitem__


def _wrapped(self, i):
    CALLS["n"] += 1
    labels = orig(self, i)
    img = labels.get("img")
    if img is not None and img.shape[0] >= 6:
        img[3:6] = 0
    return labels


ds.__class__.__getitem__ = _wrapped

for nw in (0, 4):
    CALLS["n"] = 0
    loader = build_dataloader(ds, batch=4, workers=nw, shuffle=False, rank=-1)
    b = next(iter(loader))
    x = b["img"].float()
    print(f"[workers={nw}] 调用次数={CALLS['n']}  "
          f"后3通道均值={x[:, 3:].mean():.4f}  "
          f"后3通道最大值={x[:, 3:].max():.4f}", flush=True)
    del loader

print("\n判读：")
print("  workers=0 后3通道=0 且调用次数>0 -> patch 本身正确")
print("  workers=4 后3通道≠0             -> 子进程未继承 patch（需改为 dataset 内部注入）")
