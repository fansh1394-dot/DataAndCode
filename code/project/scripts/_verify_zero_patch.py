# -*- coding: utf-8 -*-
"""校验 --zero-modality 是否真正生效。

关键风险：dataloader 用 workers>0 时，数据在子进程里读取，主进程对 dataset 类打的
patch 不会被继承（fork/spawn 时机在 build_dataloader 之后）。本脚本直接比对
「原样」与「置零」两条路径产出的 batch，确认置零后的通道确实为常数。
"""
import os
import sys

import torch

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
import yaml  # noqa: E402

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

print(f"数据集大小={len(ds)}  类型={type(ds).__name__}", flush=True)


def grab(ds, n=2):
    """直接从 dataset 取 n 个样本，看 6 通道布局。"""
    out = []
    for i in range(n):
        out.append(ds[i]["img"].float())
    return out


# 原样
imgs = grab(ds)
for k, t in enumerate(imgs):
    print(f"[原样] #{k} shape={tuple(t.shape)}  "
          f"前3通道均值={t[:3].mean():.4f}  后3通道均值={t[3:].mean():.4f}", flush=True)

# 主进程内 patch（与 eval_split 逻辑一致）
ch_half = 3
idx0, idx1 = 3, 6  # 置零 IR
orig = ds.__class__.__getitem__


def _wrapped(self, i):
    labels = orig(self, i)
    img = labels.get("img")
    if img is not None and img.shape[0] >= idx1:
        img[idx0:idx1] = 128
    return labels


ds.__class__.__getitem__ = _wrapped
imgs2 = grab(ds)
for k, t in enumerate(imgs2):
    print(f"[IR置中灰] #{k} shape={tuple(t.shape)}  "
          f"前3通道均值={t[:3].mean():.4f}  后3通道均值={t[3:].mean():.4f}")

ds.__class__.__getitem__ = orig

# 关键验证：workers=4 时子进程是否继承 patch
ds.__class__.__getitem__ = _wrapped
loader = build_dataloader(ds, batch=4, workers=2, shuffle=False, rank=-1)
b = next(iter(loader))
x = b["img"].float()
print(f"\n[workers=2 子进程] batch img shape={tuple(x.shape)}  "
      f"前3通道均值={x[:, :3].mean():.4f}  后3通道均值={x[:, 3:].mean():.4f}")
print("判定：后3通道若接近 128 -> 子进程继承了 patch；若接近真实图像均值(~135~150) -> patch 在子进程失效")
