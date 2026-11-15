#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 bd_nc3（配准）与 bd_nc3_unreg 的 AVID IR 真实差异。"""
import os
import random

import numpy as np
from PIL import Image

B = "/root/rivermind-fs/birddrone"
REG = os.path.join(B, "avid_ir_reg", "train")
UNREG = os.path.join(B, "bd_nc3_unreg", "images_ir", "train")
REG3 = os.path.join(B, "bd_nc3", "images_ir", "train")

files = sorted(f for f in os.listdir(REG) if f.endswith(".jpg"))
print("avid_ir_reg/train 帧数:", len(files))
random.seed(0)
absd, chg = [], []
for f in random.sample(files, 20):
    name = "avid__" + f
    un = os.path.join(UNREG, name)
    r3 = os.path.join(REG3, name)
    if not (os.path.exists(un) and os.path.exists(r3)):
        print("  缺失:", name, os.path.exists(un), os.path.exists(r3)); continue
    a = np.asarray(Image.open(os.path.join(REG, f)).convert("L"), dtype=np.float32)
    b = np.asarray(Image.open(un).convert("L"), dtype=np.float32)
    c = np.asarray(Image.open(r3).convert("L"), dtype=np.float32)
    if a.shape != b.shape:
        print("  尺寸不同", f, a.shape, b.shape); continue
    d = np.abs(a - b)
    absd.append(d.mean()); chg.append(100 * (d > 2).mean())
    print(f"  {f}: 配准vs未配准 均值差={d.mean():.2f} 最大={d.max():.0f} 变化像素={100*(d>2).mean():.1f}%"
          f" | bd_nc3与avid_ir_reg一致={np.array_equal(a, c)}")
print(f"\n样本 {len(absd)} 帧：平均绝对差 中位={np.median(absd):.2f} 均值={np.mean(absd):.2f}；"
      f"变化像素比例 中位={np.median(chg):.1f}%")
