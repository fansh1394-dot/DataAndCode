#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：配准后图像是否发生亮度/对比度变化（JPEG q95 重编码 + warp 插值），
这可能是 A0 的 IR 分支反而更差的原因。"""
import os
import random

import numpy as np
from PIL import Image

B = "/root/rivermind-fs/birddrone"
REG3 = os.path.join(B, "bd_nc3", "images_ir", "train")
UN = os.path.join(B, "bd_nc3_unreg", "images_ir", "train")
REGFLAT = os.path.join(B, "avid_ir_reg", "train")


def L(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)


names = sorted(f for f in os.listdir(UN) if f.startswith("avid__") and "__syn" not in f
               and os.path.exists(os.path.join(REG3, f)))
random.seed(3)
rows = []
for n in random.sample(names, 60):
    r = L(os.path.join(REG3, n))
    u = L(os.path.join(UN, n))
    if r.shape != u.shape:
        continue
    rows.append((r.mean(), u.mean(), r.std(), u.std(), r.min(), u.min(), r.max(), u.max()))

arr = np.array(rows)
print(f"n={len(arr)}")
print(f"  均值亮度: reg={arr[:,0].mean():.2f}  unreg={arr[:,1].mean():.2f}  差={arr[:,0].mean()-arr[:,1].mean():+.2f}")
print(f"  标准差  : reg={arr[:,2].mean():.2f}  unreg={arr[:,3].mean():.2f}  差={arr[:,2].mean()-arr[:,3].mean():+.2f}")
print(f"  最小灰度: reg={arr[:,4].mean():.2f}  unreg={arr[:,5].mean():.2f}")
print(f"  最大灰度: reg={arr[:,6].mean():.2f}  unreg={arr[:,7].mean():.2f}")

# JPEG 双次压缩的块效应：高频能量对比
def hf_energy(p):
    a = L(p)
    gy, gx = np.gradient(a)
    return float(np.hypot(gx, gy).mean())

e_reg = np.mean([hf_energy(os.path.join(REG3, n)) for n in random.sample(names, 20)])
e_un = np.mean([hf_energy(os.path.join(UN, n)) for n in random.sample(names, 20)])
print(f"\n  梯度能量（高频/纹理）: reg={e_reg:.3f} unreg={e_un:.3f} 相对损失={100*(1-e_reg/e_un):.1f}%")
