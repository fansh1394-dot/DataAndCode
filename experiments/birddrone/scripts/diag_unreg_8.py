#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化 A0 与 B1 的 IR 输入统计差异（训练集抽样）：配准 vs 未配准。"""
import os
import random

import numpy as np
from PIL import Image

B = "/root/rivermind-fs/birddrone"
REG3 = os.path.join(B, "bd_nc3", "images_ir", "train")
UN = os.path.join(B, "bd_nc3_unreg", "images_ir", "train")
RGB = os.path.join(B, "bd_nc3", "images", "train")

names = sorted(f for f in os.listdir(UN) if f.startswith("avid__") and "__syn" not in f)
random.seed(0)
sample = random.sample(names, 40)


def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / d) if d > 0 else float("nan")


rows = []
for n in sample:
    p_reg = os.path.join(REG3, n)
    p_un = os.path.join(UN, n)
    p_rgb = os.path.join(RGB, n)
    if not (os.path.exists(p_reg) and os.path.exists(p_un)):
        continue
    r, u = load(p_reg), load(p_un)
    if r.shape != u.shape:
        continue
    row = {"n": n, "mad": float(np.abs(r - u).mean()), "ncc_reg_un": ncc(r, u),
           "std_reg": float(r.std()), "std_un": float(u.std())}
    if os.path.exists(p_rgb):
        g = load(p_rgb)
        if g.shape == r.shape:
            row["ncc_rgb_reg"] = ncc(g, r)
            row["ncc_rgb_un"] = ncc(g, u)
    rows.append(row)

print(f"有效样本 {len(rows)}")
for k in ("mad", "ncc_reg_un", "std_reg", "std_un", "ncc_rgb_reg", "ncc_rgb_un"):
    vals = [r[k] for r in rows if k in r and not np.isnan(r[k])]
    if vals:
        print(f"{k}: 中位={np.median(vals):.4f} 均值={np.mean(vals):.4f} 最小={min(vals):.4f} 最大={max(vals):.4f}")

d_ncc = [r["ncc_rgb_reg"] - r["ncc_rgb_un"] for r in rows if "ncc_rgb_reg" in r and "ncc_rgb_un" in r]
if d_ncc:
    print(f"\nRGB 与 IR 的 NCC 提升（配准-未配准）：中位={np.median(d_ncc):+.4f} 均值={np.mean(d_ncc):+.4f}"
          f"  配准更好的比例={100*sum(1 for d in d_ncc if d>0)/len(d_ncc):.0f}%")
