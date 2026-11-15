#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确认相位相关的符号：用已知真实位移验证 estimate_shift 的方向约定。

AVID 论文称 32.3% 帧全局位移 >5 px。若实测未配准也只有 ~0.5 px，
要么 AVID 的 IR 已近似对齐，要么相位相关在跨模态（RGB vs IR）上失效。
这里用「同模态已知平移」验证符号，再用「RGB vs IR」看跨模态是否给出大位移。
"""
import os
import random

import cv2
import numpy as np

B = "/root/rivermind-fs/birddrone"
AVID = "/root/rivermind-fs/AVID"
RGB3 = os.path.join(B, "bd_nc3", "images", "train")
UN = os.path.join(B, "bd_nc3_unreg", "images_ir", "train")


def gray(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def hann(shape):
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def est(a, b):
    w = hann(a.shape)
    (dx, dy), r = cv2.phaseCorrelate(a * w, b * w)
    return dx, dy, r


# 1) 同模态自检
g0 = gray(os.path.join(UN, sorted(os.listdir(UN))[0]))
for tx, ty in ((6, 0), (0, -5), (8, 7)):
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    g1 = cv2.warpAffine(g0, m, (g0.shape[1], g0.shape[0]), borderMode=cv2.BORDER_REFLECT)
    dx, dy, r = est(g0, g1)
    print(f"同模态 真实平移(dx={tx},dy={ty}) -> 估计(dx={dx:+.2f},dy={dy:+.2f}) resp={r:.3f}")

# 2) 跨模态：RGB vs 未配准 IR
names = sorted(f for f in os.listdir(UN) if f.startswith("avid__") and "__syn" not in f)
random.seed(2)
mags = []
for n in random.sample(names, 60):
    pg, pi = os.path.join(RGB3, n), os.path.join(UN, n)
    if not (os.path.exists(pg) and os.path.exists(pi)):
        continue
    dx, dy, r = est(gray(pg), gray(pi))
    mags.append((np.hypot(dx, dy), dx, dy, r))
arr = np.array(mags)
print(f"\nRGB vs 未配准IR（n={len(arr)}）：位移 中位={np.median(arr[:,0]):.2f} 均值={arr[:,0].mean():.2f} "
      f"p90={np.percentile(arr[:,0],90):.2f} max={arr[:,0].max():.2f} px")
print(f"  |dx|中位={np.median(abs(arr[:,1])):.2f} |dy|中位={np.median(abs(arr[:,2])):.2f} resp均值={arr[:,3].mean():.3f}")
print(f"  >5px 占比={100*(arr[:,0]>5).mean():.0f}%（AVID 论文称 32.3%）")

# 3) 同场景：RGB vs RGB 自身（下界噪声）
print("\n参考：同图自身比较应得 0.00 px")
