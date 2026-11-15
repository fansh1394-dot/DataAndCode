#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证配准方向和幅度：对配准后 IR 与 RGB 再做相位相关，看残余位移是否被消掉。"""
import os
import random

import cv2
import numpy as np

B = "/root/rivermind-fs/birddrone"
AVID = "/root/rivermind-fs/AVID"
REG3 = os.path.join(B, "bd_nc3", "images_ir", "train")
RGB3 = os.path.join(B, "bd_nc3", "images", "train")
UN = os.path.join(B, "bd_nc3_unreg", "images_ir", "train")


def gray(p):
    return (cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)


def hann(shape):
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def shift(a, b):
    w = hann(a.shape)
    (dx, dy), resp = cv2.phaseCorrelate(a * w, b * w)
    return dx, dy, np.hypot(dx, dy), resp


names = sorted(f for f in os.listdir(UN) if f.startswith("avid__") and "__syn" not in f)
random.seed(1)
res = {"un": [], "reg": []}
for n in random.sample(names, 40):
    pr, pu, pg = os.path.join(REG3, n), os.path.join(UN, n), os.path.join(RGB3, n)
    if not (os.path.exists(pr) and os.path.exists(pu) and os.path.exists(pg)):
        continue
    g = gray(pg)
    for key, p in (("un", pu), ("reg", pr)):
        x = gray(p)
        if x.shape != g.shape:
            continue
        dx, dy, mag, resp = shift(g, x)
        res[key].append((mag, abs(dx), abs(dy), resp))

for key in ("un", "reg"):
    arr = np.array(res[key])
    if not len(arr):
        continue
    print(f"{key:>3}: n={len(arr)} 残余位移 中位={np.median(arr[:,0]):.2f}px 均值={arr[:,0].mean():.2f}px "
          f"p90={np.percentile(arr[:,0],90):.2f}px | |dx|中位={np.median(arr[:,1]):.2f} "
          f"|dy|中位={np.median(arr[:,2]):.2f} | resp均值={arr[:,3].mean():.3f}")
if res["un"] and res["reg"]:
    a = np.array(res["un"])[:, 0]; b = np.array(res["reg"])[:, 0]
    print(f"\n残余位移下降：{a.mean():.2f} -> {b.mean():.2f} px "
          f"（{(1-b.mean()/a.mean())*100:.0f}% 被消除）")
