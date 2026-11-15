#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确认 bd_nc3_unreg 是否真的用了未配准 IR；统计匹配率与几何位移分布。"""
import os
import random
import json

import numpy as np
from PIL import Image

B = "/root/rivermind-fs/birddrone"
AVID = "/root/rivermind-fs/AVID"
UNREG = os.path.join(B, "bd_nc3_unreg", "images_ir", "train")
REG3 = os.path.join(B, "bd_nc3", "images_ir", "train")

names = sorted(f for f in os.listdir(UNREG) if f.startswith("avid__") and f.endswith(".jpg"))
print("bd_nc3_unreg 中 AVID 基础帧数:", len(names))
random.seed(0)
sample = random.sample(names, min(60, len(names)))

def find_avid(p):
    stem = p[:-4]
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    video, frame = parts[1], parts[2]
    for split in ("train", "val"):
        c = os.path.join(AVID, "images", "ir", split, video, f"{frame}.jpg")
        if os.path.exists(c):
            return c
    return None

ok = 0
diffs = []
for n in sample:
    avid_p = find_avid(n)
    if not avid_p:
        continue
    ok += 1
    a = np.asarray(Image.open(avid_p).convert("L"), dtype=np.float32)
    u = np.asarray(Image.open(os.path.join(UNREG, n)).convert("L"), dtype=np.float32)
    if a.shape != u.shape:
        print("  尺寸差异", n, a.shape, u.shape); continue
    diffs.append(np.abs(a - u).mean())
    if len(diffs) <= 5:
        print(f"  {n}: AVID原图 vs unreg 均值差={diffs[-1]:.3f}")

print(f"\n抽样 {len(sample)} 帧，找到 AVID 原图 {ok} 帧")
if diffs:
    print(f"AVID原图 vs unreg 平均绝对差：中位={np.median(diffs):.4f} 均值={np.mean(diffs):.4f} 最大={max(diffs):.4f}")
    print("=> 差值≈0 表示 unreg 确实使用 AVID 原始未配准 IR" if np.median(diffs) < 0.5 else "=> 差值不为 0，需进一步核查")

st = os.path.join(B, "avid_ir_reg", "reg_stats.json")
if os.path.exists(st):
    s = json.load(open(st))
    print("\n配准统计键:", list(s)[:10])
    for k in ("n", "fail", "fail_rate", "median_shift", "mean_shift"):
        if k in s:
            print(f"  {k}: {s[k]}")
