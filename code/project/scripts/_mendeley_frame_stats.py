# -*- coding: utf-8 -*-
"""统计 Mendeley 数据集中可用于误报评测的帧类型：
   - 有 UAV 框的帧（正样本）
   - 空标签帧（纯背景，测 false positive）
   同时统计目标尺寸，便于分桶。
"""
import os
from collections import Counter

import numpy as np

D = ("/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/"
     "Labelled_UAV_Images/Labelled EO_IR Dataset")

for split in ("train", "val", "test"):
    lab = os.path.join(D, "UAV_eo_Dataset/labels", split)
    imgs = os.path.join(D, "UAV_eo_Dataset/images", split)
    if not os.path.isdir(lab):
        continue
    n_tot = n_empty = n_box = 0
    nbox_hist = Counter()
    sides = []
    bad = 0
    for f in sorted(os.listdir(lab)):
        if not f.endswith(".txt"):
            continue
        stem = f[:-4]
        if not os.path.exists(os.path.join(imgs, stem + ".png")):
            continue
        n_tot += 1
        try:
            txt = open(os.path.join(lab, f), encoding="utf-8").read()
        except UnicodeDecodeError:
            bad += 1
            continue
        lines = [l for l in txt.splitlines() if len(l.split()) >= 5]
        if not lines:
            n_empty += 1
            continue
        n_box += 1
        nbox_hist[len(lines)] += 1
        for l in lines:
            s = l.split()
            try:
                bw, bh = float(s[3]), float(s[4])
            except ValueError:
                continue
            # 归一化 → 假设 640x512 参考
            sides.append(np.sqrt(bw * 640 * bh * 512))
    if sides:
        a = np.array(sides)
        print(f"[{split}] 帧={n_tot} 有框={n_box} 空帧={n_empty} 损坏={bad}")
        print(f"    每帧目标数分布: {dict(sorted(nbox_hist.items()))}")
        print(f"    目标等效边长(px@640x512): 中位={np.median(a):.1f} "
              f"p10={np.percentile(a,10):.1f} p90={np.percentile(a,90):.1f}")
        print(f"    <16px {np.mean(a<16)*100:.1f}% | 16-32px {np.mean((a>=16)&(a<32))*100:.1f}% "
              f"| 32-64px {np.mean((a>=32)&(a<64))*100:.1f}% | >=64px {np.mean(a>=64)*100:.1f}%")
    else:
        print(f"[{split}] 帧={n_tot} 有框={n_box} 空帧={n_empty} 损坏={bad}")
