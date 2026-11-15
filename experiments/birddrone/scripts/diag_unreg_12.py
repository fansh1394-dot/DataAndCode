#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拆分 B1 差异来源：per-class 对比 + 分源标签统计（AVID vs Mendeley vs 合成）。"""
import csv
import json
import os
from collections import Counter

B = "/root/rivermind-fs/birddrone"


def stats(root, split):
    lines = (open(os.path.join(root, f"{split}_nc3.txt"), encoding="utf-8").read().split())
    src = Counter()
    n_box = Counter()
    n_frame = Counter()
    for ln in lines:
        name = os.path.basename(ln)
        if name.startswith("avid__"):
            k = "avid" if "__syn" not in name else "syn"
        elif name.startswith("men__"):
            k = "men"
        else:
            k = "other"
        src[k] += 1
        lp = os.path.join(root, "labels_ir", split, name[:-4] + ".txt")
        if os.path.exists(lp):
            for row in open(lp):
                row = row.strip()
                if row:
                    n_box[(k, row.split()[0])] += 1
                    n_frame[k] += 1
    return src, n_box, n_frame


for split in ("train", "val"):
    print(f"===== {split} =====")
    for root in ("bd_nc3", "bd_nc3_unreg"):
        src, n_box, _ = stats(os.path.join(B, root), split)
        print(f"-- {root}: 帧 {dict(src)}")
        for k in sorted(src):
            cls = {c: n_box[(k, c)] for c in ("0", "1", "2") if n_box[(k, c)]}
            print(f"     {k}: 框 {cls}")
