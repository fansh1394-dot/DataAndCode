# -*- coding: utf-8 -*-
"""核实 EO/IR 图像是否逐字节相同（排除 shell 引号问题）。"""
import glob
import hashlib
import os
import sys
from collections import Counter

import cv2
import numpy as np

D = "/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/Labelled_UAV_Images/Labelled EO_IR Dataset"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for split in ("test",):
    eo_dir = os.path.join(D, "UAV_eo_Dataset/images", split)
    ir_dir = os.path.join(D, "UAV_ir_Dataset/images", split)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(eo_dir))
    same = diff = 0
    filesize_same = 0
    for s in stems:
        e = os.path.join(eo_dir, s + ".png")
        i = os.path.join(ir_dir, s + ".png")
        if not os.path.exists(i):
            continue
        if os.path.getsize(e) == os.path.getsize(i):
            filesize_same += 1
        if md5(e) == md5(i):
            same += 1
        else:
            diff += 1
    print(f"[{split}] 共 {len(stems)} 对: md5 相同={same} 不同={diff}; 文件大小相同={filesize_same}")

# 像素统计
print("\n=== 抽样像素统计 ===")
for s in stems[:8]:
    e = cv2.imread(os.path.join(eo_dir, s + ".png"), cv2.IMREAD_GRAYSCALE)
    i = cv2.imread(os.path.join(ir_dir, s + ".png"), cv2.IMREAD_GRAYSCALE)
    if e is None or i is None:
        continue
    same_px = np.array_equal(e, i)
    print(f"  {s}: EO {e.shape} mean={e.mean():.1f} | IR {i.shape} mean={i.mean():.1f} "
          f"| 像素完全相同={same_px} | 相关系数={np.corrcoef(e.ravel(), i.ravel())[0,1]:.3f}")

# 标签比对
print("\n=== 标签逐字节比对（全 split）===")
for split in ("train", "val", "test"):
    el = os.path.join(D, "UAV_eo_Dataset/labels", split)
    il = os.path.join(D, "UAV_ir_Dataset/labels", split)
    same = diff = binread = 0
    ex = []
    for f in sorted(os.listdir(el)):
        p1, p2 = os.path.join(el, f), os.path.join(il, f)
        if not os.path.exists(p2):
            continue
        b1, b2 = open(p1, "rb").read(), open(p2, "rb").read()
        try:
            b1.decode("utf-8")
        except UnicodeDecodeError:
            binread += 1
            if len(ex) < 3:
                ex.append(f"  {f}: EO 是非文本二进制 ({len(b1)} bytes) | IR={b2.decode('utf-8','replace').strip()[:60]}")
            continue
        if b1 == b2:
            same += 1
        else:
            diff += 1
            if len(ex) < 3:
                ex.append(f"  {f}: EO={b1.decode().strip()[:50]} | IR={b2.decode().strip()[:50]}")
    print(f"  {split}: 相同={same} 不同={diff} 非文本={binread}")
    for e in ex:
        print(e)
