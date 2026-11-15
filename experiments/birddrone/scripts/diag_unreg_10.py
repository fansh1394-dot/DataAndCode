#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排查：未配准 IR 的残余位移为何这么小。检查原生尺寸、以及 AVID 是否自带对齐版本。"""
import os
import random

import cv2
import numpy as np

B = "/root/rivermind-fs/birddrone"
AVID = "/root/rivermind-fs/AVID"
UN = os.path.join(B, "bd_nc3_unreg", "images_ir", "train")
RGB3 = os.path.join(B, "bd_nc3", "images", "train")

names = sorted(f for f in os.listdir(UN) if f.startswith("avid__") and "__syn" not in f)[:5]
for n in names:
    stem = n[:-4]
    parts = stem.split("__")
    video, frame = parts[1], parts[2]
    rgb_p = os.path.join(RGB3, n)
    un_p = os.path.join(UN, n)
    rgb_raw = os.path.join(AVID, "images", "rgb", "train", video, f"{frame}.jpg")
    ir_raw = os.path.join(AVID, "images", "ir", "train", video, f"{frame}.jpg")
    for tag, p in (("bd_nc3 RGB", rgb_p), ("unreg IR", un_p),
                   ("AVID RGB 原生", rgb_raw), ("AVID IR 原生", ir_raw)):
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            print(f"{n[:34]:36s} {tag:14s} shape={im.shape}")
    print()

print("=== AVID 目录结构 ===")
for d in sorted(os.listdir(AVID)):
    print(" ", d)
print("\n=== images 下 ===")
ip = os.path.join(AVID, "images")
for d in sorted(os.listdir(ip)):
    print(" ", d)
