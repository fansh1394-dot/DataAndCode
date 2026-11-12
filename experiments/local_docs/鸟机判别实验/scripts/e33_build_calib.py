# -*- coding: utf-8 -*-
"""E33：构建 int8 量化校准集（6 通道 640x640 uint8）。

采样：AVID 120 + Mendeley 120 + 合成 other 60，去重后随机取；
预处理与训练一致：等比缩放 + 114 填充（letterbox），RGB/IR 用同一几何变换。
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

RAW = Path("/root/rivermind-fs/birddrone")
BD3 = RAW / "bd_nc3"
OUT = RAW / "deploy_nc3" / "calib"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(0)
IMGSZ = 640


def letterbox(img, size=IMGSZ, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out


def main():
    rels = [x.strip() for x in (BD3 / "train_nc3.txt").read_text().splitlines() if x.strip()]
    seen, men, avid, syn = set(), [], [], []
    for r in rels:
        stem = Path(r).stem
        if stem in seen:
            continue
        seen.add(stem)
        (syn if "__syn" in stem else (men if "men__" in stem else avid)).append(r)
    picks = random.sample(avid, min(120, len(avid))) + \
            random.sample(men, min(120, len(men))) + \
            random.sample(syn, min(60, len(syn)))

    n = 0
    for rel in picks:
        p = BD3 / rel
        rgb = cv2.imread(str(p))
        irs = list((BD3 / "images_ir/train").glob(Path(rel).stem + ".*"))
        if rgb is None or not irs:
            continue
        ir = cv2.imread(str(irs[0]))
        if ir is None:
            continue
        a = letterbox(rgb)
        b = letterbox(ir)
        six = np.concatenate([a, b], axis=2)  # HWC 6ch
        np.save(OUT / f"calib_{n:03d}.npy", six)
        n += 1
    print("calib", n, "->", OUT)


if __name__ == "__main__":
    main()
