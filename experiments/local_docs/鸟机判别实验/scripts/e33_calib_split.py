# -*- coding: utf-8 -*-
"""把 6 通道校准 npy 拆成 RGB/IR 两张 jpg，并生成 RKNN 双输入 dataset.txt。"""
from pathlib import Path

import cv2
import numpy as np

C = Path("/root/rivermind-fs/birddrone/deploy_nc3/calib")
OUT = Path("/root/rivermind-fs/birddrone/deploy_nc3/calib_img")
OUT.mkdir(parents=True, exist_ok=True)
lines = []
for p in sorted(C.glob("calib_*.npy")):
    six = np.load(p)
    rgb, ir = six[:, :, :3], six[:, :, 3:]
    r = OUT / f"{p.stem}_rgb.jpg"
    i = OUT / f"{p.stem}_ir.jpg"
    cv2.imwrite(str(r), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(i), ir, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines.append(f"{r} {i}")
(OUT.parent / "dataset.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("pairs", len(lines))
