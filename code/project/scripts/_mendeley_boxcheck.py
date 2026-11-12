# -*- coding: utf-8 -*-
"""验证：同一份标签套到 EO 和 IR 两个模态上，框到底准不准。

做法（不用热斑假设，因为前面已证 IR 最亮点往往不是无人机）
--------------------------------------------------------
1. 把每帧的框内区域裁出来，与**框外邻域**比较梯度能量：
   若框真的框住目标，框内应有显著高于周边的边缘/纹理能量。
2. 用 **IR 标签位置 vs IR 图像最亮点** 的距离分布，判断框与热斑是否系统性错位。
3. 用 **前后帧框位置** 判断标签是否跟随目标运动（若标签是逐帧手标，框中心应平滑移动）。
4. 直接裁图存成 PNG 拼版，供人眼判读（最重要的一步）。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

D = ("/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/"
     "Labelled_UAV_Images/Labelled EO_IR Dataset")
OUT = "/root/rivermind-fs/mendeley_eo_ir/_crops"
SPLIT = "test"


def read_boxes(p):
    out = []
    if not os.path.isfile(p):
        return out
    try:
        txt = open(p, encoding="utf-8").read()
    except UnicodeDecodeError:
        return out
    for line in txt.splitlines():
        s = line.split()
        if len(s) >= 5:
            try:
                out.append(tuple(float(x) for x in s[1:5]))
            except ValueError:
                pass
    return out


def to_px(b, w, h):
    cx, cy, bw, bh = b
    return int((cx - bw / 2) * w), int((cy - bh / 2) * h), int(bw * w), int(bh * h)


def edge_energy(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


eo_dir = os.path.join(D, "UAV_eo_Dataset/images", SPLIT)
ir_dir = os.path.join(D, "UAV_ir_Dataset/images", SPLIT)
eo_lab = os.path.join(D, "UAV_eo_Dataset/labels", SPLIT)
os.makedirs(OUT, exist_ok=True)

stems = sorted(os.path.splitext(f)[0] for f in os.listdir(eo_dir))
report = {"eo": [], "ir": []}
peaks = []
crops = []

for s in stems[:200]:
    be = read_boxes(os.path.join(eo_lab, s + ".txt"))
    if not be:
        continue
    e = cv2.imread(os.path.join(eo_dir, s + ".png"), cv2.IMREAD_GRAYSCALE)
    i = cv2.imread(os.path.join(ir_dir, s + ".png"), cv2.IMREAD_GRAYSCALE)
    if e is None or i is None or e.shape != i.shape:
        continue
    h, w = e.shape[:2]
    x, y, bw, bh = to_px(be[0], w, h)
    if bw < 4 or bh < 4:
        continue
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        continue

    for name, img in (("eo", e), ("ir", i)):
        g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
        inside = edge_energy(g[y0:y1, x0:x1])
        # 框外同尺寸邻域（上下左右各取一块，取最大）
        outs = []
        for yy in (y0 - bh - 5, y1 + 5):
            if 0 <= yy and yy + bh <= h:
                outs.append(edge_energy(g[yy:yy + bh, x0:x1]))
        for xx in (x0 - bw - 5, x1 + 5):
            if 0 <= xx and xx + bw <= w:
                outs.append(edge_energy(g[y0:y1, xx:xx + bw]))
        if outs:
            report[name].append(inside / (max(outs) + 1e-6))

    # IR 最亮点与框中心距离
    gg = cv2.GaussianBlur(i.astype(np.float32), (0, 0), 4.0)
    cy_, cx_ = np.unravel_index(int(np.argmax(gg)), gg.shape)
    dist = np.hypot((cx_ - (x0 + (x1 - x0) / 2)) / w, (cy_ - (y0 + (y1 - y0) / 2)) / h)
    peaks.append(dist)

    if len(crops) < 12:
        pad = 30
        X0, Y0 = max(0, x - pad), max(0, y - pad)
        X1, Y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        ce = cv2.cvtColor(e[Y0:Y1, X0:X1], cv2.COLOR_GRAY2BGR)
        ci = cv2.cvtColor(i[Y0:Y1, X0:X1], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(ce, (x - X0, y - Y0), (x - X0 + bw, y - Y0 + bh), (0, 255, 0), 1)
        cv2.rectangle(ci, (x - X0, y - Y0), (x - X0 + bw, y - Y0 + bh), (0, 255, 0), 1)
        ce = cv2.resize(ce, (240, 200))
        ci = cv2.resize(ci, (240, 200))
        cv2.putText(ce, f"EO {s}", (4, 14), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(ci, f"IR {s}", (4, 14), 0, 0.4, (0, 255, 255), 1)
        crops.append(np.hstack([ce, ci]))

print("=== 框内外边缘能量比（>1 表示框内有更多结构）===")
for k, v in report.items():
    if v:
        a = np.array(v)
        print(f"  {k.upper()}: n={len(a)} 中位数={np.median(a):.3f} "
              f"均值={a.mean():.3f} 比值>1 占比={np.mean(a>1)*100:.1f}% "
              f"比值>2 占比={np.mean(a>2)*100:.1f}%")

print("\n=== IR 最亮点与框中心距离（归一化）===")
if peaks:
    a = np.array(peaks)
    print(f"  n={len(a)} 中位数={np.median(a):.4f} 均值={a.mean():.4f} "
          f"p90={np.percentile(a,90):.4f}")
    print(f"  距离<0.02（约框内）占比={np.mean(a<0.02)*100:.1f}%")

if crops:
    rows = []
    for k in range(0, len(crops), 3):
        chunk = crops[k:k + 3]
        while len(chunk) < 3:
            chunk.append(np.zeros_like(crops[0]))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    p = os.path.join(OUT, "montage.png")
    cv2.imwrite(p, sheet)
    print(f"\n拼版图: {p}  尺寸 {sheet.shape}")
