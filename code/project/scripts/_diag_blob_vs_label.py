import json
import random
from pathlib import Path

import cv2
import numpy as np

D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(3)
sel = random.sample(both, 4000)
stats = []
for s in sel:
    p_rgb, p_ir = D / 'labels/train' / (s + '.txt'), D / 'labels_ir/train' / (s + '.txt')
    try:
        a = np.loadtxt(p_rgb, ndmin=2).reshape(-1, 5)
        c = np.loadtxt(p_ir, ndmin=2).reshape(-1, 5)
    except Exception:
        continue
    if a.shape[0] != 1 or c.shape[0] != 1:
        continue
    ir = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        continue
    W0, H0 = 640, 512
    # 在 IR 图上找亮斑质心（全图，取最亮像素周围 60% 阈值质心）
    blur = cv2.GaussianBlur(ir, (5, 5), 0)
    idx = int(np.argmax(blur))
    py, px = divmod(idx, blur.shape[1])
    peak = float(blur[py, px])
    r = 25
    h = 8
    y0, y1 = max(0, py - r), min(H0, py + r + 1)
    x0, x1 = max(0, px - r), min(W0, px + r + 1)
    return_patch = blur[y0:y1, x0:x1]
    m = return_patch >= peak * 0.6
    yy, xx = np.nonzero(m)
    wts = return_patch[yy, xx]
    bx = x0 + float((xx * wts).sum() / wts.sum())
    by = y0 + float((yy * wts).sum() / wts.sum())
    # 标签中心（IR 标签 / RGB 标签），像素坐标
    irx, iry = c[0][1] * W0, c[0][2] * H0
    rgx, rgy = a[0][1] * W0, a[0][2] * H0
    stats.append((bx - irx, by - iry, bx - rgx, by - rgy, peak))
st = np.array(stats)
print('n', len(st))
for i, name in enumerate(['blob-IRlabel dx', 'blob-IRlabel dy', 'blob-RGBlabel dx', 'blob-RGBlabel dy']):
    v = st[:, i]
    print('%-18s mean %7.2f median %7.2f |.| p50 %6.2f p90 %6.2f  frac|.|<=15px %.3f'
          % (name, v.mean(), np.median(v), np.percentile(np.abs(v), 50), np.percentile(np.abs(v), 90),
             float(np.mean(np.abs(v) <= 15))))
