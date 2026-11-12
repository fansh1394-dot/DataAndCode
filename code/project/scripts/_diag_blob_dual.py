import json
import random
from pathlib import Path

import cv2
import numpy as np

D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(3)
sel = random.sample(both, 1500)


def blob(ir, cx0, cy0, r=30):
    """在 (cx0,cy0) 附近 r 像素内找最亮斑质心；返回 (bx,by,peak,contrast) 或 None"""
    H, W = ir.shape
    y0, y1 = max(0, int(cy0 - r)), min(H, int(cy0 + r + 1))
    x0, x1 = max(0, int(cx0 - r)), min(W, int(cx0 + r + 1))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    p = cv2.GaussianBlur(ir[max(0, y0 - 5):min(H, y1 + 5), max(0, x0 - 5):min(W, x1 + 5)], (5, 5), 0).astype(np.float32)
    idx = int(np.argmax(p))
    py, px = divmod(idx, p.shape[1])
    peak = float(p[py, px])
    med = float(np.median(p))
    if peak - med < 30:
        return None
    m = p >= peak * 0.6
    yy, xx = np.nonzero(m)
    wts = p[yy, xx]
    bx = x0 - 5 + float((xx * wts).sum() / wts.sum())
    by = y0 - 5 + float((yy * wts).sum() / wts.sum())
    return bx, by, peak, peak - med


res = []
for s in sel:
    try:
        a = np.loadtxt(D / 'labels/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)
        c = np.loadtxt(D / 'labels_ir/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)
    except Exception:
        continue
    if a.shape[0] != 1 or c.shape[0] != 1:
        continue
    ir = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        continue
    irx, iry = c[0][1] * 640, c[0][2] * 512
    rgx, rgy = a[0][1] * 640, a[0][2] * 512
    b1 = blob(ir, irx, iry)   # 在 IR 标签附近找斑
    b2 = blob(ir, rgx, rgy)   # 在 RGB 标签附近找斑
    if b1 is None and b2 is None:
        continue
    d1 = np.hypot(b1[0] - irx, b1[1] - iry) if b1 else 999
    d2 = np.hypot(b2[0] - rgx, b2[1] - rgy) if b2 else 999
    res.append((d1, d2, (b1[3] if b1 else b2[3]),
                np.hypot(irx - rgx, iry - rgy)))

r = np.array(res)
print('n frames with a detectable blob near at least one label:', len(r))


def rep(msk, tag):
    if msk.sum() == 0:
        return
    print('%-14s n=%4d  |off|<=10px IR-label %.2f  RGB-label %.2f  (contrast %.0f, label gap %.1f px)'
          % (tag, int(msk.sum()),
             float(np.mean(r[msk, 0] <= 10)), float(np.mean(r[msk, 1] <= 10)),
             float(np.mean(r[msk, 2])), float(np.mean(r[msk, 3]))))


rep(np.ones(len(r), bool), 'ALL')
rep(r[:, 2] >= 60, 'strong blob')
rep(r[:, 2] < 60, 'weak blob')
rep(r[:, 3] <= 20, 'labels close')
rep(r[:, 3] > 20, 'labels apart')
