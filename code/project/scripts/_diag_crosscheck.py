"""对照：IR 标签 vs IR 亮斑（已知很准）与 IR 标签 vs 可见光推理峰值，给出可比的指标。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(7)
sel = random.sample(both, 200)


def blob_peak(arr, cx, cy, r=30):
    H, W = arr.shape
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r + 1))
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r + 1))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    p = cv2.GaussianBlur(arr[y0:y1, x0:x1], (5, 5), 0).astype(np.float32)
    idx = int(np.argmax(p))
    py, px = divmod(idx, p.shape[1])
    if p[py, px] - float(np.median(p)) < 25:
        return None
    m = p >= p[py, px] * 0.6
    yy, xx = np.nonzero(m)
    w = p[yy, xx]
    return x0 + float((xx * w).sum() / w.sum()), y0 + float((yy * w).sum() / w.sum())


ir_ok = []
vis_ok = []
for s in sel:
    seq = '_'.join(s.split('_')[:4])
    idx = int(s.split('_')[4])
    base = None
    for split in ('train', 'val', 'test'):
        p = ROOT / split / seq
        if (p / 'infrared.mp4').exists():
            base = p
            break
    if base is None:
        continue
    try:
        lab = np.loadtxt(D / 'labels_ir/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
    except Exception:
        continue
    lx, ly = lab[1] * 640, lab[2] * 512
    ci = cv2.VideoCapture(str(base / 'infrared.mp4'))
    cvi = cv2.VideoCapture(str(base / 'visible.mp4'))
    ci.set(cv2.CAP_PROP_POS_FRAMES, idx)
    cvi.set(cv2.CAP_PROP_POS_FRAMES, idx)
    o1, fi = ci.read()
    o2, fv = cvi.read()
    ci.release()
    cvi.release()
    if not (o1 and o2):
        continue
    gi = cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY)
    gv = cv2.cvtColor(cv2.resize(fv, (640, 512)), cv2.COLOR_BGR2GRAY)
    b1 = blob_peak(gi, lx, ly)
    b2 = blob_peak(gv, lx, ly)
    if b1:
        ir_ok.append(np.hypot(b1[0] - lx, b1[1] - ly))
    if b2:
        vis_ok.append(np.hypot(b2[0] - lx, b2[1] - ly))

for name, v in (('IR 帧内 标签→亮斑', ir_ok), ('可见光帧内 标签→亮斑', vis_ok)):
    v = np.array(v)
    if len(v):
        print('%-22s n=%3d  |off| p50 %.2f p75 %.2f p90 %.2f  frac<=10px %.2f'
              % (name, len(v), *np.percentile(v, [50, 75, 90]), float(np.mean(v <= 10))))
