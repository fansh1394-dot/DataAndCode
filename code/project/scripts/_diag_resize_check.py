"""对比原图 vs 导出图：目标亮斑在两张图上的像素位置，确认 images_ir 是否被 resize 拉伸。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(1)
sel = random.sample(both, 120)

rows = []
for s in sel:
    seq = '_'.join(s.split('_')[:4])
    idx = int(s.split('_')[4])
    src = None
    for split in ('train', 'val', 'test'):
        p = ROOT / split / seq / 'infrared.mp4'
        if p.exists():
            src = p
            break
    if src is None:
        continue
    cap = cv2.VideoCapture(str(src))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
    g = cv2.GaussianBlur(g, (5, 5), 0)
    iy, ix = np.unravel_index(int(np.argmax(g)), g.shape)
    exp = D / 'images_ir/train' / (s + '.jpg')
    e = cv2.imread(str(exp), cv2.IMREAD_GRAYSCALE)
    if e is None:
        continue
    e = cv2.GaussianBlur(e, (5, 5), 0)
    ey, ex = np.unravel_index(int(np.argmax(e)), e.shape)
    rows.append((W, H, e.shape[1], e.shape[0], ix, iy, ex, ey, g.max(), e.max()))

a = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
b = np.array([[r[4], r[5], r[6], r[7]] for r in rows], dtype=np.float32)
print('n', len(rows))
print('src size set:', sorted(set(map(tuple, a[:, :2].astype(int).tolist()))))
print('exp size set:', sorted(set(map(tuple, a[:, 2:].astype(int).tolist()))))
sx, sy = b[:, 0] / a[:, 0], b[:, 1] / a[:, 1]
exn, eyn = b[:, 2] / a[:, 2], b[:, 3] / a[:, 3]
print('归一化峰值坐标  src mean (%.4f, %.4f)  exp mean (%.4f, %.4f)' % (sx.mean(), sy.mean(), exn.mean(), eyn.mean()))
print('差异 mean |dx| %.4f  |dy| %.4f' % (np.abs(sx - exn).mean(), np.abs(sy - eyn).mean()))
