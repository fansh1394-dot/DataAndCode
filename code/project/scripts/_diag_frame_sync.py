"""确认两路视频是否逐帧对齐：比较红外亮斑在两路视频同一 idx 帧上的相对位置。"""
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
sel = random.sample(both, 150)

rows = []
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
    ci = cv2.VideoCapture(str(base / 'infrared.mp4'))
    cv2_ = cv2.VideoCapture(str(base / 'visible.mp4'))
    ci.set(cv2.CAP_PROP_POS_FRAMES, idx)
    cv2_.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok1, fi = ci.read()
    ok2, fv = cv2_.read()
    ci.release()
    cv2_.release()
    if not (ok1 and ok2):
        continue
    gi = cv2.GaussianBlur(cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gv = cv2.GaussianBlur(cv2.cvtColor(fv, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # 两路都归一化到 640x512 后取最亮像素（可见光下：目标为亮斑）
    ir_s = cv2.resize(gi, (640, 512))
    vis_s = cv2.resize(gv, (640, 512))
    iy, ix = np.unravel_index(int(np.argmax(ir_s)), ir_s.shape)
    vy, vx = np.unravel_index(int(np.argmax(vis_s)), vis_s.shape)
    if ir_s[iy, ix] < 120:
        continue
    rows.append((ix, iy, vx, vy, gi.shape[1], gi.shape[0], gv.shape[1], gv.shape[0]))

a = np.array(rows, dtype=np.float32)
print('n', len(rows))
print('IR video size set:', sorted(set(map(tuple, a[:, 4:6].astype(int).tolist()))))
print('VIS video size set:', sorted(set(map(tuple, a[:, 6:8].astype(int).tolist()))))
dx = a[:, 2] - a[:, 0]
dy = a[:, 3] - a[:, 1]
print('峰值像素位置差 vis-ir: dx mean %.1f median %.1f | dy mean %.1f median %.1f'
      % (dx.mean(), np.median(dx), dy.mean(), np.median(dy)))
print('|diff|<=10px frac: dx %.2f dy %.2f' % (np.mean(np.abs(dx) <= 10), np.mean(np.abs(dy) <= 10)))
