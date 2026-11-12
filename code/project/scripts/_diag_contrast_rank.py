"""按标签是否落在 IR 亮斑上（并给出对比度）做分档统计。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(11)
sel = random.sample(both, 400)

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
    try:
        ib = np.loadtxt(D / 'labels_ir/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
    except Exception:
        continue
    ci = cv2.VideoCapture(str(base / 'infrared.mp4'))
    ci.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fi = ci.read()
    ci.release()
    if not ok:
        continue
    g = cv2.GaussianBlur(cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    x, y, w, h = ib[1] * 640, ib[2] * 512, ib[3] * 640, ib[4] * 512
    x1, y1 = int(max(0, x - w / 2)), int(max(0, y - h / 2))
    x2, y2 = int(min(640, x + w / 2)), int(min(512, y + h / 2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        continue
    sub = g[y1:y2, x1:x2]
    outside = []
    r = max(w, h)
    for dy in (-r, r):
        ya, yb = int(max(0, y + dy - h / 2)), int(min(512, y + dy + h / 2))
        if yb - ya > 3:
            outside.append(g[ya:yb, x1:x2])
    bg = float(np.median(np.concatenate([o.ravel() for o in outside]))) if outside else float(np.median(g))
    contrast = float(sub.max()) - bg
    rows.append((contrast, s))

rows.sort(key=lambda t: -t[0])
n = len(rows)
print('n', n)
c = np.array([r[0] for r in rows])
print('框内峰值-背景中位数  contrast p10/p25/p50/p75/p90 =',
      np.round(np.percentile(c, [10, 25, 50, 75, 90]), 1))
print('contrast < 40 的比例 = %.3f' % float(np.mean(c < 40)))
print('最弱 8 帧:', [(r[1], round(r[0], 1)) for r in rows[-8:]])
print('最强 5 帧:', [(r[1], round(r[0], 1)) for r in rows[:5]])
Path('/tmp/weak_frames.txt').write_text('\n'.join(f'{r[1]} {r[0]:.1f}' for r in rows[-12:]))
