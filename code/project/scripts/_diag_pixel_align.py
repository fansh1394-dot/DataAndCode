"""导出图 vs 原始帧：用亮斑最亮像素的整数坐标比对，判定 image_ir 是否与原始帧逐像素一致。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/val.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(11)
sel = random.sample(both, 150)

same = 0
near = 0
n = 0
shifts = []
exact = 0
for s in sel:
    seq = '_'.join(s.split('_')[:4])
    idx = int(s.split('_')[4])
    base = None
    for sp in ('train', 'val', 'test'):
        p = ROOT / sp / seq
        if (p / 'infrared.mp4').exists():
            base = p
            break
    if base is None:
        continue
    ci = cv2.VideoCapture(str(base / 'infrared.mp4'))
    ci.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fi = ci.read()
    ci.release()
    if not ok:
        continue
    gi = cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY) if fi.ndim == 3 else fi
    e = cv2.imread(str(D / f'images_ir/val' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if e is None or e.shape != gi.shape:
        continue
    d = cv2.absdiff(gi, e)
    n += 1
    if d.max() <= 2:
        exact += 1
    # 用相位相关估位移
    g1 = np.float32(gi)
    g2 = np.float32(e)
    shift, resp = cv2.phaseCorrelate(g1, g2)
    shifts.append(shift)
    iy, ix = np.unravel_index(int(np.argmax(cv2.GaussianBlur(gi, (5, 5), 0))), gi.shape)
    ey, ex = np.unravel_index(int(np.argmax(cv2.GaussianBlur(e, (5, 5), 0))), e.shape)
    if (ix, iy) == (ex, ey):
        same += 1
    if abs(ix - ex) <= 1 and abs(iy - ey) <= 1:
        near += 1
sh = np.array(shifts)
print('n', n)
print('逐像素几乎相同(maxdiff<=2) 比例 %.3f' % (exact / max(1, n)))
print('峰值像素完全一致 %.3f  ±1px %.3f' % (same / max(1, n), near / max(1, n)))
print('phaseCorrelate 位移: dx mean %.2f std %.2f ; dy mean %.2f std %.2f'
      % (sh[:, 0].mean(), sh[:, 0].std(), sh[:, 1].mean(), sh[:, 1].std()))
print('|shift| p90 %.2f  max %.2f' % (np.percentile(np.hypot(sh[:, 0], sh[:, 1]), 90),
                                       np.hypot(sh[:, 0], sh[:, 1]).max()))
