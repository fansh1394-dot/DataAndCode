"""验证 labels_ir 的 RHS 是否为原始框的 [x,y,w,h]（即把宽高当成了右下角坐标）。"""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(1)
sel = random.sample(both, 400)

cache = {}
res = []
for s in sel:
    seq = '_'.join(s.split('_')[:4])
    idx = int(s.split('_')[4])
    if seq not in cache:
        j = None
        for split in ('train', 'val', 'test'):
            p = ROOT / split / seq / 'infrared.json'
            if p.exists():
                j = json.loads(p.read_text())
                break
        cache[seq] = j
    j = cache[seq]
    if j is None:
        continue
    rects = j.get('gt_rect') or []
    if idx >= len(rects):
        continue
    r = rects[idx]
    if not r or len(r) < 4:
        continue
    x, y, w, h = [float(v) for v in r[:4]]
    if w <= 0 or h <= 0:
        continue
    try:
        lab = np.loadtxt(D / 'labels_ir/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
    except Exception:
        continue
    lx, ly, lw, lh = lab[1] * 640, lab[2] * 512, lab[3] * 640, lab[4] * 512
    res.append((
        abs(lx - (x + w / 2)), abs(ly - (y + h / 2)),
        abs(lx - (x + w) / 2), abs(ly - (y + h) / 2),
        lw, lh, w, h,
    ))
a = np.array(res)
print('n', len(a))
print('假设A RHS=[x,y,w,h]：(cx,cy)=((x+w)/2,(y+h)/2)  误差 p50 dx %.2f dy %.2f'
      % (np.percentile(a[:, 2], 50), np.percentile(a[:, 3], 50)))
print('假设B RHS=中心+宽高：      (cx,cy)=(x+w/2, y+h/2) 误差 p50 dx %.2f dy %.2f'
      % (np.percentile(a[:, 0], 50), np.percentile(a[:, 1], 50)))
print('导出宽/高 vs 原始宽/高: label_w p50 %.1f vs raw_w p50 %.1f ; label_h p50 %.1f vs raw_h p50 %.1f'
      % (np.percentile(a[:, 4], 50), np.percentile(a[:, 6], 50),
         np.percentile(a[:, 5], 50), np.percentile(a[:, 7], 50)))
