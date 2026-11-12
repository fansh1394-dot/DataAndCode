"""标注框是否落在红外亮斑上：按 '框内是否含显著亮斑' + '亮斑质心是否落在框内' 两指标统计。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
out = {}
for split in ('train', 'val'):
    frames = json.load(open(D / f'metadata/{split}.json'))['frames']
    both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
    random.seed(11)
    sel = random.sample(both, min(400, len(both)))
    inside = 0
    dists = []
    n = 0
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
        try:
            ib = np.loadtxt(D / f'labels_ir/{split}' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
        except Exception:
            continue
        ci = cv2.VideoCapture(str(base / 'infrared.mp4'))
        ci.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fi = ci.read()
        ci.release()
        if not ok:
            continue
        g = cv2.GaussianBlur(cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        cy, cx = int(ib[2] * 512), int(ib[1] * 640)
        r = 40
        y0, y1 = max(0, cy - r), min(512, cy + r + 1)
        x0, x1 = max(0, cx - r), min(640, cx + r + 1)
        p = g[y0:y1, x0:x1]
        ks = int(np.argmax(p))
        py, px = divmod(ks, p.shape[1])
        peak = float(p[py, px])
        if peak - float(np.median(p)) < 80:
            continue
        n += 1
        m = p >= peak * 0.6
        yy, xx = np.nonzero(m)
        w = p[yy, xx]
        bx = x0 + float((xx * w).sum() / w.sum())
        by = y0 + float((yy * w).sum() / w.sum())
        w_box, h_box = ib[3] * 640, ib[4] * 512
        if abs(bx - cx) <= w_box / 2 and abs(by - cy) <= h_box / 2:
            inside += 1
        dists.append(np.hypot(bx - cx, by - cy))
    dists = np.array(dists)
    out[split] = (n, inside / max(1, n), np.percentile(dists, [50, 75, 90]) if n else [])
    print(f'{split}: n={n}  亮斑质心落在框内比例={inside / max(1, n):.3f}  '
          f'距离 p50/p75/p90 = {np.round(out[split][2], 2)}')
