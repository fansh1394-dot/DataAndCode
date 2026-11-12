"""框内是否基本覆盖亮斑：用相对亮度阈值（而非绝对 0.6*peak）度量覆盖率。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
for split in ('train', 'val'):
    frames = json.load(open(D / f'metadata/{split}.json'))['frames']
    both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
    random.seed(11)
    sel = random.sample(both, 400)
    cov, cen, n = [], [], 0
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
        H, W = g.shape
        x = ib[1] * W, ib[2] * H, ib[3] * W, ib[4] * H
        cx, cy, w, h = x
        x1, y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
        x2, y2 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        # 全图背景中位数
        bg = float(np.median(g))
        pmax = float(g[y1:y2, x1:x2].max())
        thr = bg + 0.35 * (pmax - bg)
        mask = g > thr
        # 只考虑离该框较近的亮块
        num, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if num <= 1:
            continue
        # 取面积最大的连通块（目标）
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh = stats[k, :4]
        tc = cents[k]
        inter = max(0, min(x2, bx + bw) - max(x1, bx)) * max(0, min(y2, by + bh) - max(y1, by))
        cov.append(inter / max(1, bw * bh))
        cen.append(abs(tc[0] - cx) <= w / 2 and abs(tc[1] - cy) <= h / 2)
        n += 1
    cov = np.array(cov)
    print(f'{split}: n={n}  亮斑块被框覆盖率 p10/p50/p90 = {np.round(np.percentile(cov, [10, 50, 90]), 2)}  '
          f'完全覆盖(>=0.9)比例 {np.mean(cov >= 0.9):.2f}  未覆盖(<=0.1)比例 {np.mean(cov <= 0.1):.2f}  '
          f'质心在框内 {np.mean(cen):.2f}')
