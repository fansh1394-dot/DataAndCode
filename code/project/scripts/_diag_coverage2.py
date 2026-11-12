"""覆盖率诊断（修正版）：只在框周围的局部窗口内找目标亮块，避免被远处大块亮区带偏。"""
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
    cov, cen = [], []
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
        cx, cy = ib[1] * W, ib[2] * H
        w, h = ib[3] * W, ib[4] * H
        rad = 45
        y0, y1 = max(0, int(cy - rad)), min(H, int(cy + rad + 1))
        x0, x1 = max(0, int(cx - rad)), min(W, int(cx + rad + 1))
        p = g[y0:y1, x0:x1]
        bg = float(np.median(p))
        pmax = float(p.max())
        if pmax - bg < 80:
            cov.append(np.nan)
            cen.append(False)
            continue
        thr = bg + 0.35 * (pmax - bg)
        mask = (p > thr).astype(np.uint8)
        num, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        if num <= 1:
            cov.append(np.nan)
            cen.append(False)
            continue
        # 含最亮像素的那块
        py, px = np.unravel_index(int(np.argmax(p)), p.shape)
        k = lab[py, px]
        if k == 0:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh = stats[k, :4]
        gx1, gy1, gx2, gy2 = bx + x0, by + y0, bx + bw + x0, by + bh + y0
        tx1, ty1, tx2, ty2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        inter = max(0, min(tx2, gx2) - max(tx1, gx1)) * max(0, min(ty2, gy2) - max(ty1, gy1))
        cov.append(inter / max(1.0, bw * bh))
        tc = cents[k]
        cen.append(abs(tc[0] + x0 - cx) <= w / 2 and abs(tc[1] + y0 - cy) <= h / 2)
    cov = np.array(cov)
    ok = ~np.isnan(cov)
    print(f'{split}: n={int(ok.sum())}  覆盖率 p10/p50/p90 = '
          f'{np.round(np.percentile(cov[ok], [10, 50, 90]), 2)}  '
          f'>=0.9 比例 {np.mean(cov[ok] >= 0.9):.2f}  <=0.1 比例 {np.mean(cov[ok] <= 0.1):.2f}  '
          f'质心在框内 {np.mean(np.array(cen)[ok]):.2f}')
