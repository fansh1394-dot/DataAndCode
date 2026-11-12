"""把 containers 未覆盖的样本画出来，确认残留偏移的性质。"""
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
sel = random.sample(both, 400)

bad = []
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
        ib = np.loadtxt(D / 'labels_ir/val' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
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
    m = p >= peak * 0.6
    yy, xx = np.nonzero(m)
    w = p[yy, xx]
    bx = x0 + float((xx * w).sum() / w.sum())
    by = y0 + float((yy * w).sum() / w.sum())
    wb, hb = ib[3] * 640, ib[4] * 512
    inside = abs(bx - cx) <= wb / 2 and abs(by - cy) <= hb / 2
    if not inside:
        bad.append((s, base, idx, cx, cy, wb, hb, bx, by))

print('bad', len(bad))
rows = []
for s, base, idx, cx, cy, wb, hb, bx, by in bad[:6]:
    ci = cv2.VideoCapture(str(base / 'infrared.mp4'))
    ci.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fi = ci.read()
    ci.release()
    g = cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY), (5, 5), 0), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(g, (int(cx - wb / 2), int(cy - hb / 2)), (int(cx + wb / 2), int(cy + hb / 2)), (0, 0, 255), 1)
    cv2.drawMarker(g, (int(bx), int(by)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
    x0, y0 = max(0, int(min(cx, bx)) - 60), max(0, int(min(cy, by)) - 50)
    crop = g[y0:y0 + 120, x0:x0 + 160]
    rows.append(cv2.resize(crop, (320, 240), interpolation=cv2.INTER_NEAREST))
while len(rows) % 3:
    rows.append(np.zeros_like(rows[0]))
if rows:
    grid = np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)])
    cv2.imwrite('/tmp/bad_cases.jpg', grid)
    print('saved /tmp/bad_cases.jpg')
