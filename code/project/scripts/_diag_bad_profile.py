"""亮斑质心落在框外的样本，其参数（框大小/位置/序列/帧号）有什么共性。"""
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

ok, bad = [], []
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
    okr, fi = ci.read()
    ci.release()
    if not okr:
        continue
    g = cv2.GaussianBlur(cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    cy, cx = int(ib[2] * 512), int(ib[1] * 640)
    r = 40
    y0, y1 = max(0, cy - r), min(512, cy + r + 1)
    x0, x1 = max(0, cx - r), min(640, cx + r + 1)
    p = g[y0:y1, x0:x1]
    ks = int(np.argmax(p))
    py, px = divmod(ks, p.shape[1])
    if float(p[py, px]) - float(np.median(p)) < 80:
        continue
    m = p >= p[py, px] * 0.6
    yy, xx = np.nonzero(m)
    w = p[yy, xx]
    bx = x0 + float((xx * w).sum() / w.sum())
    by = y0 + float((yy * w).sum() / w.sum())
    wb, hb = ib[3] * 640, ib[4] * 512
    rec = dict(stem=s, seq=seq, idx=idx, cx=cx, cy=cy, w=wb, h=hb,
               dx=bx - (cx - wb / 2), dy=by - (cy - hb / 2),
               ratio=wb / max(hb, 1e-6), peak=float(p[py, px]))
    (ok if (abs(bx - cx) <= wb / 2 and abs(by - cy) <= hb / 2) else bad).append(rec)

print('ok', len(ok), 'bad', len(bad))
for tag, arr in (('ok', ok), ('bad', bad)):
    w = np.array([a['w'] for a in arr])
    h = np.array([a['h'] for a in arr])
    ra = np.array([a['ratio'] for a in arr])
    print(f'{tag:3s}  w p50 {np.median(w):6.1f}  h p50 {np.median(h):5.1f}  w/h p50 {np.median(ra):5.2f}  '
          f'w/h>2.5 比例 {np.mean(ra > 2.5):.2f}')
seqs_bad = {}
for a in bad:
    seqs_bad[a['seq']] = seqs_bad.get(a['seq'], 0) + 1
print('bad 的序列分布:', sorted(seqs_bad.items(), key=lambda t: -t[1])[:8])
for a in bad[:10]:
    print('  %s  w=%.0f h=%.0f cx=%.0f cy=%.0f  blob偏离框左上=(%.0f,%.0f)' %
          (a['stem'], a['w'], a['h'], a['cx'], a['cy'], a['dx'], a['dy']))
