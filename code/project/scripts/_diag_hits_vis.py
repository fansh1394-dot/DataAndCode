"""找出 RGB 标签贴近 IR 目标亮斑的帧，并生成可视化，用于确认标签绘制是否真的偏移。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(3)
sel = random.sample(both, 1500)
OUT = Path('/root/rivermind-data/MutilModel_3485988898/YOLO26/logs/_diag_panel')
OUT.mkdir(parents=True, exist_ok=True)


def blob(ir, cx0, cy0, r=30):
    H, W = ir.shape
    y0, y1 = max(0, int(cy0 - r)), min(H, int(cy0 + r + 1))
    x0, x1 = max(0, int(cx0 - r)), min(W, int(cx0 + r + 1))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    p = cv2.GaussianBlur(ir[y0:y1, x0:x1], (5, 5), 0).astype(np.float32)
    idx = int(np.argmax(p))
    py, px = divmod(idx, p.shape[1])
    peak = float(p[py, px])
    if peak - float(np.median(p)) < 60:
        return None
    m = p >= peak * 0.6
    yy, xx = np.nonzero(m)
    wts = p[yy, xx]
    return x0 + float((xx * wts).sum() / wts.sum()), y0 + float((yy * wts).sum() / wts.sum())


hits = []
for s in sel:
    try:
        a = np.loadtxt(D / 'labels/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)
        c = np.loadtxt(D / 'labels_ir/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)
    except Exception:
        continue
    if a.shape[0] != 1 or c.shape[0] != 1:
        continue
    ir = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        continue
    rgx, rgy = a[0][1] * 640, a[0][2] * 512
    b = blob(ir, rgx, rgy)
    if b is None:
        continue
    d = float(np.hypot(b[0] - rgx, b[1] - rgy))
    if d <= 10:
        hits.append((s, a[0].tolist(), c[0].tolist(), b, d))
print('hits', len(hits))
for h in hits[:6]:
    print(h[0], 'score=%.1f' % h[4], 'rgb', [round(v, 3) for v in h[1][1:5]], 'ir', [round(v, 3) for v in h[2][1:5]],
          'blob', [round(v, 1) for v in h[3]])

rows = []
for s, aa, cc, b, d in hits[:4]:
    rgb = cv2.imread(str(D / 'images/train' / (s + '.jpg')))
    ir = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        continue
    v = np.hstack([rgb, np.repeat(ir[..., None], 3, axis=2)])
    for lab, color, tag in ((aa, (0, 200, 255), 'rgb'), (cc, (0, 0, 255), 'ir')):
        x, y, w, h = lab[1:5]
        for off in (0, 640):
            cv2.rectangle(v, (int((x - w / 2) * 640) + off, int((y - h / 2) * 512)),
                          (int((x + w / 2) * 640) + off, int((y + h / 2) * 512)), color, 1)
            cv2.putText(v, tag, (int((x - w / 2) * 640) + off, max(12, int((y - h / 2) * 512) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.drawMarker(v, (int(b[0]) + 640, int(b[1])), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    rows.append(cv2.resize(v, (1280, 512)))
if rows:
    cv2.imwrite('/tmp/hit_grid.jpg', np.vstack(rows))
    print('saved /tmp/hit_grid.jpg')
