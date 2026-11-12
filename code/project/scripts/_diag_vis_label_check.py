import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/val.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(5)
sel = random.sample(both, 6)

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
    cvi = cv2.VideoCapture(str(base / 'visible.mp4'))
    cvi.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fv = cvi.read()
    cvi.release()
    if not ok:
        continue
    try:
        vb = np.loadtxt(D / 'labels/val' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
        ib = np.loadtxt(D / 'labels_ir/val' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
    except Exception:
        continue
    v = fv.copy()
    for lab, color, tag in ((vb, (0, 200, 255), 'rgb'), (ib, (0, 0, 255), 'ir')):
        x, y, w, h = lab[1:5]
        x1, y1 = int((x - w / 2) * 1920), int((y - h / 2) * 1080)
        x2, y2 = int((x + w / 2) * 1920), int((y + h / 2) * 1080)
        cv2.rectangle(v, (x1, y1), (x2, y2), color, 2)
        cv2.putText(v, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cx, cy = int(vb[1] * 1920), int(vb[2] * 1080)
    crop = v[max(0, cy - 150):cy + 150, max(0, cx - 220):cx + 220]
    if crop.size == 0:
        continue
    rows.append(cv2.resize(crop, (440, 300)))
    print(s, 'rgb label center', cx, cy, 'ir label center', int(ib[1] * 1920), int(ib[2] * 1080))
if rows:
    while len(rows) % 2:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
    cv2.imwrite('/tmp/vis_label_check.jpg', grid)
    print('saved /tmp/vis_label_check.jpg')
