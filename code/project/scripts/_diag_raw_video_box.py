"""用原始视频帧直接画标签，排除任何导出/resize 步骤。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/rivermind-data/Anti-UAV300')
D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(1)
sel = random.sample(both, 6)

rows = []
for s in sel:
    seq = '_'.join(s.split('_')[:4])
    idx = int(s.split('_')[4])
    src = None
    for split in ('train', 'val', 'test'):
        p = ROOT / split / seq / 'infrared.mp4'
        if p.exists():
            src = p
            break
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for pth, color, tag in ((D / 'labels/train' / (s + '.txt'), (0, 200, 255), 'rgb'),
                            (D / 'labels_ir/train' / (s + '.txt'), (0, 0, 255), 'ir')):
        for ln in open(pth).read().strip().splitlines():
            t = ln.split()
            if len(t) < 5:
                continue
            x, y, w, h = [float(z) for z in t[1:5]]
            cv2.rectangle(v, (int((x - w / 2) * 640), int((y - h / 2) * 512)),
                          (int((x + w / 2) * 640), int((y + h / 2) * 512)), color, 1)
            cv2.putText(v, tag, (int((x - w / 2) * 640), max(12, int((y - h / 2) * 512) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    # 放大局部（以 IR 标签为中心）
    t = open(D / 'labels_ir/train' / (s + '.txt')).read().split()
    cx, cy = int(float(t[1]) * 640), int(float(t[2]) * 512)
    x0, y0 = max(0, cx - 90), max(0, cy - 70)
    crop = v[y0:y0 + 140, x0:x0 + 180]
    rows.append(cv2.resize(crop, (360, 280), interpolation=cv2.INTER_NEAREST))
    print(s, 'label ir center', cx, cy)
if rows:
    cv2.imwrite('/tmp/raw_label_zoom.jpg', np.vstack(rows))
    print('saved /tmp/raw_label_zoom.jpg')
