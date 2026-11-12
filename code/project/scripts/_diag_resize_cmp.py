"""把原始视频帧与导出图叠在一起，看目标位置是否一致（排除 resize/裁剪）。"""
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
sel = random.sample(both, 4)

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
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
    e = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(g, e)
    print(s, 'src', g.shape, 'exp', e.shape, 'mean|diff| %.2f' % diff.mean())
    v = np.hstack([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                   cv2.cvtColor(e, cv2.COLOR_GRAY2BGR),
                   cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                     cv2.COLORMAP_JET)])
    rows.append(v)
if rows:
    cv2.imwrite('/tmp/resize_cmp.jpg', np.vstack(rows))
    print('saved /tmp/resize_cmp.jpg')
