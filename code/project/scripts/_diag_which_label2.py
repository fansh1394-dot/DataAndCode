"""左右对照：同一批 IR 帧，左=RGB 标签画在 IR 上，右=IR 标签画在 IR 上。
只挑 RGB 标签也落在窗口内的帧，保证两边都看得见框。
"""
import json
from pathlib import Path

import cv2
import numpy as np

D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])

# 挑 RGB 与 IR 标签都在同一局部窗口内的帧（差 20~70px，最能体现偏移）
picked = []
for s in both[::37]:
    pr, pi = D / 'labels/train' / (s + '.txt'), D / 'labels_ir/train' / (s + '.txt')
    if not (pr.exists() and pi.exists()):
        continue
    try:
        a = np.loadtxt(pr, ndmin=2).reshape(-1, 5)
        c = np.loadtxt(pi, ndmin=2).reshape(-1, 5)
    except Exception:
        continue
    if len(a) != 1 or len(c) != 1:
        continue
    dy = (c[0][2] - a[0][2]) * 512
    if 20 <= abs(dy) <= 70:
        picked.append((s, a[0], c[0]))
    if len(picked) >= 6:
        break

print('picked', [p[0] for p in picked])
pairs = []
for s, ra, rc in picked:
    ir = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        continue
    H, W = ir.shape
    left = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    right = left.copy()
    cv2.rectangle(left, (int((ra[1] - ra[3] / 2) * W), int((ra[2] - ra[4] / 2) * H)),
                  (int((ra[1] + ra[3] / 2) * W), int((ra[2] + ra[4] / 2) * H)), (255, 0, 0), 1)
    cv2.rectangle(right, (int((rc[1] - rc[3] / 2) * W), int((rc[2] - rc[4] / 2) * H)),
                  (int((rc[1] + rc[3] / 2) * W), int((rc[2] + rc[4] / 2) * H)), (255, 0, 0), 1)
    cx, cy = rc[1] * W, rc[2] * H
    r = 55
    x0 = int(np.clip(cx - r, 0, W - 2 * r)); y0 = int(np.clip(cy - r, 0, H - 2 * r))
    l = left[y0:y0 + 2 * r, x0:x0 + 2 * r]
    rr = right[y0:y0 + 2 * r, x0:x0 + 2 * r]
    cell = np.hstack([l, np.full((2 * r, 4, 3), 255, np.uint8), rr])
    cv2.putText(cell, 'RGB-lbl', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(cell, 'IR-lbl', (2 * r + 7, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    pairs.append(cv2.resize(cell, (500, 240), interpolation=cv2.INTER_NEAREST))

if pairs:
    grid = np.vstack(pairs)
    cv2.imwrite('/tmp/which_label2.jpg', grid)
    print('saved /tmp/which_label2.jpg', grid.shape)
