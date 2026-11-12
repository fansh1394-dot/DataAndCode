"""同一批 IR 帧、同一标尺下的左右对照：
  左 = RGB 标签（labels/）画在 IR 图上（YOLO11 的 *_labels_ir.jpg 实际画的就是这个）
  右 = IR 标签（labels_ir/）画在 IR 图上（正确画法）
每格放大到能看清框与热斑的相对位置。
"""
import json
from pathlib import Path

import cv2
import numpy as np

D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])

SEL = ['20190925_201320_1_1_00340', '20190925_153100_1_3_00245',
       '20190926_262414_1_1_00160', '20190926_255300_1_1_00225',
       '20190925_212000_1_1_00080', '20190926_180000_1_1_00120']
SEL = [s for s in SEL if (D / 'images_ir/train' / (s + '.jpg')).exists()]
if len(SEL) < 6:
    # 用序列前缀 + 帧号兜底：按面板里出现的序列名挑
    import re
    cands = [s for s in both if s.startswith(('20190925_201320', '20190925_153100',
                                             '20190926_262414', '20190926_255300',
                                             '20190925_2120', '20190926_18'))]
    SEL = (SEL + cands)[:6]

pairs = []
for s in SEL:
    ir = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        continue
    H, W = ir.shape
    left = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    for path, canvas, color in ((D / 'labels/train' / (s + '.txt'), left, (255, 0, 0)),
                                (D / 'labels_ir/train' / (s + '.txt'), right, (255, 0, 0))):
        if not path.exists():
            continue
        for ln in path.read_text().strip().splitlines():
            t = ln.split()
            if len(t) < 5:
                continue
            x, y, w, h = [float(z) for z in t[1:5]]
            cv2.rectangle(canvas, (int((x - w / 2) * W), int((y - h / 2) * H)),
                          (int((x + w / 2) * W), int((y + h / 2) * H)), color, 1)
    # 以 IR 标签为中心裁同样大小的窗口
    p = D / 'labels_ir/train' / (s + '.txt')
    cx, cy = W / 2, H / 2
    if p.exists():
        t = p.read_text().split()
        if len(t) >= 5:
            cx, cy = float(t[1]) * W, float(t[2]) * H
    r = 55
    x0, y0 = int(np.clip(cx - r, 0, W - 2 * r)), int(np.clip(cy - r, 0, H - 2 * r))
    l = left[y0:y0 + 2 * r, x0:x0 + 2 * r]
    rr = right[y0:y0 + 2 * r, x0:x0 + 2 * r]
    cell = np.hstack([l, np.full((2 * r, 4, 3), 255, np.uint8), rr])
    pairs.append(cv2.resize(cell, (456, 220), interpolation=cv2.INTER_NEAREST))

if pairs:
    grid = np.vstack(pairs)
    cv2.putText(grid, 'LEFT: RGB label on IR   |   RIGHT: IR label on IR',
                (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite('/tmp/which_label.jpg', grid)
    print('saved /tmp/which_label.jpg', grid.shape, 'n', len(pairs))
