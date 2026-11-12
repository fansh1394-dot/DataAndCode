import json
import random

import cv2
import numpy as np

D = '/root/rivermind-data/uav_data_v2/'
frames = json.load(open(D + 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(1)
sel = random.sample(both, 2000)
res = []
for s in sel:
    try:
        a = np.loadtxt(D + 'labels/train/' + s + '.txt', ndmin=2)
        c = np.loadtxt(D + 'labels_ir/train/' + s + '.txt', ndmin=2)
    except Exception:
        continue
    if a.shape[0] != 1 or c.shape[0] != 1:
        continue
    ir = cv2.imread(D + 'images_ir/train/' + s + '.jpg', 0)
    if ir is None:
        continue
    res.append((s, ir.mean(), float(a[0][2]), float(c[0][2])))
print('n', len(res))
m = np.array([r[1] for r in res])
print('IR mean pct 10/50/90', np.percentile(m, [10, 50, 90]).round(1))
# 亮背景（IR 均值高）vs 暗背景，两套标签的 dy 差异
dy = np.array([(r[3] - r[2]) * 512 for r in res])
bins = [(0, 60), (60, 120), (120, 256)]
for lo, hi in bins:
    k = (m >= lo) & (m < hi)
    if k.sum():
        print(f'IR mean [{lo},{hi}): n={int(k.sum())}  mean dy(ir-rgb)={dy[k].mean():.1f}px  '
              f'|dy| p50={np.percentile(np.abs(dy[k]), 50):.1f}  frac|dy|>15={np.mean(np.abs(dy[k]) > 15):.2f}')
# 与 IR 峰值位置无关，只看两套标签的一致性
agree = np.abs(dy) <= 15
print('label-agreement (|dy|<=15px) frac =', round(float(agree.mean()), 3))
