"""生成三栏对比图：RGB 面板 / IR 面板（同一套 RGB 标签）/ IR 面板（IR 自己的标签）。

左两栏直接取 YOLO11 run 目录里的 *_labels_rgb.jpg 与 *_labels_ir.jpg（同一批次、同一套 RGB 标签）；
右栏用 labels_ir 在 images_ir 上重画，作为"应该长什么样"的参照。
输出：_diag_vis/batch2_compare.jpg
"""
import json
from pathlib import Path

import cv2
import numpy as np

RUN = Path('/3250410011/root/rivermind-data/MutilModel_3485988898/ResTest/v2c_E1_rgbonly_s1')
D = Path('/root/rivermind-data/uav_data_v2')
OUT_LOCAL = Path('/tmp/batch2_compare.jpg')

rgb_panel = cv2.imread(str(RUN / 'train_batch2_labels_rgb.jpg'))
ir_panel = cv2.imread(str(RUN / 'train_batch2_labels_ir.jpg'))
print('rgb panel', rgb_panel.shape, 'ir panel', ir_panel.shape)

# ---- 第三栏：用 labels_ir 在 IR 原图上重画同一批帧 ----
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
# 取与 batch2 面板同一批 stem（面板标题里出现的前 6 个），逐个画单帧竖排
import re
# 标题叠在图上难以解析，改为：用与面板相同的 6x5 网格布局，随机抽 30 帧 both
rng = np.random.default_rng(2)
pick = rng.choice(len(both), size=30, replace=False)
cells = []
for k in pick:
    s = both[k]
    ir = cv2.imread(str(D / 'images_ir/train' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        cells.append(np.zeros((96, 128, 3), np.uint8))
        continue
    v = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    lp = D / 'labels_ir/train' / (s + '.txt')
    if lp.exists():
        for ln in lp.read_text().strip().splitlines():
            t = ln.split()
            if len(t) < 5:
                continue
            x, y, w, h = [float(z) for z in t[1:5]]
            H, W = v.shape[:2]
            cv2.rectangle(v, (int((x - w / 2) * W), int((y - h / 2) * H)),
                          (int((x + w / 2) * W), int((y + h / 2) * H)), (255, 0, 0), 1)
    cells.append(cv2.resize(v, (128, 96)))
grid = np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, 30, 6)])

# 统一高度后横向拼接三栏
h_target = rgb_panel.shape[0]
def fit(img):
    r = h_target / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * r), h_target))
g = fit(grid)
out = np.hstack([rgb_panel, ir_panel, g])
cv2.imwrite(str(OUT_LOCAL), out)
print('saved', OUT_LOCAL, out.shape)
