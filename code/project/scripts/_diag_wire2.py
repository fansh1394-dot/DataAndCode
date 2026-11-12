"""接线验证（终端加固）：同一帧、同一 IR 图，对比
  A) bboxes_x=bboxes_ir  -> IR 面板应贴合热斑
  B) bboxes_x=None       -> IR 面板回退用 RGB 标签，应偏上
并把两个 IR 面板拼成一张图。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/3250410011/root/rivermind-data/MutilModel_3485988898')
sys.path.insert(0, str(ROOT))
from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter  # noqa: E402

D = Path('/root/rivermind-data/uav_data_v2')
OUT = Path('/tmp/_wire2')
for t in ('A_with', 'B_without'):
    (OUT / t).mkdir(parents=True, exist_ok=True)

stem = '20190925_101846_1_1_00000'
rgb = cv2.imread(str(D / 'images/train' / (stem + '.jpg')))
ir = cv2.imread(str(D / 'images_ir/train' / (stem + '.jpg')), cv2.IMREAD_GRAYSCALE)
H, W = rgb.shape[:2]
mm = np.concatenate([rgb, np.repeat(ir[..., None], 3, axis=2)], axis=2)
img = torch.from_numpy(mm.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

lr = np.loadtxt(D / 'labels/train' / (stem + '.txt')).reshape(-1, 5)[0]
li = np.loadtxt(D / 'labels_ir/train' / (stem + '.txt')).reshape(-1, 5)[0]
b_rgb = torch.tensor(np.array([lr[1:5]]), dtype=torch.float32)
b_ir = torch.tensor(np.array([li[1:5]]), dtype=torch.float32)
bi = torch.zeros(1, dtype=torch.long)
cls = torch.zeros(1, dtype=torch.float32)

for tag, bx in (('A_with', b_ir), ('B_without', None)):
    MultiModalPlotter(OUT / tag, on_plot=None).plot_batch(
        stage='train', source='labels', images=img,
        batch_idx=bi, cls=cls, bboxes=b_rgb,
        bboxes_x=bx, cls_x=cls if bx is not None else None,
        batch_idx_x=bi if bx is not None else None,
        xch=3, x_modality_name='ir',
        precision_contract={'rgb': 'legacy', 'x': 'legacy'},
        modality_mode='dual',
        paths=[stem], batch_ni=0, threaded=False,
    )

# 只取 IR 面板，叠成上下两张做对比
a = cv2.imread(str(OUT / 'A_with' / 'train_batch0_labels_ir.jpg'))
b = cv2.imread(str(OUT / 'B_without' / 'train_batch0_labels_ir.jpg'))
print('A', a.shape, 'B', b.shape)
cv2.putText(a, 'A: bboxes_x = bboxes_ir (correct)', (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(b, 'B: bboxes_x = None (fallback to RGB label)', (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite('/tmp/wire_AB.jpg', np.vstack([b, np.full((6, a.shape[1], 3), 255, np.uint8), a]))
print('saved /tmp/wire_AB.jpg')
