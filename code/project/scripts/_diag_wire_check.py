"""验证 YOLO26 plot_training_samples 的接线：直接构造 batch，走真实 MultiModalPlotter，
分别以 bboxes_ir 有 / 无 两种输入绘图，比较 IR 面板里框的位置是否不同（不同=接线生效）。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/3250410011/root/rivermind-data/MutilModel_3485988898')
sys.path.insert(0, str(ROOT))

from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter  # noqa: E402

D = Path('/root/rivermind-data/uav_data_v2')
OUT = Path('/tmp/_wire_check')
OUT.mkdir(parents=True, exist_ok=True)

# 取一帧，RGB 标签与 IR 标签差 ~38px（y 方向）
stem = '20190925_101846_1_1_00000'
rgb = cv2.imread(str(D / 'images/train' / (stem + '.jpg')))
ir = cv2.imread(str(D / 'images_ir/train' / (stem + '.jpg')), cv2.IMREAD_GRAYSCALE)
H, W = rgb.shape[:2]
mm = np.concatenate([rgb, np.repeat(ir[..., None], 3, axis=2)], axis=2)  # H,W,6
img = torch.from_numpy(mm.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

lr = np.loadtxt(D / 'labels/train' / (stem + '.txt')).reshape(-1, 5)[0]
li = np.loadtxt(D / 'labels_ir/train' / (stem + '.txt')).reshape(-1, 5)[0]
b_rgb = torch.tensor([lr[1:5]], dtype=torch.float32)
b_ir = torch.tensor([li[1:5]], dtype=torch.float32)
bi = torch.zeros(1, dtype=torch.long)
cls = torch.zeros(1, dtype=torch.float32)

print('RGB label xywh(norm) =', b_rgb.tolist())
print('IR  label xywh(norm) =', b_ir.tolist())
print('两者中心 dy = %.1f px' % ((li[2] - lr[2]) * 512))

for tag, bx in (('with_bboxes_ir', b_ir), ('without_bboxes_ir', None)):
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    p = MultiModalPlotter(outdir, on_plot=None)
    p.plot_batch(
        stage='train', source='labels', images=img,
        batch_idx=bi, cls=cls, bboxes=b_rgb,
        bboxes_x=bx, cls_x=cls if bx is not None else None,
        batch_idx_x=bi if bx is not None else None,
        xch=3, x_modality_name='ir',
        precision_contract={'rgb': 'legacy', 'x': 'legacy'},
        modality_mode='dual',
        paths=[stem], batch_ni=0, threaded=False,
    )
    print('drew', tag, '->', sorted(x.name for x in outdir.glob('*.jpg')))
