"""排查：复现 batch39 的样本，检查 IR 通道数值与 images_ir 原图是否一致。"""
import sys
from pathlib import Path

sys.path.insert(0, '/3250410011/root/rivermind-data/MutilModel_3485988898')

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.mm.utils import get_modality_path_from_data, resolve_modality_pair
from ultralytics.utils import YAML

RUN = Path('/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest/v26_E1_rgbonly_s0')
a = YAML.load(RUN / 'args.yaml')
cfg = get_cfg(overrides={k: v for k, v in a.items()
                         if k in ('imgsz', 'batch', 'cache', 'rect', 'single_cls',
                                  'fraction', 'seed', 'workers', 'task')})
for k, v in a.items():
    if k.startswith('mm_'):
        setattr(cfg, k, v)
for k in ('mosaic', 'mixup', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
          'shear', 'perspective', 'flipud', 'fliplr', 'bgr', 'use_multimodal_aug'):
    if k in a:
        setattr(cfg, k, a[k])

data = check_det_dataset(a['data'])
_rgb, x_mod = resolve_modality_pair(data, strict=True)
x_dir = get_modality_path_from_data(data, x_mod)
print('x_mod', x_mod, 'x_dir', x_dir, 'Xch', data.get('Xch'))

ds = build_yolo_dataset(cfg, data['train'], 32, data, mode='train', rect=False, stride=32,
                        multi_modal_image=True, x_modality=x_mod, x_modality_dir=x_dir)
print('precision_contract', ds.precision_contract)
for i in (1248, 1249, 1250):
    s = ds[i]
    img = s['img']
    print(f'--- idx {i}  {Path(s["im_file"]).name}  img {tuple(img.shape)} {img.dtype}')
    print('   rgb ch mean', [round(float(img[c].float().mean()), 1) for c in range(3)])
    print('   ir  ch mean', [round(float(img[c].float().mean()), 1) for c in range(3, 6)])
    print('   ir  max', round(float(img[3:6].float().max()), 1))
    print('   bboxes', s['bboxes'].tolist(), 'bboxes_ir', s.get('bboxes_ir', torch.zeros(0)).tolist())
