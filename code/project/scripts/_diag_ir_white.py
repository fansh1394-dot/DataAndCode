"""查 IR 底图为何全白：对比派生 yaml 与原始 yaml 下 dataset 样本的通道数值。"""
import sys
from pathlib import Path

sys.path.insert(0, '/3250410011/root/rivermind-data/MutilModel_3485988898')

import numpy as np
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.mm.utils import get_modality_path_from_data, resolve_modality_pair
from ultralytics.utils import YAML
from ultralytics.utils.patches import imread

RUN = Path('/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest/v26_E1_rgbonly_s0')
a = YAML.load(RUN / 'args.yaml')
DERIVED = '/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/_d2p1/eval_yamls/plot_dual_data.yaml'


def build(yaml_path):
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
    data = check_det_dataset(yaml_path)
    _rgb, x_mod = resolve_modality_pair(data, strict=True)
    x_dir = get_modality_path_from_data(data, x_mod)
    ds = build_yolo_dataset(cfg, data['train'], 32, data, mode='train', rect=False, stride=32,
                            multi_modal_image=True, x_modality=x_mod, x_modality_dir=x_dir)
    return ds, data, x_mod, x_dir


for tag, y in (('original', a['data']), ('derived', DERIVED)):
    ds, data, x_mod, x_dir = build(y)
    s = ds[1248]
    img = s['img']
    print(f'[{tag}] labels={data.get("labels")} x_mod={x_mod} x_dir={x_dir} '
          f'dual_configured={getattr(ds, "dual_labels_configured", None)}')
    print('   img dtype', img.dtype, 'rgb mean', [round(float(img[c].float().mean()), 1) for c in range(3)],
          'ir mean', [round(float(img[c].float().mean()), 1) for c in range(3, 6)])
    xp = ds._x_paths[1248]
    print('   x_path', xp)
    raw = imread(xp)
    print('   raw x dtype', raw.dtype, 'shape', raw.shape, 'mean', round(float(np.asarray(raw).mean()), 2),
          'p1/p50/p99', np.percentile(raw, [1, 50, 99]))
    print('   bboxes_ir', (s.get('bboxes_ir') if 'bboxes_ir' in s else None))
