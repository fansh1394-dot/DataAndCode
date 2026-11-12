# -*- coding: utf-8 -*-
"""诊断：dataset 到底从哪个目录读标签。

data.yaml 现在写的是 labels:{rgb: labels_ir, ir: labels_ir}（E3 启动时被 --label-from ir
改写），但 E3 的训练 loss 与 E2 逐位相同、且 cache 落在 labels/ 下 —— 说明实际读的
仍是 labels/。本脚本直接打印 dataset 解析出的 primary/secondary 标签目录与派生路径。
"""
import sys
from pathlib import Path

import yaml

PROJ = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
ROOT = Path("/root/rivermind-data/uav_data_v2")
sys.path.insert(0, str(PROJ))
import os

os.chdir(PROJ)

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.mm.utils import get_label_path_from_data, get_modality_path_from_data, resolve_modality_pair  # noqa: E402

cfg = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
print("=== data.yaml labels 映射 ===")
print(" ", cfg.get("labels"))
print("   modality:", cfg.get("modality"))

print("\n=== resolve_modality_pair / 各目录解析 ===")
primary, secondary = resolve_modality_pair(cfg, default_secondary="ir")
print(f"  primary={primary!r}  secondary={secondary!r}")
for tok in ("rgb", "ir"):
    try:
        print(f"  get_label_path_from_data({tok!r})    = {get_label_path_from_data(cfg, tok)!r}")
        print(f"  get_modality_path_from_data({tok!r}) = {get_modality_path_from_data(cfg, tok)!r}")
    except Exception as e:
        print(f"  {tok}: {type(e).__name__}: {e}")

print("\n=== 实际构建 dataset（val 集，primary 图像目录 images/val） ===")
from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.data.dataset import YOLOMultiModalImageDataset  # noqa: E402

hyp = get_cfg()
for k, v in (("mosaic", 0.0), ("mixup", 0.0), ("cutmix", 0.0), ("copy_paste", 0.0)):
    setattr(hyp, k, v)
cfg2 = dict(cfg)
for k in ("cache", "cache_ir"):
    cfg2.pop(k, None)

for tag, img_dir, xm in (
    ("A: img_path=images/val, x=ir", ROOT / "images" / "val", "ir"),
    ("B: img_path=images_ir/val, x=ir", ROOT / "images_ir" / "val", "ir"),
):
    try:
        ds = YOLOMultiModalImageDataset(
            img_path=str(img_dir), data=cfg2, x_modality=xm, imgsz=640,
            augment=False, hyp=hyp, rect=False, cache=False,
        )
        print(f"  [{tag}]")
        print(f"    primary_modality_dir = {ds.primary_modality_dir!r}")
        print(f"    primary_label_dir    = {ds.primary_label_dir!r}")
        print(f"    secondary_label_dir  = {ds.secondary_label_dir!r}")
        print(f"    dual_labels_configured = {ds.dual_labels_configured}")
        if ds.im_files:
            f0 = ds.im_files[0]
            print(f"    im_files[0] = {f0}")
            print(f"    labels[0].im_file = {ds.labels[0].get('im_file')}")
            print(f"    派生主标签路径 = {ds._img2label_path(f0)}")
    except Exception as e:
        print(f"  [{tag}] 构建失败: {type(e).__name__}: {e}")
