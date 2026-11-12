# -*- coding: utf-8 -*-
"""验证：开启几何增强后，RGB/IR 两路的图像与 bbox 是否仍严格对应。

为什么必须验证：
  数据是 align:true（逐像素对齐），几何增强走的是原生 RandomPerspective。
  代码上已确认它会用同一个仿射矩阵 M 同时变换 6 通道图和 instances_ir，
  但「代码看起来对」不等于「实际对」。本脚本直接从 dataloader 取真实 batch，
  数值核对：
    1) 6 通道图确实被 warp（不是原图）
    2) instances 与 instances_ir 的框数量变化一致
    3) 两路框落在同一图像坐标系的同一区域（center 接近）

判据：若对齐被破坏，两路框中心会系统性偏移，或框数出现不合理差异。
"""
import os
import sys

import numpy as np

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import torch  # noqa: E402
import yaml  # noqa: E402

from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer  # noqa: E402


def build(geo_on: bool):
    """直接构造多模态训练 dataset，返回一个 batch。

    不走 Trainer：Trainer 的完整初始化会触发模型加载、EMA、AMP 等一堆无关逻辑，
    这里只需要 dataset 的变换链，直接调 build_yolo_dataset 最干净。
    """
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    data = check_det_dataset("/3250410011/root/rivermind-data/mendeley_split/data.yaml")
    raw = yaml.safe_load(open(
        "/3250410011/root/rivermind-data/mendeley_split/data.yaml", encoding="utf-8"))
    for k, v in raw.items():
        data.setdefault(k, v)

    # 以 default.yaml 为基底，避免漏字段（mask_ratio/rect/overlap_mask 等都是
    # dataset 变换链要读的）。只覆盖我们关心的增强项。
    hyp = YAML.load(check_yaml("default.yaml"))
    hyp.update({
        "degrees": 10.0 if geo_on else 0.0,
        "shear": 5.0 if geo_on else 0.0,
        "perspective": 0.0005 if geo_on else 0.0,
        "translate": 0.1, "scale": 0.5,
        "fliplr": 0.5, "flipud": 0.0,
        "mosaic": 0.0,        # 关掉 mosaic，隔离几何变换本身的效果
        "mixup": 0.0, "copy_paste": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,   # 关掉光度，只看几何
        "erasing": 0.0, "auto_augment": None,
        "use_multimodal_aug": True,
        "mm_async_geo_p": 0.0,
        "mm_rgb_albu_enable": False, "mm_x_non_geo_enable": False,
        "imgsz": 640, "cache": False, "single_cls": False,
        "rect": False, "task": "detect", "fraction": 1.0,
        "overlap_mask": False, "classes": None, "data": None,
    })
    args = IterableSimpleNamespace(**hyp)

    ds = build_yolo_dataset(
        args, data["train"], batch=4, data=data, mode="train", rect=False, stride=32,
        multi_modal_image=True, x_modality="ir",
        x_modality_dir=data.get("modality", {}).get("ir", "images_ir"),
        modalities=data.get("modality_used", ["rgb", "ir"]),
    )
    ds.augment = True
    loader = build_dataloader(ds, batch=4, workers=0, shuffle=False, rank=-1)
    return next(iter(loader))


def summarize(tag, batch):
    img = batch["img"]
    print(f"\n[{tag}]")
    print(f"  img shape={tuple(img.shape)}  mean={img.float().mean():.3f} "
          f"std={img.float().std():.3f}")
    print(f"  batch keys={sorted(batch.keys())}")

    # Format 变换后，batch 里是 cls/bboxes/batch_idx，不再是 Instances 对象。
    # 双标签模式下 IR 侧对应为 bboxes_ir 等，或存在独立 instances_ir。
    b = batch.get("bboxes")
    bi = batch.get("bboxes_ir")
    if b is None:
        print("  无 bboxes，跳过框分析")
        return
    print(f"  RGB 框数={len(b)}  IR 框数={len(bi) if bi is not None else 'N/A'}")
    if bi is not None and len(b) > 0 and len(bi) > 0:
        bn = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
        bin_ = bi.numpy() if hasattr(bi, "numpy") else np.asarray(bi)
        cbx, cby = (bn[:, 0] + bn[:, 2]) / 2, (bn[:, 1] + bn[:, 3]) / 2
        cix, ciy = (bin_[:, 0] + bin_[:, 2]) / 2, (bin_[:, 1] + bin_[:, 3]) / 2
        print(f"  RGB 框中心均值=({cbx.mean():.1f}, {cby.mean():.1f})")
        print(f"  IR  框中心均值=({cix.mean():.1f}, {ciy.mean():.1f})")
        print(f"  中心偏差 |Δx|={abs(cbx.mean() - cix.mean()):.1f} "
              f"|Δy|={abs(cby.mean() - ciy.mean()):.1f}")
        # 更强的判据：按实例逐一比较（两路框应是同一批目标）
        if len(bn) == len(bin_):
            d = np.abs(bn - bin_).max(axis=1)
            print(f"  逐框最大坐标差: 均值={d.mean():.2f} 最大={d.max():.2f} 像素")
            print(f"  判据：最大差 < 2px -> 对齐良好；> 10px -> 对齐被破坏")


if __name__ == "__main__":
    print("=== 几何增强关闭（基线）===")
    b0 = build(False)
    summarize("geo OFF", b0)

    print("\n=== 几何增强开启（degrees=10, shear=5, perspective=5e-4）===")
    b1 = build(True)
    summarize("geo ON", b1)
