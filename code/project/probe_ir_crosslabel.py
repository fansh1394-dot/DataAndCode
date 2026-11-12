# -*- coding: utf-8 -*-
"""判别实验：IR 头的 0.64 是「真 IR 能力」，还是「标签口径」造成的假象？

动机
----
单模态 IR 基线 = 0.207 / 0.211（真 IR 图、真 IR 特征）
双头 C4 的 ir 头 = 0.6415

输入路由已实测正确（IR 支路吃到的是 images_ir 段，max|Δ|=0），所以差异只能来自
监督/评测的**标签口径**：
  * IR-only 基线用 `--label-from dual` -> 训练与评测都拿 `labels`(RGB 坐标系)
  * C4 的 ir 头用 `bboxes_ir`        -> IR 坐标系

本脚本在**同一份数据、同一次前向**上做交叉口径评测，直接量化「标签口径」的代价：

  C4 模型（dual）：
      ir   vs labels_ir  （标称口径，应≈0.6415）
      ir   vs labels     （交叉口径）
      rgb  vs labels     （标称，应≈0.7052）
      rgb  vs labels_ir  （交叉）
      fuse vs labels / labels_ir
  单模态 IR 模型（只出 "fuse" 分支）：
      fuse vs labels     （= 其训练口径，应≈0.2074）
      fuse vs labels_ir  （交叉口径）

用法：
  python probe_ir_crosslabel.py --weights <best.pt> --data <data.yaml> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

import torch
import yaml


def find_root() -> Path:
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (p / "train_full.py").exists() and (p / "ultralytics").is_dir():
            return p
    return Path("/3250410011/root/rivermind-data/MutilModel_3485988898")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 batch（冒烟用）")
    args = ap.parse_args()

    root = find_root()
    sys.path.insert(0, str(root))
    os.chdir(root)

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLOMultiModalImageDataset
    from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator
    from ultralytics.utils.metrics import DetMetrics
    from ultralytics.utils.patches import torch_load

    # ---------------- 交叉口径评测器 ----------------
    CROSS_SPECS = {
        # 被评分支 -> [(结果键, box_key, cls_key, valid_key)]
        "ir":   [("ir@rgblabel", "bboxes", "cls", "valid_rgb")],
        "rgb":  [("rgb@irlabel", "bboxes_ir", "cls_ir", "valid_ir")],
        "fuse": [("fuse@irlabel", "bboxes_ir", "cls_ir", "valid_ir")],
    }

    class ProbeVal(DualHeadDetectionValidator):
        """在标准分支指标之外，额外按「另一模态标签」再评一次。"""

        def init_metrics(self, model) -> None:
            super().init_metrics(model)
            self.extra_metrics: dict = {}
            self._dbg_once: dict = {}
            for specs in CROSS_SPECS.values():
                for key, *_ in specs:
                    if key not in self.extra_metrics:
                        m = DetMetrics()
                        m.names = self.names
                        self.extra_metrics[key] = m

        def _update_branch_metrics(self, name, preds, batch) -> None:
            super()._update_branch_metrics(name, preds, batch)
            for key, box_key, cls_key, valid_key in CROSS_SPECS.get(name, []):
                try:
                    src = self._swap_labels(batch, box_key, cls_key)
                    have = {k: (tuple(batch[k].shape) if hasattr(batch.get(k), "shape") else type(batch.get(k)).__name__)
                            for k in ("bboxes", "cls", "batch_idx", "bboxes_ir", "cls_ir", "batch_idx_ir",
                                      "valid_rgb", "valid_ir")}
                    dbg = f"[dbg {key}] batch 键状态={have} swapped_boxes={tuple(src['bboxes'].shape)}"
                    p, b = self._select_valid_samples(src, preds, valid_key)
                    dbg += f" -> preds={0 if not p else len(p)} batch={'None' if b is None else 'ok'}"
                    if b is not None and p:
                        self._accumulate(self.extra_metrics[key], p, b)
                        dbg += f" 已累加(nt={len(self.extra_metrics[key].stats.get('target_cls', []))})"
                    if not self._dbg_once.get(key):
                        self._dbg_once[key] = True
                        print(dbg)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] 交叉口径 {key} 失败: {type(e).__name__}: {e}")

    # ---------------- 数据 ----------------
    data_path = Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    droot = Path(cfg.get("path") or data_path.parent)
    val_sub = str(cfg.get("val", "images/val")).split("/")[-1]
    print(f"[data] labels 映射 = {cfg.get('labels')}")

    hyp = get_cfg()
    for k, v in (("mosaic", 0.0), ("mixup", 0.0), ("cutmix", 0.0), ("copy_paste", 0.0),
                 ("mm_rgb_albu_enable", False), ("mm_x_non_geo_enable", False)):
        setattr(hyp, k, v)
    for k in ("cache", "cache_ir"):
        cfg.pop(k, None)
    ds = YOLOMultiModalImageDataset(
        img_path=str(droot / "images" / val_sub), data=cfg, x_modality="ir",
        imgsz=args.imgsz, augment=False, hyp=hyp, rect=False, cache=False,
    )
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)

    # ---------------- 模型 ----------------
    ow = getattr(pathlib, "WindowsPath", None)
    if hasattr(pathlib, "WindowsPath"):
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        ck = torch_load(args.weights, map_location="cpu", weights_only=False)
    finally:
        if ow is not None:
            pathlib.WindowsPath = ow
    model = (ck.get("ema") or ck.get("model")).float().eval()
    print(f"[model] {args.weights}  层数={len(model.model)}  末层={type(model.model[-1]).__name__}")
    print(f"[model] single_modal_build = {ck.get('single_modal_build')}")

    vargs = get_cfg()
    vargs.data = str(data_path)
    vargs.imgsz = args.imgsz
    vargs.batch = args.batch
    vargs.plots = False
    vargs.half = False
    vargs.workers = args.workers
    vargs.device = args.device
    vargs.model = str(args.weights)

    from ultralytics.utils.torch_utils import select_device

    v = ProbeVal(dataloader=loader, save_dir=Path("/tmp/probe_ir_crosslabel"), args=vargs)
    v.device = select_device(args.device)
    v.data = cfg
    # 非训练路径：loss 与列名置空，避免 get_stats 因缺项 KeyError
    v.training = False
    v.loss = torch.zeros(1)
    v._val_loss_names = ()
    if args.limit:
        class _Trunc:
            def __init__(self, it, n):
                self.it, self.n = it, n
            def __iter__(self):
                for i, b in enumerate(self.it):
                    if i >= self.n:
                        break
                    yield b
            def __len__(self):
                return min(self.n, len(self.it))
        v.dataloader = _Trunc(v.dataloader, args.limit)

    model = model.to(v.device)
    v(model=model)

    print("\n================ 结果 ================")
    res = {}
    for name, m in v.branch_metrics.items():
        key = f"{name}|标称口径"
        try:
            m.process(save_dir=v.save_dir, plot=False, on_plot=v.on_plot)
        except Exception:
            pass
        res[key] = float(m.box.map)
        print(f"  {key:<20} mAP50-95 = {m.box.map:.4f}   mAP50 = {m.box.map50:.4f}  nt={len(m.stats.get('target_cls', []))}")
    for key, m in v.extra_metrics.items():
        try:
            m.process(save_dir=v.save_dir, plot=False, on_plot=v.on_plot)
        except Exception:
            pass
        res[key] = float(m.box.map)
        print(f"  {key:<20} mAP50-95 = {m.box.map:.4f}   mAP50 = {m.box.map50:.4f}  nt={len(m.stats.get('target_cls', []))}")

    out = Path("/tmp/probe_ir_crosslabel_result.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[已写入] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
