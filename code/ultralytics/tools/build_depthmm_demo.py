#!/usr/bin/env python3
"""Build a small RGB+IR+gt_depth demo set under /datasets/depthmm from M3FD pairs."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC_ROOT = Path("/root/datasets/M3FD_split")
DST_ROOT = Path("/datasets/depthmm")
N_TRAIN = 32
N_VAL = 8
DEPTH_SCALE = 1000.0
MIN_M, MAX_M = 1.0, 25.0


def _stems(split: str) -> list[str]:
    rgb_dir = SRC_ROOT / "images" / split
    ir_dir = SRC_ROOT / "images_ir" / split
    names = []
    for p in sorted(rgb_dir.glob("*.png")):
        if (ir_dir / p.name).is_file():
            names.append(p.name)
    if not names:
        raise FileNotFoundError(f"No RGB+IR pairs in {rgb_dir}")
    return names


def _metric_depth(rgb_bgr: np.ndarray) -> np.ndarray:
    """uint16 meters*1000. Image structure + ground-near vertical prior."""
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    # Invert luma (darker -> farther) and pull the bottom of the frame closer.
    rel = 0.65 * (1.0 - gray) + 0.35 * (1.0 - yy)
    rel = cv2.GaussianBlur(rel, (0, 0), 3)
    rel = (rel - rel.min()) / (float(rel.max() - rel.min()) + 1e-6)
    meters = MIN_M + rel * (MAX_M - MIN_M)
    return np.clip(meters * DEPTH_SCALE, 1, 65535).astype(np.uint16)


def _save_u16(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _copy_split(split: str, names: list[str]) -> None:
    for name in names:
        stem = Path(name).stem
        rgb_src = SRC_ROOT / "images" / split / name
        ir_src = SRC_ROOT / "images_ir" / split / name
        rgb = cv2.imread(str(rgb_src), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_src)
        shutil.copy2(rgb_src, DST_ROOT / "images" / split / name)
        shutil.copy2(ir_src, DST_ROOT / "images_ir" / split / name)
        _save_u16(DST_ROOT / "gt_depth" / split / f"{stem}.png", _metric_depth(rgb))


def _write_yaml() -> None:
    text = f"""# YOLOMM task=depth demo set (M3FD RGB+IR subset)
# RGB/IR copied from /root/datasets/M3FD_split; gt_depth is uint16 meters (scale={int(DEPTH_SCALE)}).
path: {DST_ROOT}
train: images/train
val: images/val
gt_depth: gt_depth
modality:
  rgb: images
  ir: images_ir
modality_used: ["rgb", "ir"]
Xch: 3
nc: 1
names:
  0: depth
min_depth: 0.001
max_depth: 100
depth_scale: {int(DEPTH_SCALE)}
"""
    (DST_ROOT / "data.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    for split in ("train", "val"):
        for sub in ("images", "images_ir", "gt_depth"):
            (DST_ROOT / sub / split).mkdir(parents=True, exist_ok=True)

    train_names = _stems("train")[:N_TRAIN]
    val_names = _stems("val")[:N_VAL]
    if len(train_names) < N_TRAIN or len(val_names) < N_VAL:
        raise RuntimeError(f"Not enough pairs: train={len(train_names)} val={len(val_names)}")
    _copy_split("train", train_names)
    _copy_split("val", val_names)
    _write_yaml()
    print(f"Wrote {DST_ROOT} train={len(train_names)} val={len(val_names)}")


if __name__ == "__main__":
    main()
