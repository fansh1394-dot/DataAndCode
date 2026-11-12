# -*- coding: utf-8 -*-
"""验证绘图修复：X(IR) 面板是否改用 X 自己的标注集。

用真实的一帧 both 数据合成 batch，分别以「旧行为」（不传 bboxes_x）与
「新行为」（传 bboxes_x=labels_ir）各画一次，输出到两个目录，
对比 train_batch0_labels_ir.jpg 中框是否落在无人机上。

纯 CPU，不占 GPU。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
ROOT = Path("/root/rivermind-data/uav_data_v2")
sys.path.insert(0, str(PROJ))
import os

os.chdir(PROJ)

from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def find_img(d: Path, stem: str):
    for e in EXT:
        p = d / f"{stem}{e}"
        if p.exists():
            return p
    return None


def read_last_box(p: Path):
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        s = line.split()
        if len(s) >= 5:
            _, cx, cy, w, h = (float(v) for v in s[:5])
            return (cx, cy, w, h)
    return None


def main() -> int:
    meta = json.loads((ROOT / "metadata" / "val.json").read_text(encoding="utf-8"))["frames"]
    stems = [s for s, i in meta.items() if i.get("valid_rgb") and i.get("valid_ir")]
    print(f"both 帧数 = {len(stems)}")

    picked = None
    for s in stems:
        ip = find_img(ROOT / "images_ir" / "val", s)
        rp = find_img(ROOT / "images" / "val", s)
        if ip is None or rp is None:
            continue
        ib = read_last_box(ROOT / "labels_ir" / "val" / f"{s}.txt")
        rb = read_last_box(ROOT / "labels" / "val" / f"{s}.txt")
        if ib and rb:
            picked = (s, ip, rp, ib, rb)
            break
    if picked is None:
        print("[错误] 未找到可用的 both 帧")
        return 1
    stem, ip, rp, ir_box, rgb_box = picked
    print(f"使用帧 {stem}")
    print(f"  IR 框(labels_ir) = {ir_box}")
    print(f"  RGB 框(labels)   = {rgb_box}")
    print(f"  两者中心差  dy={ir_box[1] - rgb_box[1]:+.4f}  dx={ir_box[0] - rgb_box[0]:+.4f}")

    rgb = cv2.imread(str(rp))
    ir = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.resize(rgb, (640, 640))[:, :, ::-1]           # BGR -> RGB
    ir = cv2.resize(ir, (640, 640))
    ir3 = np.stack([ir, ir, ir], axis=2)
    img = np.concatenate([rgb, ir3], axis=2).transpose(2, 0, 1)  # [6,640,640]
    timg = torch.from_numpy(img.copy()).float().unsqueeze(0) / 255.0

    logi = torch.tensor([ir_box], dtype=torch.float32)
    logr = torch.tensor([rgb_box], dtype=torch.float32)
    bi = torch.tensor([0], dtype=torch.long)
    cls = torch.tensor([0], dtype=torch.long)

    xname = "ir"
    common = dict(
        stage="train", source="labels", images=timg,
        batch_idx=bi, cls=cls, xch=3, x_modality_name=xname,
        precision_contract={"rgb": "legacy", "x": "legacy"},
        modality_mode="dual",
        paths=[str(rp)], names={0: "drone"}, batch_ni=0,
    )

    out_old = PROJ / "_d2p1" / "plot_verify" / "old"
    out_new = PROJ / "_d2p1" / "plot_verify" / "new"
    for d in (out_old, out_new):
        d.mkdir(parents=True, exist_ok=True)

    p1 = MultiModalPlotter(out_old, on_plot=None)
    p1.plot_batch(bboxes=logr, **common)
    print(f"[旧] 已写入 {out_old}")

    p2 = MultiModalPlotter(out_new, on_plot=None)
    p2.plot_batch(bboxes=logr, bboxes_x=logi, cls_x=cls, batch_idx_x=bi, **common)
    print(f"[新] 已写入 {out_new}")

    for d in (out_old, out_new):
        fs = sorted(p.name for p in d.iterdir()) if d.exists() else []
        print(f"  {d.name}: {fs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
