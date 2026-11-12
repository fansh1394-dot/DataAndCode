# -*- coding: utf-8 -*-
"""构建 bd_nc3_unreg：RGB/标签与 bd_nc3 相同，AVID 基础帧的 IR 换成未配准原图。

用途：E30 IR 配准消融。合成帧的 IR 本来就是灰度复制，Mendeley IR 本身像素对齐，
      因此只需要把 avid__ 基础帧的 IR 指回 AVID 原始 images/ir。
"""
import json
import os
from pathlib import Path

RAW = Path("/root/rivermind-fs/birddrone")
SRC, DST = RAW / "bd_nc3", RAW / "bd_nc3_unreg"
AVID = Path("/root/rivermind-fs/AVID")


def link(src, dst):
    if not dst.exists():
        os.symlink(src, dst)


there = 0
for split in ("train", "val"):
    for sub in ("images", "images_ir", "labels", "labels_ir"):
        (DST / sub / split).mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels", "labels_ir"):
        for p in (SRC / sub / split).iterdir():
            link(p, DST / sub / split / p.name)
    for p in (SRC / "images_ir" / split).iterdir():
        stem = p.stem
        dst = DST / "images_ir" / split / p.name
        if "__syn" in stem or stem.startswith("men__"):
            link(p, dst)
            continue
        parts = stem.split("__")
        video, frame = parts[1], parts[2]
        cand = AVID / "images/ir/train" / video / f"{frame}.jpg"
        if not cand.exists():
            cand = AVID / "images/ir/val" / video / f"{frame}.jpg"
        if cand.exists():
            link(cand, dst)
            there += 1
        else:
            link(p, dst)
    for f in ("train_nc3.txt", "val_nc3.txt"):
        (DST / f).write_text((SRC / f).read_text(), encoding="utf-8")

s = (SRC / "data_nc3.yaml").read_text(encoding="utf-8").replace(str(SRC), str(DST))
(DST / "data_unreg.yaml").write_text(s, encoding="utf-8")
print("unregistered IR frames", there, "->", DST)
