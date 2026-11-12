# -*- coding: utf-8 -*-
"""把 AVID（images/<split>/<video>/<frame>）摊平成加载器要求的扁平布局。

原因：ultralytics/data/dataset.py 的 _find_corresponding_x_image 用
    dataset_root = rgb_path.parent.parent.parent
推导 X 模态根目录，只兼容 images/<split>/<file> 两级结构。
AVID 多一层 <video>，会把根目录定位到 images/ 下，拼出 images/images_ir/...。

做法：不复制像素，只建符号链接：
    avid_flat/images/val/<video>__<frame>.jpg      -> AVID/images/rgb/val/<video>/<frame>.jpg
    avid_flat/images_ir/val/<video>__<frame>.jpg   -> AVID/images/ir/val/<video>/<frame>.jpg
    avid_flat/labels/val/<video>__<frame>.txt      -> AVID/labels/val/<video>/<frame>.txt
    avid_flat/labels_ir/val/...                    -> 同上（AVID 只有一套 RGB 坐标标签）
同时重写鸟 val 列表为扁平相对路径。
"""
import os
from pathlib import Path

ROOT = Path("/root/rivermind-fs/birddrone")
AVID = Path("/root/rivermind-fs/AVID")
FLAT = ROOT / "avid_flat"

# 目录名 -> AVID 源目录
DIRS = {
    "images": AVID / "images/rgb",
    "images_ir": AVID / "images/ir",
    "labels": AVID / "labels",
    "labels_ir": AVID / "labels",
}
SPLITS = ("train", "val")

total = 0
for name, src_root in DIRS.items():
    for split in SPLITS:
        src = src_root / split
        dst = FLAT / name / split
        dst.mkdir(parents=True, exist_ok=True)
        for video in sorted(p for p in src.iterdir() if p.is_dir()):
            for f in sorted(video.iterdir()):
                if not f.is_file():
                    continue
                link = dst / f"{video.name}__{f.name}"
                if link.is_symlink() or link.exists():
                    continue
                os.symlink(f, link)
                total += 1
    print(f"[flat] {name}: ok")

# 重写鸟 val 列表：AVID 绝对路径 -> avid_flat 下扁平相对路径
old = [x.strip() for x in (ROOT / "avid_bird_val.txt").read_text().splitlines() if x.strip()]
new = []
for x in old:
    p = Path(x)
    new.append(f"images/val/{p.parent.name}__{p.name}")
(ROOT / "avid_bird_val_flat.txt").write_text("\n".join(new) + "\n", encoding="utf-8")

yaml_text = f"""path: {FLAT}
train: images/train
val: images/val
test: images/val
modality:
  rgb: images
  ir: images_ir
modality_used:
- rgb
- ir
labels:
  rgb: labels
  ir: labels_ir
pair_policy: any
align: false
channels: 3
Xch: 3
nc: 1
names:
  0: UAV
"""
(ROOT / "data_avid_nc1_flat.yaml").write_text(yaml_text, encoding="utf-8")

print("links_created", total)
print("val_list", len(new), new[0])
print("sample_exists", (FLAT / ("images/val/" + Path(new[0]).name)).exists())
print("sample_x_exists", (FLAT / ("images_ir/val/" + Path(new[0]).name)).exists())
