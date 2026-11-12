# -*- coding: utf-8 -*-
"""E28：构建 nc=2（0=bird / 1=drone）双光谱数据集 bd_nc2。

对齐口径（重要）
----------------
- Mendeley（mendeley_split）：RGB/IR 像素对齐，labels 与 labels_ir 各自独立标注；
  原类别 0=UAV，一律 remap 为 1=drone。
- AVID：RGB/IR 仅「帧级配对」而非像素配准（论文 32.3% 全局位移 >5 px），
  且只有一套 RGB 坐标标签。因此 **不做逐帧 warp**（部署端也不做配准），
  原样接入，靠训练期的 IR 通路口径处理；
  逐帧配准版本留给 E30「配准 vs 不配准」消融，不与本数据集混用。
- 所有文件扁平化，文件名前缀区分来源：avid__ / men__。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

AVID = Path("/root/rivermind-fs/AVID")
AVID_IR_REG = Path("/root/rivermind-fs/birddrone/avid_ir_reg")  # 已配准 IR
MEN = Path("/3250410011/root/rivermind-data/mendeley_split")
OUT = Path("/root/rivermind-fs/birddrone/bd_nc2")
SPLIT_JSON = Path("/root/rivermind-fs/birddrone/split.json")

SPLITS = ("train", "val", "test")


def link(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(src, dst)


def remap_label(src: Path, dst: Path, delta: int = 1):
    """类别 id +delta（Mendeley 0=UAV -> 1=drone）。"""
    if dst.exists():
        return
    lines = []
    for ln in src.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            p[0] = str(int(float(p[0])) + delta)
            lines.append(" ".join(p))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    cfg = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    avid_splits = cfg["sources"]["avid"]["splits"]

    for split in SPLITS:
        for sub in ("images", "images_ir", "labels", "labels_ir"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in SPLITS:
        # 允许重建：清掉旧的 AVID IR 链接，改用最新配准结果
        for f in (OUT / "images_ir" / split).glob("avid__*"):
            if f.is_symlink():
                f.unlink()
        # AVID：test 用官方 Val；train/val 都从官方 Train 内按视频切分
        avid_src_split = "val" if split == "test" else "train"
        n_avid = 0
        n_avid_reg = 0
        for v in avid_splits[split]:
            video = v["video"]
            rgb_dir = AVID / "images/rgb" / avid_src_split / video
            ir_dir = AVID / "images/ir" / avid_src_split / video
            lb_dir = AVID / "labels" / avid_src_split / video
            for img in sorted(rgb_dir.iterdir()):
                if not img.is_file():
                    continue
                stem = f"avid__{video}__{img.stem}"
                link(img, OUT / "images" / split / f"{stem}{img.suffix}")
                ir_reg = AVID_IR_REG / avid_src_split / f"{video}__{img.stem}.jpg"
                ir_src = ir_reg if ir_reg.exists() else (ir_dir / img.name)
                n_avid_reg += int(ir_reg.exists())
                link(ir_src, OUT / "images_ir" / split / f"{stem}{img.suffix}")
                lb = lb_dir / f"{img.stem}.txt"
                if lb.exists():
                    link(lb, OUT / "labels" / split / f"{stem}.txt")
                    link(lb, OUT / "labels_ir" / split / f"{stem}.txt")
                n_avid += 1

        n_men = 0
        for img in sorted((MEN / "images" / split).iterdir()):
            if not img.is_file():
                continue
            stem = f"men__{img.stem}"
            link(img, OUT / "images" / split / f"{stem}{img.suffix}")
            ir = MEN / "images_ir" / split / img.name
            if ir.exists():
                link(ir, OUT / "images_ir" / split / f"{stem}{ir.suffix}")
            lb = MEN / "labels" / split / f"{img.stem}.txt"
            if lb.exists():
                remap_label(lb, OUT / "labels" / split / f"{stem}.txt")
            lbi = MEN / "labels_ir" / split / f"{img.stem}.txt"
            if lbi.exists():
                remap_label(lbi, OUT / "labels_ir" / split / f"{stem}.txt")
            n_men += 1

        # 校验：图像 / IR / 标签 数量一致，且不含原 0=UAV 的漏改（仅 Mendeley 侧）
        n_img = len(list((OUT / "images" / split).iterdir()))
        n_ir = len(list((OUT / "images_ir" / split).iterdir()))
        n_lb = len(list((OUT / "labels" / split).iterdir()))
        n_lbi = len(list((OUT / "labels_ir" / split).iterdir()))
        cls = Counter()
        for f in (OUT / "labels" / split).iterdir():
            for ln in f.read_text().splitlines():
                p = ln.split()
                if p:
                    cls[int(float(p[0]))] += 1
        stats[split] = {
            "avid_frames": n_avid,
            "avid_ir_registered": n_avid_reg,
            "mendeley_frames": n_men,
            "images": n_img,
            "images_ir": n_ir,
            "labels": n_lb,
            "labels_ir": n_lbi,
            "class_counts": dict(cls),
        }
        print(split, stats[split])

    yaml_text = f"""path: {OUT}
train: images/train
val: images/val
test: images/test
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
nc: 2
names:
  0: bird
  1: drone
"""
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (OUT / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("data.yaml ->", OUT / "data.yaml")


if __name__ == "__main__":
    main()
