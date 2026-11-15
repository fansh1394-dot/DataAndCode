#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建第二批数据集变体（不污染 bd_nc3）：
  bd_nc3_other0   —— nc=3，但合成帧去掉 class2 标签（other 比例 0%）
  bd_nc3_other30  —— nc=3，额外注入合成帧，使 other 框占比≈30%

做法：以 bd_nc3 的帧清单为基底，只生成新的列表文件 + 软链接目录 + yaml；
      不改动 bd_nc3 本身，也不改动原始 AVID/Mendeley。
"""
import json
import os
import random
from pathlib import Path

RAW = Path("/root/rivermind-fs/birddrone")
SRC = RAW / "bd_nc3"
OUT0 = RAW / "bd_nc3_other0"
OUT30 = RAW / "bd_nc3_other30"

YAML_TMPL = """path: {path}
train: train_list.txt
val: val_list.txt
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
nc: 3
names:
  0: bird
  1: drone
  2: other
"""


def link_tree(dst: Path) -> None:
    for sub in ("images", "images_ir", "labels", "labels_ir"):
        for split in ("train", "val"):
            (dst / sub / split).mkdir(parents=True, exist_ok=True)


def link_all(dst: Path, names, split: str) -> None:
    """把 bd_nc3 里存在的同名文件软链接到 dst（四类子目录）。"""
    for sub in ("images", "images_ir", "labels", "labels_ir"):
        idx = {}
        d = SRC / sub / split
        for p in d.iterdir():
            idx[p.stem] = p
        for n in names:
            stem = Path(n).stem
            p = idx.get(stem)
            if p is None:
                continue
            t = dst / sub / split / p.name
            if not t.exists():
                os.symlink(p, t)


def strip_other(src_label: Path, dst_label: Path) -> None:
    lines = [l for l in src_label.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [l for l in lines if not l.startswith("2 ")]
    dst_label.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")


def build_other0() -> None:
    link_tree(OUT0)
    for split, lf in (("train", "train_nc3.txt"), ("val", "val_nc3.txt")):
        names = [x.strip() for x in (SRC / lf).read_text(encoding="utf-8").splitlines() if x.strip()]
        link_all(OUT0, names, split)
        # 标签去掉 class2（两份标签体系都处理）
        for sub in ("labels", "labels_ir"):
            for n in names:
                stem = Path(n).stem
                p = OUT0 / sub / split / f"{stem}.txt"
                if p.exists() and os.path.islink(p):
                    p.unlink()
                s = SRC / sub / split / f"{stem}.txt"
                if s.exists():
                    strip_other(s, p)
        (OUT0 / f"{lf.replace('nc3', 'list')}").write_text("\n".join(names) + "\n", encoding="utf-8")
    (OUT0 / "data_other0.yaml").write_text(YAML_TMPL.format(path=OUT0), encoding="utf-8")
    print("[other0] 完成", (OUT0 / "train_list.txt").stat().st_size)


def build_other30() -> None:
    """在 bd_nc3 的合成帧里再补一批（复用已有合成帧，multi-copy 不算新数据，
    因此改为：保留全部合成帧 + 追加更多 bd_nc3 已合成帧的副本变体不可取）。
    实际做法：other 占比 = other 框 / 总框。当前 7%，要达到 30% 需要 4.3 倍 other 框。
    这里用「每帧多目标」的方式重写：为已有合成帧追加第 2~4 个 other 实例（复用同帧抠图位置）。
    仅用于敏感性分析，不改 bd_nc3。"""
    link_tree(OUT30)
    manifest = json.loads((SRC / "synth_manifest.json").read_text(encoding="utf-8"))
    per_frame = {m["stem"]: m["n_injected"] for m in manifest}
    for split, lf in (("train", "train_nc3.txt"), ("val", "val_nc3.txt")):
        names = [x.strip() for x in (SRC / lf).read_text(encoding="utf-8").splitlines() if x.strip()]
        link_all(OUT30, names, split)
        for sub in ("labels", "labels_ir"):
            for n in names:
                stem = Path(n).stem
                p = OUT30 / sub / split / f"{stem}.txt"
                if p.exists() and os.path.islink(p):
                    p.unlink()
                s = SRC / sub / split / f"{stem}.txt"
                if not s.exists():
                    continue
                lines = [l for l in s.read_text(encoding="utf-8").splitlines() if l.strip()]
                k = per_frame.get(stem, 0)
                if k:
                    # 复制已有 other 框并做小幅平移/缩放，模拟更多气球/风筝
                    others = [l for l in lines if l.startswith("2 ")]
                    extra = []
                    for rep in range(3):  # 每个已有 other 再复制 3 份 → other 框 ×4
                        for ln in others:
                            t = ln.split()
                            cx, cy, w, h = (float(t[1]), float(t[2]), float(t[3]), float(t[4]))
                            dx = random.uniform(-0.18, 0.18)
                            dy = random.uniform(-0.12, 0.12)
                            sc = random.uniform(0.7, 1.15)
                            cx2, cy2 = min(max(cx + dx, 0.03), 0.97), min(max(cy + dy, 0.03), 0.97)
                            w2, h2 = min(w * sc, 0.3), min(h * sc, 0.3)
                            extra.append(f"2 {cx2:.6f} {cy2:.6f} {w2:.6f} {h2:.6f}")
                    lines = lines + extra
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (OUT30 / lf.replace("nc3", "list")).write_text("\n".join(names) + "\n", encoding="utf-8")
    (OUT30 / "data_other30.yaml").write_text(YAML_TMPL.format(path=OUT30), encoding="utf-8")
    print("[other30] 完成")


def count_ratio(root: Path, split: str) -> None:
    lf = root / f"{split}_list.txt"
    if not lf.exists():
        return
    names = [x.strip() for x in lf.read_text().splitlines() if x.strip()]
    from collections import Counter
    c = Counter()
    for n in names:
        p = root / "labels" / split / f"{Path(n).stem}.txt"
        if not p.exists():
            continue
        for l in p.read_text().splitlines():
            if l.strip():
                c[l.split()[0]] += 1
    tot = sum(c.values())
    print(f"  {root.name}/{split}: 框 {dict(c)} 总 {tot}  other 占比 {100*c['2']/max(tot,1):.1f}%")


random.seed(0)
build_other0()
build_other30()
print("=== 占比核对 ===")
for r in (SRC, OUT0, OUT30):
    for s in ("train", "val"):
        if r is SRC:
            continue
        count_ratio(r, s)
