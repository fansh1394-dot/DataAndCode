# -*- coding: utf-8 -*-
"""E28b：构造平衡训练/选模列表。

问题：E28 的 train 里 AVID 占 83%，导致 Mendeley UAV 精度掉 10 pp。
做法：不改动已有数据，只生成抽样列表：
  - train_bal.txt = Mendeley train 全量(4 604) + AVID train 分层抽样(4 604)
    （按视频轮转抽样，保证 82 段视频都出现，且不按帧随机打乱类别）
  - val_bal.txt   = Mendeley val 616 + AVID val 抽样 616（选模用，避免 AVID 主导）
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path("/root/rivermind-fs/birddrone/bd_nc2")
SPLIT = json.loads(Path("/root/rivermind-fs/birddrone/split.json").read_text())["sources"]["avid"]["splits"]
random.seed(0)


def round_robin(groups, target):
    """按视频轮转取帧：每个视频先随机打乱内部顺序，再各取一帧循环。"""
    ordered = []
    for g in groups:
        g = list(g)
        random.shuffle(g)
        ordered.append(g)
    sel, i = [], 0
    while len(sel) < target:
        added = False
        for g in ordered:
            if i < len(g) and len(sel) < target:
                sel.append(g[i])
                added = True
        if not added:
            break
        i += 1
    return sel


def class_counts(files):
    c = Counter()
    for p in files:
        lb = ROOT / "labels" / p.parent.name / (p.stem + ".txt")
        for ln in lb.read_text().splitlines():
            t = ln.split()
            if t:
                c[int(float(t[0]))] += 1
    return dict(c)


def main():
    split = "train"
    groups = []
    for v in SPLIT[split]:
        groups.append(sorted((ROOT / "images" / split).glob(f"avid__{v['video']}__*")))
    avid_sel = round_robin(groups, 4604)
    men_all = sorted((ROOT / "images" / split).glob("men__*"))

    train_files = men_all + avid_sel
    (ROOT / "train_bal.txt").write_text(
        "\n".join(f"images/{split}/{p.name}" for p in train_files) + "\n", encoding="utf-8"
    )

    vgroups = []
    for v in SPLIT["val"]:
        vgroups.append(sorted((ROOT / "images/val").glob(f"avid__{v['video']}__*")))
    avid_v = round_robin(vgroups, 616)
    men_v = sorted((ROOT / "images/val").glob("men__*"))
    (ROOT / "val_bal.txt").write_text(
        "\n".join(f"images/val/{p.name}" for p in men_v + avid_v) + "\n", encoding="utf-8"
    )

    print("men_train", len(men_all), "avid_train_sampled", len(avid_sel),
          "avid_train_boxes", class_counts(avid_sel))
    print("men_val", len(men_v), "avid_val_sampled", len(avid_v),
          "avid_val_boxes", class_counts(avid_v))

    base = (ROOT / "data.yaml").read_text(encoding="utf-8")
    lines = []
    for ln in base.splitlines():
        if ln.startswith("train:"):
            ln = "train: train_bal.txt"
        elif ln.startswith("val:"):
            ln = "val: val_bal.txt"
        elif ln.startswith("test:"):
            ln = "test: images/test"
        lines.append(ln)
    (ROOT / "data_bal.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote data_bal.yaml")


if __name__ == "__main__":
    main()
