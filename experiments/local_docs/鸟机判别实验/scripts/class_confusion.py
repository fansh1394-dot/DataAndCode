# -*- coding: utf-8 -*-
"""通用类别混淆统计：GT 框匹配到的预测类别分布。"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def iou_one(b, bs):
    if len(bs) == 0:
        return np.zeros(0)
    x1 = np.maximum(b[0], bs[:, 0]); y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2]); y2 = np.minimum(b[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1])
    bb = (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / np.clip(a + bb - inter, 1e-9, None)


ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--root", default="/root/rivermind-fs/birddrone/bd_nc2")
ap.add_argument("--split", default="val")
ap.add_argument("--list", required=True)
ap.add_argument("--tau", type=float, default=0.25)
a = ap.parse_args()

preds = {}
for line in Path(a.preds).read_text().splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    preds[Path(d["im_file"]).name] = (np.asarray(d["bboxes"], np.float32).reshape(-1, 4),
                                      np.asarray(d["conf"], np.float32),
                                      np.asarray(d["cls"], np.int32))

names = [x.strip() for x in Path(a.root, a.list).read_text().splitlines() if x.strip()]
stat = Counter(); n_gt = Counter()
for rel in names:
    p = Path(rel)
    stem = p.stem
    lb = Path(a.root) / "labels" / a.split / f"{stem}.txt"
    if not lb.exists():
        continue
    img = Path(a.root) / rel
    if not img.exists():
        cand = list((Path(a.root) / "images" / a.split).glob(f"{stem}.*"))
        if not cand:
            continue
        img = cand[0]
    with Image.open(img) as im:
        w, h = im.size
    gts = []
    for ln in lb.read_text().splitlines():
        t = ln.split()
        if len(t) >= 5:
            c = int(float(t[0])); cx, cy, bw, bh = [float(v) for v in t[1:5]]
            gts.append((c, np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                     (cx + bw / 2) * w, (cy + bh / 2) * h], np.float32)))
    pb, pc, pk = preds.get(img.name, (np.zeros((0, 4), np.float32),
                                     np.zeros(0, np.float32), np.zeros(0, np.int32)))
    keep = pc >= a.tau
    pb, pk = pb[keep], pk[keep]
    for c, g in gts:
        n_gt[c] += 1
        ious = iou_one(g, pb)
        stat[(c, int(pk[int(ious.argmax())]) if len(ious) and ious.max() >= 0.5 else -1)] += 1

print("GT", dict(n_gt))
labels = {0: "bird", 1: "drone", 2: "other", -1: "miss"}
for c in sorted(n_gt):
    n = n_gt[c]
    row = {labels[k]: round(stat[(c, k)] / n, 4) for k in (-1, 0, 1, 2)}
    print(f"GT={labels[c]} n={n}", row)
