# -*- coding: utf-8 -*-
"""E36：合成 other 上的类别混淆统计（重点：other 被判成 drone 的比例）。"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

R = Path("/root/rivermind-fs/birddrone/bd_nc3")
PRED = Path("/root/rivermind-fs/birddrone/E36/preds_syn.jsonl")
TAU, IOU = 0.25, 0.5


def iou_one(b, bs):
    if len(bs) == 0:
        return np.zeros(0)
    x1 = np.maximum(b[0], bs[:, 0]); y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2]); y2 = np.minimum(b[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1])
    bb = (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / np.clip(a + bb - inter, 1e-9, None)


preds = {}
for line in PRED.read_text().splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    preds[Path(d["im_file"]).name] = (np.asarray(d["bboxes"], np.float32).reshape(-1, 4),
                                      np.asarray(d["conf"], np.float32),
                                      np.asarray(d["cls"], np.int32))

stat = Counter()
n_gt = Counter()
for lb in sorted((R / "labels/val").glob("*__syn*.txt")):
    img = R / "images/val" / (lb.stem + ".jpg")
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
    keep = pc >= TAU
    pb, pk = pb[keep], pk[keep]
    for c, g in gts:
        n_gt[c] += 1
        ious = iou_one(g, pb)
        if len(ious) and ious.max() >= IOU:
            stat[(c, int(pk[int(ious.argmax())]))] += 1
        else:
            stat[(c, -1)] += 1

print("GT 计数", dict(n_gt))
for c in sorted(n_gt):
    row = {k: stat[(c, k)] for k in (-1, 0, 1, 2)}
    print(f"GT={c} 漏检={row[-1]} 判bird={row[0]} 判drone={row[1]} 判other={row[2]}")
n2 = n_gt[2]
if n2:
    print("other->drone 比例", round(stat[(2, 1)] / n2, 4),
          " other 漏检比例", round(stat[(2, -1)] / n2, 4))
