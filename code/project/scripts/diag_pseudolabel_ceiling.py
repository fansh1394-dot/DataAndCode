# -*- coding: utf-8 -*-
"""伪标签蒸馏可行性诊断：教师(800)框与 GT 的一致性。

用途：在决定「把 GT 框替换成教师预测框再重训 640」之前，先量化替换会引入
多大的标签误差。若替换框与 GT 的中位 IoU 低于 GT 自身的 1.0，那么学生能学到
的上限就是教师自身的 AP——即伪标签在信息论上劣于原始 GT。

只读已有产物：perframe_boxes2/boxes_*.json + ratio_*.json + labels/test。
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
LABEL_DIR = "/3250410011/root/rivermind-data/mendeley_split/labels/test"
PF = os.path.join(PROJ, "perframe_boxes2")

BINS = [("lt16", 0.0, 16.0), ("b16_32", 16.0, 32.0), ("b32_64", 32.0, 64.0), ("gt64", 64.0, 1e9)]
CONF = 0.30


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-12 else 0.0


def bin_of(side):
    for name, lo, hi in BINS:
        if lo <= side < hi:
            return name
    return BINS[-1][0]


def analyze(tag, head="fuse"):
    boxes = json.load(open(os.path.join(PF, f"boxes_{tag}.json")))
    ratio = json.load(open(os.path.join(PF, f"ratio_{tag}.json")))
    hit = defaultdict(list)
    miss = defaultdict(int)
    ngt = defaultdict(int)
    for name, rec in boxes.items():
        stem = os.path.splitext(name)[0]
        lp = os.path.join(LABEL_DIR, stem + ".txt")
        if not os.path.exists(lp) or name not in ratio:
            continue
        r = ratio[name]
        h, w = r["ori_shape"]
        gx, gy = r["gain"]
        px, py = r["pad"]
        gts = []
        with open(lp) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                x, y, bw, bh = map(float, p[1:5])
                gts.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
        preds = rec[head]
        pb = []
        for bb, c in zip(preds["xyxy"], preds["conf"]):
            if c < CONF:
                continue
            pb.append([(bb[0] - px) / gx, (bb[1] - py) / gy, (bb[2] - px) / gx, (bb[3] - py) / gy])
        for g in gts:
            side = math.sqrt(max(g[2] - g[0], 0.0) * max(g[3] - g[1], 0.0))
            b = bin_of(side)
            ngt[b] += 1
            best = 0.0
            for bb in pb:
                v = iou(g, bb)
                if v > best:
                    best = v
            if best <= 0.0:
                miss[b] += 1
            else:
                hit[b].append(best)

    out = {"tag": tag, "head": head, "conf": CONF, "buckets": {}}
    for name, _, _ in BINS:
        v = sorted(hit[name])
        n = ngt[name]
        if not v:
            out["buckets"][name] = {"nGT": n, "matched": 0, "unmatched": miss[name]}
            continue
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        out["buckets"][name] = {
            "nGT": n,
            "matched": len(v),
            "match_rate": round(len(v) / n, 4),
            "unmatched": miss[name],
            "iou_p10": round(q(0.10), 4),
            "iou_median": round(q(0.50), 4),
            "iou_p90": round(q(0.90), 4),
            "frac_iou_ge_0.5": round(sum(x >= 0.5 for x in v) / len(v), 4),
            "frac_iou_ge_0.75": round(sum(x >= 0.75 for x in v) / len(v), 4),
            "frac_iou_ge_0.9": round(sum(x >= 0.9 for x in v) / len(v), 4),
        }
    return out


if __name__ == "__main__":
    res = {}
    for tag in ("s0", "base800"):
        try:
            res[tag] = analyze(tag)
        except FileNotFoundError as e:
            print(f"[skip] {tag}: {e}")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(PF, "pseudolabel_ceiling.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
