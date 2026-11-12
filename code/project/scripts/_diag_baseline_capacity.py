# -*- coding: utf-8 -*-
"""对 baseline 权重做尺度分桶 + 置信度分布的检测能力诊断（eval-only）。

回答的问题
----------
1. 模型在不同目标尺寸下 mAP 如何（远距离小目标是不是真的差）
2. 高 IoU 阈值下还剩多少（定位精度是否够）
3. 召回率随置信度阈值怎么变（漏检在哪个操作点）
4. 置信度分布是否双峰（是否有大量低置信度真检被压掉）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def iou_xywh(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def yolo_to_xyxy(b, w, h):
    cx, cy, bw, bh = b
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def read_yolo(p):
    out = []
    if not os.path.isfile(p):
        return out
    for line in open(p, encoding="utf-8", errors="replace"):
        s = line.split()
        if len(s) >= 5:
            try:
                out.append(tuple(float(x) for x in s[1:5]))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="/root/rivermind-data/uav_data_v2/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-images", type=int, default=3000)
    ap.add_argument("--labels-mode", default="vis")
    args = ap.parse_args()

    proj = Path(os.environ.get("MM_PROJ", "/3250410011/root/rivermind-data/MutilModel_3485988898"))
    sys.path.insert(0, str(proj))
    from ultralytics import YOLOMM

    import yaml
    cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    root = Path(cfg["path"])
    img_dir = root / cfg[args.split]
    lab_name = cfg["labels"]["rgb"] if args.labels_mode == "vis" else cfg["labels"]["ir"]
    lab_dir = root / lab_name / args.split

    files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")])[: args.max_images]
    print(f"[diag] {len(files)} 张，标签目录 {lab_dir}")

    model = YOLOMM(args.weights)
    results = model.predict(
        source=[str(p) for p in files],
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        conf=0.001, iou=0.7, max_det=300, verbose=False, stream=True,
    )

    buckets = {"<16px": [], "16-32px": [], "32-64px": [], ">=64px": []}
    conf_tp, conf_fp = [], []
    n_gt = n_miss = 0
    per_iou = {t: [0, 0] for t in (0.5, 0.75, 0.9)}

    for res, fp in zip(results, files):
        # MultiModalResults: boxes = [N,6] numpy (x1,y1,x2,y2,conf,cls)，形状在 meta['ori_shape']
        h, w = res.meta.get("ori_shape", (512, 640))[:2]
        gts = [yolo_to_xyxy(b, w, h) for b in read_yolo(lab_dir / f"{fp.stem}.txt")]
        n_gt += len(gts)
        if res.boxes is None or len(res.boxes) == 0:
            n_miss += len(gts)
            continue
        pb = np.asarray(res.boxes)[:, :4]
        pc = np.asarray(res.boxes)[:, 4]
        used = set()
        matched_gt = set()
        for k in range(len(pb)):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in used:
                    continue
                v = iou_xywh(tuple(pb[k]), g)
                if v > best:
                    best, bj = v, j
            if best >= 0.5 and bj >= 0:
                used.add(bj)
                matched_gt.add(bj)
                conf_tp.append(float(pc[k]))
                gx0, gy0, gx1, gy1 = gts[bj]
                side = np.sqrt(max(gx1 - gx0, 1) * max(gy1 - gy0, 1))
                key = ("<16px" if side < 16 else "16-32px" if side < 32
                       else "32-64px" if side < 64 else ">=64px")
                buckets[key].append(best)
                for t in per_iou:
                    per_iou[t][1] += 1
                    if best >= t:
                        per_iou[t][0] += 1
            else:
                conf_fp.append(float(pc[k]))
        n_miss += len(gts) - len(matched_gt)

    print(f"\n=== 检测能力诊断（conf=0.001, IoU=0.5 判定）===")
    print(f"  GT 总数 {n_gt}，漏检 {n_miss}，召回上限 {(n_gt-n_miss)/max(n_gt,1)*100:.2f}%")
    print(f"\n  IoU 阈值下的匹配保留率（相对已匹配）：")
    for t, (a, b) in per_iou.items():
        print(f"    IoU>={t}: {a/max(b,1)*100:.2f}%")
    print(f"\n  尺度分桶（已匹配 GT 的 IoU 分布）：")
    for k, v in buckets.items():
        if v:
            a = np.array(v)
            print(f"    {k:<10} n={len(a):<5} IoU 中位={np.median(a):.3f} 均值={a.mean():.3f}")
    if conf_tp:
        a, b = np.array(conf_tp), np.array(conf_fp) if conf_fp else np.array([0.0])
        print(f"\n  置信度分布：")
        print(f"    TP 中位数={np.median(a):.4f} 均值={a.mean():.4f} | "
              f"FP 中位数={np.median(b):.4f} 均值={b.mean():.4f} | FP/TP={len(b)/max(len(a),1):.2f}")
        for thr in (0.01, 0.05, 0.1, 0.25, 0.5):
            tp = int((a >= thr).sum()); fp = int((b >= thr).sum())
            print(f"    conf>={thr:<5}: TP={tp:<6} FP={fp:<6} P={tp/max(tp+fp,1):.4f} "
                  f"R={tp/max(n_gt,1):.4f}")


if __name__ == "__main__":
    main()
