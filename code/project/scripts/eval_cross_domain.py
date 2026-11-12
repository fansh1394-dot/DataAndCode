# -*- coding: utf-8 -*-
"""零样本误报评测：把 v2 训练出的单模态 RGB 基线（E1）直接用在 Mendeley 上。

目的
----
回答一个具体问题：**把 conf 阈值从 0.5 降到 0.05 换取召回时，虚警涨多少？**
Mendeley 含鸟类与复杂背景（树木、房屋、室内），正好是 v2 缺的干扰源。

关键口径
--------
- 模型是 RGB 单模态（E1，`--single-modal rgb`），只喂 EO 图像；不喂 IR。
- Mendeley 有 66 帧 test 空标签帧（纯背景），是**纯误报测试床**：
  这些帧上任何检测都是 FP，可以直接算 FPPI。
- 有框帧用来算召回与定位质量（但属于跨域，数值必然低于域内）。
- 图像尺寸不统一，模型内部会 letterbox 到 640，输出坐标已还原到原图。

输出
----
各 conf 阈值的 TP/FP/FN、Precision/Recall、每帧平均 FP 数（FPPI 的构成量），
以及空帧（纯背景）与有框帧分开统计。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def iou_xywh(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_yolo(p):
    out = []
    if not os.path.isfile(p):
        return out
    try:
        txt = open(p, encoding="utf-8").read()
    except UnicodeDecodeError:
        return out
    for line in txt.splitlines():
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
    ap.add_argument("--eo-root", required=True, help="UAV_eo_Dataset 路径")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-images", type=int, default=0, help="0=全部")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    proj = Path(os.environ.get("MM_PROJ", "/3250410011/root/rivermind-data/MutilModel_3485988898"))
    sys.path.insert(0, str(proj))
    from ultralytics import YOLOMM

    eo = Path(args.eo_root)
    img_dir, lab_dir = eo / "images" / args.split, eo / "labels" / args.split
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
    if args.max_images:
        files = files[: args.max_images]
    print(f"[xdomain] {len(files)} 帧  weights={args.weights}")

    model = YOLOMM(args.weights)
    results = model.predict(
        source=[str(p) for p in files],
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        conf=0.001, iou=0.7, max_det=300, verbose=False, stream=True,
    )

    # 每帧记录：GT 数、检测框 (xyxy, conf)
    per_frame = []
    for res, fp in zip(results, files):
        h, w = res.meta.get("ori_shape", (512, 640))[:2]
        gts = [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
               for cx, cy, bw, bh in read_yolo(lab_dir / f"{fp.stem}.txt")]
        if res.boxes is not None and len(res.boxes):
            arr = np.asarray(res.boxes)
            dets = [(tuple(arr[k, :4]), float(arr[k, 4])) for k in range(len(arr))]
        else:
            dets = []
        per_frame.append({"stem": fp.stem, "gt": gts, "dets": dets,
                          "hw": (h, w), "empty": len(gts) == 0})

    n_gt = sum(len(f["gt"]) for f in per_frame)
    n_empty = sum(1 for f in per_frame if f["empty"])
    print(f"[xdomain] GT={n_gt}  空帧(纯背景)={n_empty}  有框帧={len(per_frame)-n_empty}")

    rows = []
    for thr in (0.5, 0.25, 0.1, 0.05, 0.01, 0.001):
        tp = fp_ = fn = 0
        fp_empty = 0
        ious = []
        for f in per_frame:
            d = [x for x in f["dets"] if x[1] >= thr]
            if f["empty"]:
                fp_empty += len(d)   # 空帧上任何检测都是 FP
                continue
            used = set()
            for box, c in d:
                best, bj = 0.0, -1
                for j, g in enumerate(f["gt"]):
                    if j in used:
                        continue
                    v = iou_xywh(box, g)
                    if v > best:
                        best, bj = v, j
                if best >= 0.5 and bj >= 0:
                    used.add(bj)
                    tp += 1
                    ious.append(best)
                else:
                    fp_ += 1
            fn += len(f["gt"]) - len(used)
        fps = fp_ + fp_empty
        rows.append({
            "conf": thr, "TP": tp, "FP": fps, "FP_on_empty": fp_empty, "FN": fn,
            "precision": tp / max(tp + fps, 1),
            "recall": tp / max(n_gt, 1),
            "fp_per_empty_frame": fp_empty / max(n_empty, 1),
            "mean_iou_tp": float(np.mean(ious)) if ious else 0.0,
        })
        print(f"  conf>={thr:<6} TP={tp:<5} FP={fps:<5}(空帧{fp_empty:<4}) FN={fn:<5} "
              f"P={tp/max(tp+fps,1):.4f} R={tp/max(n_gt,1):.4f} "
              f"空帧FP/帧={fp_empty/max(n_empty,1):.2f} IoU={np.mean(ious) if ious else 0:.3f}")

    out = proj / "ResTest" / "_test_eval" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag, "weights": args.weights, "domain": "mendeley_eo_ir",
        "split": args.split, "n_frames": len(per_frame), "n_gt": n_gt,
        "n_empty_frames": n_empty, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
