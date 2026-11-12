# -*- coding: utf-8 -*-
"""E27 / D0：现有 nc=1 无人机检测器在 AVID 鸟验证集上的提议召回。

不训练，只做匹配统计：
  - 每张 GT 鸟框是否被「任一类」预测框以 IoU>=0.3 命中（可加置信度门槛）；
  - 按原生等效边长分桶（<8 / 8-16 / 16-24 / >=24 px）；
  - 同时报 IoU 0.5 口径，便于与常规 mAP 对照。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BUCKETS = [(0, 8), (8, 16), (16, 24), (24, 1e9)]
TAUS = [0.0, 0.01, 0.05, 0.1, 0.25]


def xywhn_to_xyxy(cx, cy, w, h, iw, ih):
    return np.array([(cx - w / 2) * iw, (cy - h / 2) * ih,
                     (cx + w / 2) * iw, (cy + h / 2) * ih], dtype=np.float32)


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a + b - inter, 1e-9, None)


def bucket_of(side):
    for lo, hi in BUCKETS:
        if lo <= side < hi:
            return f"{lo}-{hi}" if hi < 1e9 else f"{lo}+"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", default="/root/rivermind-fs/birddrone/avid_flat")
    ap.add_argument("--list", default="/root/rivermind-fs/birddrone/avid_flat/avid_bird_val_flat.txt")
    ap.add_argument("--preds", default="/root/rivermind-fs/birddrone/E27/preds.jsonl")
    ap.add_argument("--out", default="/root/rivermind-fs/birddrone/E27/E27_d0_recall.json")
    a = ap.parse_args()

    flat = Path(a.flat)
    names = [x.strip() for x in Path(a.list).read_text().splitlines() if x.strip()]

    preds = {}
    for line in Path(a.preds).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        preds[Path(r["im_file"]).name] = (
            np.asarray(r["bboxes"], dtype=np.float32).reshape(-1, 4),
            np.asarray(r["conf"], dtype=np.float32),
        )

    # 逐 GT 记录：边长分桶、各 tau 下 IoU0.3/0.5 是否命中、最佳 IoU/置信度
    rows = []
    n_img_with_pred = 0
    n_pred_total = 0
    for rel in names:
        img_name = Path(rel).name
        label = flat / "labels/val" / (Path(rel).stem + ".txt")
        gts = []
        for ln in label.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                gts.append([float(v) for v in p[1:5]])
        if not gts:
            continue
        from PIL import Image
        with Image.open(flat / rel) as im:
            iw, ih = im.size
        gts = [xywhn_to_xyxy(*g, iw, ih) for g in gts]
        pb, pc = preds.get(img_name, (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)))
        n_pred_total += len(pb)
        n_img_with_pred += int(len(pb) > 0)
        for g in gts:
            ious = iou_one_to_many(g, pb)
            side = float(np.sqrt(max((g[2] - g[0]) * (g[3] - g[1]), 1e-6)))
            row = {
                "img": img_name,
                "side": side,
                "bucket": bucket_of(side),
                "best_iou": float(ious.max()) if len(ious) else 0.0,
                "best_iou_conf": float(pc[int(ious.argmax())]) if len(ious) else 0.0,
                "n_pred": int(len(pb)),
                "hit": {},
            }
            for tau in TAUS:
                keep = pc >= tau
                for iou_thr in (0.3, 0.5):
                    hit = bool(len(ious) and np.any((ious >= iou_thr) & keep))
                    row["hit"][f"iou{iou_thr}_tau{tau}"] = hit
            rows.append(row)

    def agg(sel):
        d = {"n": len(sel)}
        for key in rows[0]["hit"] if rows else []:
            d[key] = round(float(np.mean([r["hit"][key] for r in sel])), 4) if sel else None
        d["side_median"] = round(float(np.median([r["side"] for r in sel])), 2) if sel else None
        d["best_iou_mean"] = round(float(np.mean([r["best_iou"] for r in sel])), 4) if sel else None
        d["best_iou_median"] = round(float(np.median([r["best_iou"] for r in sel])), 4) if sel else None
        return d

    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    out = {
        "images": len(names),
        "images_with_pred": n_img_with_pred,
        "pred_boxes": n_pred_total,
        "pred_per_image_mean": round(n_pred_total / max(len(names), 1), 3),
        "gt_boxes": len(rows),
        "taus": TAUS,
        "overall": agg(rows),
        "by_bucket": {k: agg(v) for k, v in sorted(by_bucket.items())},
        "preds_file": str(a.preds),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
