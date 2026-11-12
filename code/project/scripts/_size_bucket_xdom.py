# -*- coding: utf-8 -*-
"""跨域召回率按目标尺寸分桶，定位失效位置。

在 eval_cross_domain 已保存的逐帧结果基础上，按 **GT 等效边长**分桶统计召回：
    <16px / 16-32px / 32-64px / 64-128px / >=128px
并同时给出域内（v2 test）同口径对照，回答「跨域掉点是否集中在特定尺寸段」。

同时输出「按检测框尺寸分桶」的 FP 分布，判断是否模型只会输出某一尺度的框。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
LABELS = ["<16px", "16-32px", "32-64px", "64-128px", ">=128px"]


def bin_of(side):
    for k, (lo, hi) in enumerate(BINS):
        if lo <= side < hi:
            return LABELS[k]
    return LABELS[-1]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def side_of(b):
    return float(np.sqrt(max(b[2] - b[0], 1e-6) * max(b[3] - b[1], 1e-6)))


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


def run(weights, img_dir, lab_dir, conf_thr, device="0"):
    prod = Path(os.environ["MM_PROJ"])
    sys.path.insert(0, str(prod))
    from ultralytics import YOLOMM
    files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    model = YOLOMM(weights)
    res_iter = model.predict(source=[str(p) for p in files], imgsz=640, batch=16,
                             device=device, conf=0.001, iou=0.7, max_det=300,
                             verbose=False, stream=True)
    gt_stat = defaultdict(lambda: [0, 0])   # bin -> [hit, total]
    det_bins = defaultdict(int)
    det_conf = defaultdict(list)
    for res, fp in zip(res_iter, files):
        h, w = res.meta.get("ori_shape", (512, 640))[:2]
        gts = [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
               for cx, cy, bw, bh in read_yolo(Path(lab_dir) / f"{fp.stem}.txt")]
        dets = []
        if res.boxes is not None and len(res.boxes):
            arr = np.asarray(res.boxes)
            dets = [(tuple(arr[k, :4]), float(arr[k, 4])) for k in range(len(arr))]
        # GT 命中（conf>=thr 已匹配）
        for g in gts:
            b = bin_of(side_of(g))
            gt_stat[b][1] += 1
            for box, c in dets:
                if c >= conf_thr and iou(box, g) >= 0.5:
                    gt_stat[b][0] += 1
                    break
        # 检测框尺寸分布
        for box, c in dets:
            if c >= conf_thr:
                b = bin_of(side_of(box))
                det_bins[b] += 1
                det_conf[b].append(c)
    return gt_stat, det_bins, det_conf


def show(name, gt_stat, det_bins, det_conf, conf_thr):
    print(f"\n=== {name} (conf>={conf_thr}) ===")
    print(f"  {'尺寸桶':<10}{'GT数':>7}{'命中':>7}{'召回':>9}{'检测框数':>10}{'框均conf':>10}")
    tot_h = tot_n = 0
    for b in LABELS:
        h, n = gt_stat.get(b, [0, 0])
        tot_h += h
        tot_n += n
        mc = np.mean(det_conf.get(b, [0])) if det_conf.get(b) else 0
        print(f"  {b:<10}{n:>7}{h:>7}{(h/max(n,1)):>9.3f}{det_bins.get(b,0):>10}{mc:>10.3f}")
    print(f"  {'合计':<10}{tot_n:>7}{tot_h:>7}{(tot_h/max(tot_n,1)):>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--domain", required=True, choices=["v2", "mendeley"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.domain == "v2":
        root = Path("/root/rivermind-data/uav_data_v2")
        img_dir, lab_dir = root / "images/test", root / "labels/test"
        name = "v2 test（域内）"
    else:
        root = Path("/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/"
                    "Labelled_UAV_Images/Labelled EO_IR Dataset/UAV_eo_Dataset")
        img_dir, lab_dir = root / "images/test", root / "labels/test"
        name = "Mendeley test（跨域）"

    gt_stat, det_bins, det_conf = run(args.weights, img_dir, lab_dir, args.conf)
    show(name, gt_stat, det_bins, det_conf, args.conf)

    out = Path(os.environ["MM_PROJ"]) / "ResTest/_test_eval" / f"sizebucket_{args.tag or args.domain}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "domain": args.domain, "conf": args.conf,
        "gt": {b: gt_stat.get(b, [0, 0]) for b in LABELS},
        "det_bins": {b: det_bins.get(b, 0) for b in LABELS},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
