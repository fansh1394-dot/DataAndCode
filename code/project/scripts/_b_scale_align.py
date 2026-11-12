# -*- coding: utf-8 -*-
"""B 方案：尺度对齐后重测跨域召回。

设计
----
先量出两域的**目标中位等效边长**（在模型输入 640 尺度下）：
   v2 test 与 Mendeley test 各自的中位数。
然后把 Mendeley 每帧按因子 s = ref_v2 / ref_mendeley 整体缩放（图像 + 标签同步变换），
使缩放后 Mendeley 的目标中位尺寸等于 v2。再跑同一权重，比较召回。

为什么用「整帧等比缩放」而不是「逐目标缩放」
------------------------------------------
逐目标缩放会破坏图像内容（目标与背景比例失真），属人为造数据；
整帧缩放只改变成像尺度，等价于「换成不同焦距的相机」，是公平的域变量隔离。

实现
----
- 用 cv2.resize 把每帧缩放到 (W', H')，并同步缩放 YOLO 标签（归一化坐标不变！）
- 归一化坐标在等比缩放后**保持不变**，所以标签文件可以直接复用
- 只改变输入图像的分辨率，模型内部再 letterbox 到 640 —— 这恰好模拟了不同的成像尺度
- 为避免磁盘占用，只在内存里流式处理（用临时目录，测完即删）

注意
----
因为模型 input 固定 640、内部 letterbox，缩放整帧再 letterbox 到 640 后，
**目标在 640 画布上的像素尺寸确实会变**，这正是我们想要的效果。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
LABELS = ["<16px", "16-32px", "32-64px", "64-128px", ">=128px"]


def bin_of(s):
    for k, (lo, hi) in enumerate(BINS):
        if lo <= s < hi:
            return LABELS[k]
    return LABELS[-1]


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


def median_side_at640(img_dir, lab_dir, stride=1):
    """在 imgsz=640 letterbox 后的坐标系里，计算目标等效边长中位数。"""
    sides = []
    files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    for fp in files[::stride]:
        b = read_yolo(Path(lab_dir) / f"{fp.stem}.txt")
        if not b:
            continue
        im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        h, w = im.shape[:2]
        r = min(640 / w, 640 / h)
        nw, nh = round(w * r), round(h * r)
        for _, _, bw, bh in b:
            sides.append(float(np.sqrt(bw * nw * bh * nh)))
    return float(np.median(sides)) if sides else 0.0, sides


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mendeley-root", required=True)
    ap.add_argument("--v2-root", default="/root/rivermind-data/uav_data_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--scale-mode", default="auto", choices=["auto", "none"],
                    help="auto=按中位尺寸比缩放；none=原图（对照）")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    proj = Path(os.environ["MM_PROJ"])
    sys.path.insert(0, str(proj))
    from ultralytics import YOLOMM

    eo = Path(args.mendeley_root)
    img_dir, lab_dir = eo / "images" / args.split, eo / "labels" / args.split

    # ---------- 1. 量中位尺寸 ----------
    print("[B] 统计两域在 640 输入下的目标中位等效边长 ...")
    med_v2, _ = median_side_at640(Path(args.v2_root) / "images/test",
                                  Path(args.v2_root) / "labels/test", stride=10)
    med_md, _ = median_side_at640(img_dir, lab_dir)
    print(f"    v2 test  中位 = {med_v2:.2f}px")
    print(f"    Mendeley 中位 = {med_md:.2f}px")
    scale = (med_v2 / med_md) if (args.scale_mode == "auto" and med_md > 0) else 1.0
    print(f"    缩放因子 s = {scale:.4f}  （图像尺寸 ×{scale:.3f}）")

    # ---------- 2. 生成缩放后的临时图像 ----------
    tmp = Path(os.environ["MM_PROJ"]) / "_d2p1" / f"_b_scale_{args.tag}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
    out_files = []
    for fp in files:
        im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        h, w = im.shape[:2]
        nw, nh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        out = cv2.resize(im, (nw, nh), interpolation=interp)
        dst = tmp / fp.name
        cv2.imwrite(str(dst), out)
        out_files.append(dst)
    print(f"[B] 已生成 {len(out_files)} 张缩放图 -> {tmp}")

    # ---------- 3. 推理 ----------
    model = YOLOMM(args.weights)
    res_iter = model.predict(source=[str(p) for p in out_files], imgsz=640, batch=16,
                             device="0", conf=0.001, iou=0.7, max_det=300,
                             verbose=False, stream=True)

    gt_stat = defaultdict(lambda: [0, 0])
    det_bins = defaultdict(int)
    n_gt = n_hit = n_fp_empty = n_empty = 0

    for res, fp in zip(res_iter, out_files):
        h, w = res.meta.get("ori_shape", (512, 640))[:2]
        # 标签：归一化坐标与缩放无关，直接用原标签
        boxes = read_yolo(lab_dir / f"{fp.stem}.txt")
        gts = [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
               for cx, cy, bw, bh in boxes]
        dets = []
        if res.boxes is not None and len(res.boxes):
            arr = np.asarray(res.boxes)
            dets = [(tuple(arr[k, :4]), float(arr[k, 4])) for k in range(len(arr))]
        if not gts:
            n_empty += 1
            n_fp_empty += sum(1 for _, c in dets if c >= args.conf)
            continue
        used = set()
        for g in gts:
            n_gt += 1
            s = float(np.sqrt(max(g[2] - g[0], 1e-6) * max(g[3] - g[1], 1e-6)))
            b = bin_of(s)
            gt_stat[b][1] += 1
            hit = False
            for k, (box, c) in enumerate(dets):
                if k in used or c < args.conf:
                    continue
                if iou(box, g) >= 0.5:
                    used.add(k)
                    hit = True
                    break
            if hit:
                gt_stat[b][0] += 1
                n_hit += 1
        for box, c in dets:
            if c >= args.conf:
                s = float(np.sqrt(max(box[2] - box[0], 1e-6) * max(box[3] - box[1], 1e-6)))
                det_bins[bin_of(s)] += 1

    print(f"\n=== B 结果 tag={args.tag} scale={scale:.3f} conf={args.conf} ===")
    print(f"  空帧 {n_empty}，空帧 FP {n_fp_empty}")
    print(f"  {'尺寸桶':<10}{'GT数':>7}{'命中':>7}{'召回':>9}{'检测框':>8}")
    for b in LABELS:
        hh, nn = gt_stat.get(b, [0, 0])
        print(f"  {b:<10}{nn:>7}{hh:>7}{(hh/max(nn,1)):>9.3f}{det_bins.get(b,0):>8}")
    print(f"  {'合计':<10}{n_gt:>7}{n_hit:>7}{(n_hit/max(n_gt,1)):>9.3f}")

    out = proj / "ResTest/_test_eval" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag, "scale": scale, "med_v2": med_v2, "med_mendeley": med_md,
        "conf": args.conf, "n_gt": n_gt, "n_hit": n_hit,
        "recall": n_hit / max(n_gt, 1), "n_empty": n_empty, "fp_on_empty": n_fp_empty,
        "buckets": {b: gt_stat.get(b, [0, 0]) for b in LABELS},
        "det_bins": {b: det_bins.get(b, 0) for b in LABELS},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
