# -*- coding: utf-8 -*-
"""DIV-C 背景库构建（C2 变体：RGB 与 IR 成对保存同一坐标的背景块）。

为什么需要
----------
v2 训练集以纯天空为主，跨域失败样本（C1 可视化 12/12）全部落在树林/树梢/草地/建筑
等纹理背景。把纹理背景直接造进训练分布，是 DIV-C 的干预手段。

构建规则
--------
1. 只取 valid_rgb 且 valid_ir 的 train 帧（C2 需要两模态背景严格同坐标）
2. 每序列按 frame_index 排序后隔 frame_step 取一帧
3. 128×128 窗口、步长 64 滑动；与任一 GT 框（RGB 与 IR 两套标签，外扩 margin）相交即剔除
4. RGB 灰度方差 < min_var 的低纹理窗口剔除
5. 每序列最多 per_seq 个窗口，候选位置先收集、再随机抽per_seq，避免只用最早几帧

输出
----
<out>/index.json      {"patch":128, "total":N, "seqs":{seq:{"n":k,"rgb":"x.npy","ir":"x_ir.npy"}}}
<out>/seq_map.json    {frame_stem: sequence}，供训练期排除「同序列背景」
<out>/<seq>.npy       uint8 (k,128,128,3) RGB
<out>/<seq>_ir.npy    uint8 (k,128,128,3) IR

用法
----
python scripts/build_bg_bank.py \
    --data-root /root/rivermind-data/uav_data_v2 \
    --out /root/rivermind-data/uav_data_v2/bg_bank \
    --per-seq 50 --frame-step 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    ap = argparse.ArgumentParser(description="构建 DIV-C 背景库（C2：RGB/IR 成对）")
    ap.add_argument("--data-root", default="/root/rivermind-data/uav_data_v2")
    ap.add_argument("--out", default=None, help="默认 <data-root>/bg_bank")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--min-var", type=float, default=15.0, help="RGB 灰度方差下限")
    ap.add_argument("--per-seq", type=int, default=50)
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--margin", type=int, default=8, help="GT 外扩像素，避免窗口压到目标边缘")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def find_image(folder: Path, stem: str):
    for ext in IMG_EXT:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_boxes(txt: Path, W: int, H: int, margin: int):
    """读归一化 xywh 标签，返回像素 xyxy（已外扩 margin）。空文件/缺文件返回空列表。"""
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        x0 = (cx - w / 2) * W - margin
        y0 = (cy - h / 2) * H - margin
        x1 = (cx + w / 2) * W + margin
        y1 = (cy + h / 2) * H + margin
        out.append((x0, y0, x1, y1))
    return out


def overlaps_any(x, y, p, boxes) -> bool:
    x1, y1 = x + p, y + p
    for bx0, by0, bx1, by1 in boxes:
        if x1 > bx0 and x < bx1 and y1 > by0 and y < by1:
            return True
    return False


def scan_sequence(stems, rgb_dir, ir_dir, lab_dir, lab_ir_dir, args):
    """收集候选窗口坐标（不做像素拷贝，避免大内存）。"""
    cands = []
    for stem in stems:
        rgb_p = find_image(rgb_dir, stem)
        ir_p = find_image(ir_dir, stem)
        if rgb_p is None or ir_p is None:
            continue
        rgb = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        boxes = read_boxes(lab_dir / f"{stem}.txt", W, H, args.margin)
        boxes += read_boxes(lab_ir_dir / f"{stem}.txt", W, H, args.margin)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        p = args.patch
        for y in range(0, H - p + 1, args.stride):
            for x in range(0, W - p + 1, args.stride):
                if overlaps_any(x, y, p, boxes):
                    continue
                if float(gray[y:y + p, x:x + p].var()) < args.min_var:
                    continue
                cands.append((stem, x, y))
    return cands


def extract(cands, rgb_dir, ir_dir, patch):
    rgb_list, ir_list = [], []
    cache = {}
    for stem, x, y in cands:
        if stem not in cache:
            rgb_p = find_image(rgb_dir, stem)
            ir_p = find_image(ir_dir, stem)
            cache[stem] = (cv2.imread(str(rgb_p), cv2.IMREAD_COLOR),
                           cv2.imread(str(ir_p), cv2.IMREAD_COLOR))
        rgb, ir = cache[stem]
        if rgb is None or ir is None:
            continue
        rgb_list.append(rgb[y:y + patch, x:x + patch])
        ir_list.append(ir[y:y + patch, x:x + patch])
    if not rgb_list:
        return None, None
    return np.stack(rgb_list).astype(np.uint8), np.stack(ir_list).astype(np.uint8)


def main():
    args = parse_args()
    root = Path(args.data_root)
    out = Path(args.out) if args.out else root / "bg_bank"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    meta = json.loads((root / "metadata" / "train.json").read_text(encoding="utf-8"))
    frames = meta["frames"]
    by_seq = defaultdict(list)
    seq_map = {}
    for stem, info in frames.items():
        seq = info.get("sequence")
        if not seq:
            continue
        seq_map[stem] = seq
        if info.get("valid_rgb") and info.get("valid_ir"):
            by_seq[seq].append((int(info.get("frame_index", 0)), stem))
    for seq in by_seq:
        by_seq[seq] = [s for _, s in sorted(by_seq[seq])][:: max(1, args.frame_step)]

    rgb_dir = root / "images" / "train"
    ir_dir = root / "images_ir" / "train"
    lab_dir = root / "labels" / "train"
    lab_ir_dir = root / "labels_ir" / "train"

    index = {"patch": args.patch, "seqs": {}, "total": 0}
    total = 0
    for i, seq in enumerate(sorted(by_seq), 1):
        stems = by_seq[seq]
        cands = scan_sequence(stems, rgb_dir, ir_dir, lab_dir, lab_ir_dir, args)
        if len(cands) > args.per_seq:
            cands = random.sample(cands, args.per_seq)
        rgb_arr, ir_arr = extract(cands, rgb_dir, ir_dir, args.patch)
        if rgb_arr is None:
            print(f"[{i:3d}/{len(by_seq)}] {seq}: 无可用窗口，跳过")
            continue
        safe = re.sub(r"[^0-9A-Za-z_.-]", "_", seq)
        np.save(out / f"{safe}.npy", rgb_arr)
        np.save(out / f"{safe}_ir.npy", ir_arr)
        index["seqs"][seq] = {"n": int(len(rgb_arr)), "rgb": f"{safe}.npy", "ir": f"{safe}_ir.npy"}
        total += len(rgb_arr)
        print(f"[{i:3d}/{len(by_seq)}] {seq}: {len(rgb_arr)} 窗口（候选 {len(cands)}）")

    index["total"] = total
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "seq_map.json").write_text(json.dumps(seq_map, ensure_ascii=False), encoding="utf-8")

    ns = [v["n"] for v in index["seqs"].values()]
    print(f"\n完成：{len(index['seqs'])} 个序列 / {total} 个背景块")
    if ns:
        print(f"每序列块数: min={min(ns)} 中位={int(np.median(ns))} max={max(ns)}")
    size_gb = total * args.patch * args.patch * 3 * 2 / 1024 ** 3
    print(f"磁盘占用约 {size_gb:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
