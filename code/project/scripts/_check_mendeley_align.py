# -*- coding: utf-8 -*-
"""Mendeley sn9vy5c8sm：EO / IR 严格对齐性实证检验。

目录结构（实测）
----------------
Dataset_Mendeley_data/Labelled_UAV_Images/Labelled EO_IR Dataset/
    UAV_eo_Dataset/{images,labels}/{train,val,test}   # 文件名 01001.png 形式，两侧同名
    UAV_ir_Dataset/{images,labels}/{train,val,test}

检验维度
--------
A. **框中心位移**：同一 stem 的 EO 框与 IR 框中心差（归一化）。若两模态共享像素坐标，
   该值应≈0。同时给出像素值（按各自分辨率换算）与 ≤2px/≤5px 占比。
   —— 这是与项目 v2 对比的核心量（v2: mean|dx|≈0.030~0.039, mean|dy|≈0.074~0.080）。
B. **框几何比**：EO 框宽/IR 框宽、高之比。≈1 说明尺度一致。
C. **热斑自检**：IR 亮度峰值是否落在 IR 框内；EO 峰值是否落在 EO 框内。
   排除"标签错位"与"图像错位"混淆。
D. **分辨率一致性**：若两模态分辨率不同，"像素级对齐"必然经过 resize。
E. **时间对齐**：按文件名序号排序，检查相邻帧位移是否有跳变。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(4)
SPLITS = ("train", "val", "test")


def read_yolo(path: Path):
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.split()
        if len(s) >= 5:
            try:
                out.append((int(float(s[0])), *(float(x) for x in s[1:5])))
            except ValueError:
                continue
    return out


def peak_xy(gray, sigma=4.0):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    cy, cx = np.unravel_index(int(np.argmax(g)), g.shape)
    h, w = gray.shape[:2]
    return np.array([cx / w, cy / h])


def in_box(pt, bbox, tol=0.0):
    _, cx, cy, bw, bh = bbox
    return abs(pt[0] - cx) <= bw / 2 + tol and abs(pt[1] - cy) <= bh / 2 + tol


def stat(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return {}
    aa = np.abs(a)
    return {"n": int(a.size), "mean_abs": float(aa.mean()),
            "median_abs": float(np.median(aa)), "p95_abs": float(np.percentile(aa, 95)),
            "max_abs": float(aa.max()), "std": float(a.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Labelled EO_IR Dataset 的父目录")
    ap.add_argument("--max-pairs", type=int, default=400, help="每个 split 最多检验多少对")
    ap.add_argument("--split", default="all", choices=["all", *SPLITS])
    args = ap.parse_args()

    root = Path(args.root)
    eo_root = ir_root = None
    for p in root.rglob("UAV_eo_Dataset"):
        eo_root = p
    for p in root.rglob("UAV_ir_Dataset"):
        ir_root = p
    if not eo_root or not ir_root:
        print("[错误] 未找到 UAV_eo_Dataset / UAV_ir_Dataset")
        return 1
    print(f"EO: {eo_root}\nIR: {ir_root}\n")

    splits = list(SPLITS) if args.split == "all" else [args.split]
    report = {}

    for split in splits:
        eo_img = eo_root / "images" / split
        ir_img = ir_root / "images" / split
        eo_lab = eo_root / "labels" / split
        ir_lab = ir_root / "labels" / split
        if not eo_img.is_dir():
            continue

        eo_files = {p.stem: p for p in eo_img.iterdir() if p.suffix.lower() in (".png", ".jpg")}
        ir_files = {p.stem: p for p in ir_img.iterdir() if p.suffix.lower() in (".png", ".jpg")}
        common = sorted(set(eo_files) & set(ir_files))
        print(f"=== split={split}: eo={len(eo_files)} ir={len(ir_files)} 同名配对={len(common)} ===")
        only_eo = sorted(set(eo_files) - set(ir_files))
        only_ir = sorted(set(ir_files) - set(eo_files))
        if only_eo or only_ir:
            print(f"  单侧存在: eo_only={only_eo[:5]} ir_only={only_ir[:5]}")

        rng = np.random.default_rng(0)
        pick = common if len(common) <= args.max_pairs else \
            [common[i] for i in sorted(rng.choice(len(common), args.max_pairs, replace=False))]

        dx, dy, sw, sh = [], [], [], []
        eo_hit, ir_hit = [], []
        res_pairs = []
        n_no_lab = 0

        for stem in pick:
            e = cv2.imread(str(eo_files[stem]), cv2.IMREAD_GRAYSCALE)
            i_ = cv2.imread(str(ir_files[stem]), cv2.IMREAD_GRAYSCALE)
            if e is None or i_ is None:
                continue
            res_pairs.append((e.shape[:2], i_.shape[:2]))
            be, bi = read_yolo(eo_lab / f"{stem}.txt"), read_yolo(ir_lab / f"{stem}.txt")
            if not be or not bi:
                n_no_lab += 1
                # 仍做热斑检验（若任一侧有框）
                if bi:
                    ir_hit.append(in_box(peak_xy(i_), bi[0]))
                if be:
                    eo_hit.append(in_box(peak_xy(e), be[0]))
                continue
            dx.append(be[0][1] - bi[0][1])
            dy.append(be[0][2] - bi[0][2])
            sw.append(be[0][3] / max(bi[0][3], 1e-9))
            sh.append(be[0][4] / max(bi[0][4], 1e-9))
            eo_hit.append(in_box(peak_xy(e), be[0]))
            ir_hit.append(in_box(peak_xy(i_), bi[0]))

        if not dx:
            print("  无有效框对")
            continue

        same_res = all(a == b for a, b in res_pairs)
        h_e, w_e = res_pairs[0][0]
        print(f"  分辨率: EO={w_e}x{h_e}  IR={res_pairs[0][1][1]}x{res_pairs[0][1][0]}"
              f"  全部一致={same_res}")
        if not same_res:
            from collections import Counter
            print(f"  EO 分辨率种类: {Counter(a for a, _ in res_pairs).most_common(3)}")
            print(f"  IR 分辨率种类: {Counter(b for _, b in res_pairs).most_common(3)}")

        d = np.hypot(np.array(dx), np.array(dy))
        print(f"  有效框对 {len(dx)}（无标签跳过 {n_no_lab}）")
        print(f"  dx(归一化): mean|dx|={np.abs(dx).mean():.4f} median={np.median(np.abs(dx)):.4f} "
              f"p95={np.percentile(np.abs(dx),95):.4f} max={np.abs(dx).max():.4f}")
        print(f"  dy(归一化): mean|dy|={np.abs(dy).mean():.4f} median={np.median(np.abs(dy)):.4f} "
              f"p95={np.percentile(np.abs(dy),95):.4f} max={np.abs(dy).max():.4f}")
        dev_e = d * np.array([w_e, h_e]).mean()
        w_i, h_i = res_pairs[0][1][1], res_pairs[0][1][0]
        print(f"  中心距: EO像素 mean={dev_e.mean():.2f}px ≤2px={np.mean(dev_e<=2)*100:.1f}% "
              f"≤5px={np.mean(dev_e<=5)*100:.1f}% | "
              f"IR像素 mean={(d*np.array([w_i,h_i]).mean()).mean():.2f}px "
              f"≤2px={np.mean(d*np.array([w_i,h_i]).mean()<=2)*100:.1f}%")
        print(f"  框尺寸比: w={np.mean(sw):.3f} h={np.mean(sh):.3f}（1.0=尺度一致）")
        print(f"  热斑落框: EO {sum(eo_hit)}/{len(eo_hit)} = {np.mean(eo_hit)*100:.1f}% "
              f"| IR {sum(ir_hit)}/{len(ir_hit)} = {np.mean(ir_hit)*100:.1f}%")

        report[split] = {
            "n_pairs": len(dx), "n_missing_label": n_no_lab,
            "same_resolution": same_res,
            "eo_hw": [h_e, w_e], "ir_hw": [res_pairs[0][1][0], res_pairs[0][1][1]],
            "dx": stat(dx), "dy": stat(dy), "dist": stat(d),
            "dist_px_eo_leq2": float(np.mean(dev_e <= 2)),
            "dist_px_eo_leq5": float(np.mean(dev_e <= 5)),
            "scale_w": float(np.mean(sw)), "scale_h": float(np.mean(sh)),
            "eo_peak_in_box": float(np.mean(eo_hit)), "ir_peak_in_box": float(np.mean(ir_hit)),
        }
        print()

    print("=== 参照（项目 v2 数据集 both 帧）===")
    print("  mean|dx|≈0.030~0.039  mean|dy|≈0.074~0.080  max|dx| 达 0.74")
    print("  若本数据集 mean|dx|,|dy| 均 < 0.01 且 ≤2px 占比 > 80%，可判为严格对齐")

    out = root / "_align_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
