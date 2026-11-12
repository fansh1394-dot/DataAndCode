# -*- coding: utf-8 -*-
"""S0-V2 对象级频域诊断（零训练成本）。

S0 的帧级指标已证明"全图高频占比"不是极小目标的代理量（与 lt16 存在负相关）。
本脚本改判据为**对象级**：对每个 GT 框，比较框内与同尺寸背景环的高频能量比。

回答两个问题
------------
Q4 极小目标框内的高频占比，是否显著高于其周围背景环？
    显著 -> 频域分离在图像层面对小目标是可分辨的，频域路线有依据。
    不显著 -> 高频主要由背景贡献，频域增强只会放大杂波，路线终止。

Q5 在 P3(stride=8) 上，lt16 目标的特征足迹是否小到无法做局部频域度量？
    side/8 < 1.5 个单元时，局部带分解在特征图上没有空间支撑，
    这与 Q4 的结论共同决定"是否只能在图像级做、不能在特征级做"。

方法
----
- 读原始（未 letterbox）RGB 与 IR 灰度图；
- 带分解：L = 3x3 高斯平滑，H = I - L，ratio = sqrt(sum H^2 / sum I^2)；
- 框内 = GT 框；背景环 = 框外扩 k 倍后的区域减去框内；
- 统计每条 GT 的 delta = ratio_in - ratio_bg，按等效边长分桶聚合；
- 显著性用符号检验（正态近似），不依赖 scipy。

用法
----
    python scripts/diag_spectral_v2.py \
        --root /3250410011/root/rivermind-data/mendeley_split \
        --split test --out /3250410011/root/rivermind-data/spec_diag_v2.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

BINS = [("lt16", 0.0, 16.0), ("b16_32", 16.0, 32.0), ("b32_64", 32.0, 64.0), ("gt64", 64.0, 1e9)]
STRIDE_P3 = 8


def read_gray(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32)


def band_ratio(img: np.ndarray, mask: np.ndarray) -> float:
    """返回 mask 区域内的 sqrt(sum H^2 / sum I^2)，H 为 3x3 高斯高通残差。"""
    lo = cv2.GaussianBlur(img, (3, 3), 0.8)
    hi = img - lo
    sel = mask > 0
    e_all = float((img[sel] ** 2).sum())
    e_hi = float((hi[sel] ** 2).sum())
    if e_all <= 1e-9:
        return float("nan")
    return math.sqrt(max(e_hi, 0.0) / e_all)


def sign_test_p(n_pos: int, n: int) -> float:
    """双侧符号检验（正态近似）。"""
    if n <= 0:
        return float("nan")
    z = abs(n_pos - n / 2) / math.sqrt(n / 4)
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def bucket_of(side: float) -> str:
    for name, lo, hi in BINS:
        if lo <= side < hi:
            return name
    return "gt64"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ring", type=float, default=1.8, help="背景环外扩倍数")
    ap.add_argument("--ring-min", type=int, default=4, help="背景环最小外扩像素")
    ap.add_argument("--out", default="spec_diag_v2.json")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "images" / args.split
    ir_dir = root / "images_ir" / args.split
    lab_dir = root / "labels" / args.split
    print("=" * 78)
    print("S0-V2 对象级频域诊断（框内 vs 背景环）")
    print("=" * 78)
    print(f"  图像: {img_dir}\n  标签: {lab_dir}")

    recs = []
    n_img = n_missing = 0
    for lab in sorted(lab_dir.glob("*.txt")):
        img_p = img_dir / lab.with_suffix(".png").name
        ir_p = ir_dir / lab.with_suffix(".png").name
        rgb = read_gray(img_p)
        ir = read_gray(ir_p)
        if rgb is None or ir is None:
            n_missing += 1
            continue
        n_img += 1
        h, w = rgb.shape
        if ir.shape != rgb.shape:
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)
        for line in lab.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts[:5])
            x1, x2 = (cx - bw / 2) * w, (cx + bw / 2) * w
            y1, y2 = (cy - bh / 2) * h, (cy + bh / 2) * h
            x1, x2 = max(0, int(round(x1))), min(w, int(round(x2)))
            y1, y2 = max(0, int(round(y1))), min(h, int(round(y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            side = math.sqrt((x2 - x1) * (y2 - y1))

            mx = max(int(round((x2 - x1) * args.ring / 2)), args.ring_min)
            my = max(int(round((y2 - y1) * args.ring / 2)), args.ring_min)
            rx1, rx2 = max(0, x1 - mx), min(w, x2 + mx)
            ry1, ry2 = max(0, y1 - my), min(h, y2 + my)

            mask_in = np.zeros((h, w), np.uint8)
            mask_in[y1:y2, x1:x2] = 1
            mask_ring = np.zeros((h, w), np.uint8)
            mask_ring[ry1:ry2, rx1:rx2] = 1
            mask_ring[y1:y2, x1:x2] = 0

            rec = {"bucket": bucket_of(side), "side": round(side, 2), "file": lab.stem}
            for name, im in (("rgb", rgb), ("ir", ir)):
                r_in = band_ratio(im, mask_in)
                r_bg = band_ratio(im, mask_ring)
                rec[f"{name}_in"] = r_in
                rec[f"{name}_bg"] = r_bg
                rec[f"{name}_delta"] = (r_in - r_bg) if (r_in == r_in and r_bg == r_bg) else float("nan")
            rec["p3_cells"] = round(side / STRIDE_P3, 2)
            recs.append(rec)

    print(f"\n  处理帧: {n_img}   缺图: {n_missing}   GT 框: {len(recs)}")
    print(f"  等效边长中位数(全): {np.median([r['side'] for r in recs]):.1f} px")

    def agg(sub, key):
        vals = [r[key] for r in sub if r[key] == r[key]]
        return (float(np.mean(vals)), float(np.median(vals)), len(vals)) if vals else (float("nan"),) * 2 + (0,)

    print("\n" + "=" * 78)
    print("Q4 框内 vs 背景环：高频占比差 delta = ratio_in - ratio_bg")
    print("=" * 78)
    print(f"  {'桶':<8}{'模态':<6}{'n':>5}{'ratio_in':>10}{'ratio_bg':>10}{'delta均值':>11}"
          f"{'delta中位':>11}{'正例占比':>10}{'符号检验p':>11}")
    summary = {}
    for bname, _, _ in BINS:
        sub = [r for r in recs if r["bucket"] == bname]
        if not sub:
            continue
        summary[bname] = {"n": len(sub)}
        for mod in ("rgb", "ir"):
            mi, md, n = agg(sub, f"{mod}_in")
            mb, _, _ = agg(sub, f"{mod}_bg")
            _, mdlt, nd = agg(sub, f"{mod}_delta")
            pos = sum(1 for r in sub if r[f"{mod}_delta"] > 0 and r[f"{mod}_delta"] == r[f"{mod}_delta"])
            p = sign_test_p(pos, nd)
            summary[bname][mod] = {"n": len(sub), "ratio_in": mi, "ratio_bg": mb,
                                   "delta_mean": mdlt, "delta_median": mdlt,
                                   "pos_frac": pos / max(nd, 1), "p": p}
            print(f"  {bname:<8}{mod:<6}{len(sub):>5}{mi:>10.4f}{mb:>10.4f}{md:>11.4f}"
                  f"{mdlt:>11.4f}{100 * pos / max(nd, 1):>9.1f}%{p:>11.4f}")

    print("\n" + "=" * 78)
    print("Q5 lt16 在 P3(stride=8) 上的特征足迹")
    print("=" * 78)
    lt = [r["p3_cells"] for r in recs if r["bucket"] == "lt16"]
    if lt:
        frac = sum(1 for c in lt if c < 1.5) / len(lt)
        summary["p3_footprint"] = {
            "median_cells": float(np.median(lt)),
            "frac_below_1.5_cells": frac,
        }
        print(f"  lt16 目标在 P3 的单元数：中位 {np.median(lt):.2f}，"
              f"<1.5 单元的占比 {100 * frac:.1f}%")
        print("  含义：<1.5 单元时，特征图上的局部带分解没有空间支撑，"
              "只能在图像级做频域度量。")

    # ---- 判读 ----
    print("\n" + "=" * 78)
    print("判读")
    print("=" * 78)
    lt_rgb = summary.get("lt16", {}).get("rgb", {})
    lt_ir = summary.get("lt16", {}).get("ir", {})
    ok_rgb = lt_rgb.get("delta_median", -1) > 0 and lt_rgb.get("p", 1) < 0.05
    ok_ir = lt_ir.get("delta_median", -1) > 0 and lt_ir.get("p", 1) < 0.05
    verdict = {
        "lt16_rgb_sig_positive": ok_rgb,
        "lt16_ir_sig_positive": ok_ir,
        "go": bool(ok_rgb or ok_ir),
    }
    summary["verdict"] = verdict
    print(f"  lt16 RGB 框内高频显著高于背景: {'是' if ok_rgb else '否'}"
          f"  (delta中位 {lt_rgb.get('delta_median', float('nan')):+.4f}, p={lt_rgb.get('p', float('nan')):.4f})")
    print(f"  lt16 IR  框内高频显著高于背景: {'是' if ok_ir else '否'}"
          f"  (delta中位 {lt_ir.get('delta_median', float('nan')):+.4f}, p={lt_ir.get('p', float('nan')):.4f})")
    print(f"  => 频域路线: {'可继续（需按图像级证据重写模块位置与作用域）' if verdict['go'] else '终止（高频由背景主导，框内无显著频域特征）'}")

    out = Path(args.out)
    out.write_text(json.dumps({"summary": summary, "boxes": recs}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
