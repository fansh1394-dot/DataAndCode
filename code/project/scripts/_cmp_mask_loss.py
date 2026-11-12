# -*- coding: utf-8 -*-
"""对比 M0mask 与 M0 基线的三头 loss 曲线，验证 RGB 屏蔽是否真正生效。

判据（来自 train_full.py 日志中的提示）：
    屏蔽生效时 fuse 头的 loss 应明显高于 rgb 头；若两者接近，
    说明屏蔽没有命中融合 Concat，或融合分支本就在读 IR。
"""
import csv
import sys

BASE = "YOLO26/ResTest/mdp150_M0_concat_s0/results.csv"
MASK = "YOLO26/ResTest/mdp150_M0mask_s0/results.csv"

COLS = [
    ("train/cls_loss_rgb", "rgb"),
    ("train/cls_loss_ir", "ir"),
    ("train/cls_loss_fuse", "fuse"),
]


def load(path):
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    except FileNotFoundError:
        print(f"[缺失] {path}")
        return []
    out = []
    for r in rows:
        r = {k.strip(): v for k, v in r.items()}
        try:
            out.append({k: float(r[k]) for k, _ in COLS} | {"epoch": int(float(r["epoch"]))})
        except (KeyError, ValueError):
            continue
    return out


def show(title, rows, epochs):
    print(f"\n{title}")
    print(f"{'轮次':<6}{'cls_rgb':>10}{'cls_ir':>10}{'cls_fuse':>11}{'fuse-rgb':>11}")
    print("-" * 48)
    for e in epochs:
        m = next((r for r in rows if r["epoch"] == e), None)
        if m is None:
            continue
        d = m["train/cls_loss_fuse"] - m["train/cls_loss_rgb"]
        print(f"ep{e:<4}{m['train/cls_loss_rgb']:>10.4f}{m['train/cls_loss_ir']:>10.4f}"
              f"{m['train/cls_loss_fuse']:>11.4f}{d:>+11.4f}")


base = load(BASE)
mask = load(MASK)
print(f"M0 基线可用轮次: {base[-1]['epoch'] if base else '-'}")
print(f"M0mask 可用轮次: {mask[-1]['epoch'] if mask else '-'}")

eps = [0, 1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, 120, 140, 149]
show("=== M0 基线（无屏蔽）===", base, eps)
show("=== M0mask（屏蔽 p=0.3）===", mask, eps)

if base and mask:
    common = sorted({r["epoch"] for r in base} & {r["epoch"] for r in mask})
    if common:
        e = common[-1]
        b = next(r for r in base if r["epoch"] == e)
        m = next(r for r in mask if r["epoch"] == e)
        print(f"\n=== 同轮次 ep{e} 对照 ===")
        for k, label in COLS:
            print(f"  {label:<5} M0={b[k]:.4f}  M0mask={m[k]:.4f}  Δ={m[k]-b[k]:+.4f}")
