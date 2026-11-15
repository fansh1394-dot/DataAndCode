#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总第一批：results.csv 最佳行 + eval_split 独立复评，输出对比表。"""
import csv
import glob
import json
import os

R = "/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest"
EV = "/root/rivermind-fs/birddrone/batch1/eval_avidsrc"
RUNS = [
    ("A1_rgbonly", "bdc3_rgbonly_yolo11n_s0", "RGB-only YOLO11n（COCO 预训练）"),
    ("A2_ironly", "bdc3_ironly_yolo26n_s0", "IR-only YOLO26n（从零）"),
    ("A0_nc3_s0", "bdc3_nc3_s0_150", "双模态 concat seed0（150ep）"),
    ("B1_unreg_s0", "bdc3_nc3_unreg_s0", "双模态 concat 未配准 IR（150ep）"),
    ("C1_nc3_s1", "bdc3_nc3_s1", "双模态 concat seed1（150ep）"),
]

def f(v):
    try:
        return float(v)
    except Exception:
        return None

def best_row(p):
    rows = [r for r in csv.DictReader(open(p)) if f(r.get("metrics/mAP50-95(B)")) is not None]
    if not rows:
        return None, 0
    b = max(rows, key=lambda r: f(r["metrics/mAP50-95(B)"]))
    return b, len(rows)

print("| 任务 | 说明 | ep(best/总) | AVID val mAP50-95 | mAP50 | P | R | RGB 分支 | IR 分支 | 融合头 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for tag, name, desc in RUNS:
    p = os.path.join(R, name, "results.csv")
    b, n = best_row(p)
    if not b:
        print(f"| {tag} | {desc} | - | - | - | - | - | - | - | - |"); continue
    g = lambda k: b.get(k, "")
    rg = g("metrics/rgb/metrics/mAP50-95(B)")
    ir = g("metrics/ir/metrics/mAP50-95(B)")
    fu = g("metrics/fuse/metrics/mAP50-95(B)")
    fmt = lambda x: f"{float(x):.3f}" if x not in ("", None) else "—"
    print(f"| {tag} | {desc} | {int(float(b['epoch']))}/{n} | "
          f"{float(b['metrics/mAP50-95(B)']):.3f} | {float(b['metrics/mAP50(B)']):.3f} | "
          f"{float(b['metrics/precision(B)']):.3f} | {float(b['metrics/recall(B)']):.3f} | "
          f"{fmt(rg)} | {fmt(ir)} | {fmt(fu)} |")

print("\n## 独立复评（eval_split，bd_nc3 val，best.pt）\n")
print("| 任务 | mAP50-95 | mAP50 | P | R | RGB | IR | fuse | fuse_ir |")
print("|---|---|---|---|---|---|---|---|---|")
for tag, name, _ in RUNS:
    p = os.path.join(EV, f"{tag}.json")
    if not os.path.exists(p):
        print(f"| {tag} | — | — | — | — | — | — | — | — |"); continue
    d = json.load(open(p))
    m = d.get("metrics") or d
    def pick(*keys):
        for k in keys:
            if k in m and m[k] is not None:
                return f"{float(m[k]):.3f}"
        return "—"
    print(f"| {tag} | " + " | ".join([
        pick("metrics/mAP50-95(B)", "mAP50-95"),
        pick("metrics/mAP50(B)", "mAP50"),
        pick("metrics/precision(B)", "precision"),
        pick("metrics/recall(B)", "recall"),
        pick("metrics/rgb/metrics/mAP50-95(B)"),
        pick("metrics/ir/metrics/mAP50-95(B)"),
        pick("metrics/fuse/metrics/mAP50-95(B)"),
        pick("metrics/fuse_ir/metrics/mAP50-95(B)"),
    ]) + " |")

print("\n## 类指标（训练期 val，best.pt）\n")
print("| 任务 | bird 50-95 | bird mAP50 | bird R | drone 50-95 | drone mAP50 | drone R | other 50-95 |")
print("|---|---|---|---|---|---|---|---|")
for tag, name, _ in RUNS:
    p = os.path.join(R, name, "metrics_per_class.csv")
    if not os.path.exists(p):
        print(f"| {tag} | — | — | — | — | — | — | — |"); continue
    rows = {r["Class"]: r for r in csv.DictReader(open(p))}
    def g(c, k):
        return f"{float(rows[c][k]):.3f}" if c in rows else "—"
    print(f"| {tag} | {g('bird','mAP50-95')} | {g('bird','mAP50')} | {g('bird','Recall')} | "
          f"{g('drone','mAP50-95')} | {g('drone','mAP50')} | {g('drone','Recall')} | {g('other','mAP50-95')} |")
