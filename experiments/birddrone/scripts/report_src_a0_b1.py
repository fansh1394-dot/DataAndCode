#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印 A0/B1 分源指标：整体/分支，用于定位配准消融差异来自哪一源。"""
import json
import os

EV = "/root/rivermind-fs/birddrone/batch1/eval_src"
rows = []
for tag in ("A0", "B1"):
    for grp in ("avid", "men", "syn"):
        p = os.path.join(EV, f"{tag}_{grp}.json")
        d = json.load(open(p))
        m = d.get("metrics") or d
        def g(k):
            v = m.get(k)
            return float(v) if v is not None else float("nan")
        rows.append((tag, grp, g("metrics/mAP50-95(B)"), g("metrics/mAP50(B)"),
                     g("metrics/rgb/metrics/mAP50-95(B)"),
                     g("metrics/ir/metrics/mAP50-95(B)"),
                     g("metrics/fuse/metrics/mAP50-95(B)")))

print("| 模型 | 子集 | 整体 50-95 | 整体 mAP50 | RGB 分支 | IR 分支 | 融合头 |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} |")

print("\n差值（B1 - A0）：")
for grp in ("avid", "men", "syn"):
    a = [r for r in rows if r[0] == "A0" and r[1] == grp][0]
    b = [r for r in rows if r[0] == "B1" and r[1] == grp][0]
    print(f"  {grp}: 整体 {b[2]-a[2]:+.3f} | RGB {b[4]-a[4]:+.3f} | IR {b[5]-a[5]:+.3f} | 融合 {b[6]-a[6]:+.3f}")
