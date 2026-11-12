# -*- coding: utf-8 -*-
"""查看指定 run 的 val 曲线关键轮次。用法: python _peek_run.py <run_name>"""
import csv
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "mdp150_M0mask_s0"
path = f"YOLO26/ResTest/{name}/results.csv"

rows = list(csv.DictReader(open(path, encoding="utf-8")))
rows = [{k.strip(): v for k, v in r.items()} for r in rows]

key = "metrics/mAP50-95(B)"
print(f"{name}: 共 {len(rows)} 轮")
print(f"{'轮次':<7}{'mAP50-95':>10}{'mAP50':>9}")
print("-" * 27)

best_v, best_e = -1.0, -1
for i, r in enumerate(rows):
    try:
        v = float(r[key])
    except (KeyError, ValueError):
        continue
    if v > best_v:
        best_v, best_e = v, i

for e in (0, 20, 50, 80, 100, 110, 120, 130, 135, 140, 145, 149):
    if e >= len(rows):
        continue
    r = rows[e]
    try:
        print(f"ep{e:<5}{float(r[key]):>10.4f}{float(r['metrics/mAP50(B)']):>9.4f}")
    except (KeyError, ValueError):
        continue

print("-" * 27)
print(f"val 最优: ep{best_e}  mAP50-95={best_v:.4f}")
