# -*- coding: utf-8 -*-
"""核对 part3h 续训对收敛曲线的影响：打印关键轮次的 val 指标。"""
import csv
import re

PATH = "YOLO26/ResTest/mdp150_M3part3h_s0/results.csv"

rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
rows = [{k.strip(): v for k, v in r.items()} for r in rows]

overall = [c for c in rows[0] if re.fullmatch(r"metrics/mAP50-95\(B\)", c)]
per_branch = [c for c in rows[0] if re.fullmatch(r"metrics/\w+/metrics/mAP50-95\(B\)", c)]
print("总体列   =", overall)
print("分支列   =", per_branch)
print()

want = [0, 5, 10, 15, 18, 19, 20, 21, 22, 23, 25, 30, 40, 60, 80, 100, 120, 140, 149]
print(f"{'轮次':<6}{'总体':>10}{'rgb':>10}{'ir':>10}{'fuse':>10}")
print("-" * 46)
for e in want:
    if e >= len(rows):
        continue
    r = rows[e]
    def g(col):
        v = r.get(col, "")
        return f"{float(v):.4f}" if v else "  -   "
    b = {c.split("/")[1]: g(c) for c in per_branch}
    print(f"ep{e:<4}{g(overall[0]):>10}{b.get('rgb','-'):>10}"
          f"{b.get('ir','-'):>10}{b.get('fuse','-'):>10}")

# 找 best 轮次
best, best_ep = -1.0, -1
for i, r in enumerate(rows):
    v = r.get(overall[0], "")
    if v and float(v) > best:
        best, best_ep = float(v), i
print()
print(f"val 最优轮次 = ep{best_ep}  mAP50-95 = {best:.4f}")
