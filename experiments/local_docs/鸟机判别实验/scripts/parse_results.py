# -*- coding: utf-8 -*-
"""汇总 E28 results.csv：各分支精度、最佳 epoch、融合增益。"""
import csv
import sys

p = sys.argv[1] if len(sys.argv) > 1 else (
    "/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/"
    "ResTest/mdp150_M0BD_nc2_s0/results.csv"
)
rows = list(csv.DictReader(open(p)))
cols = [
    ("main", "metrics/mAP50(B)", "metrics/mAP50-95(B)"),
    ("rgb", "metrics/rgb/metrics/mAP50(B)", "metrics/rgb/metrics/mAP50-95(B)"),
    ("ir", "metrics/ir/metrics/mAP50(B)", "metrics/ir/metrics/mAP50-95(B)"),
    ("fuse", "metrics/fuse/metrics/mAP50(B)", "metrics/fuse/metrics/mAP50-95(B)"),
    ("fuse_ir", "metrics/fuse_ir/metrics/mAP50(B)", "metrics/fuse_ir/metrics/mAP50-95(B)"),
]
print(f"epochs={len(rows)}  last_ep={rows[-1]['epoch']}")
print("ep " + "  ".join(f"{n}(50/50-95)" for n, _, _ in cols))
step = max(len(rows) // 10, 1)
for r in rows[::step]:
    print(r["epoch"].rjust(3), "  ".join(f"{float(r[a]):.3f}/{float(r[b]):.3f}"
                                         for _, a, b in cols))
r = rows[-1]
print("last", r["epoch"], "  ".join(f"{float(r[a]):.3f}/{float(r[b]):.3f}"
                                     for _, a, b in cols))
best = max(rows, key=lambda x: float(x["metrics/mAP50-95(B)"]))
print(f"best mAP50-95={float(best['metrics/mAP50-95(B)']):.4f} @ep{best['epoch']}")
for k in ("metrics/fusion_gain_over_rgb", "metrics/fusion_gain_over_ir",
          "metrics/subset/fuse/both/metrics/mAP50-95(B)",
          "metrics/subset/fuse_ir/both/metrics/mAP50-95(B)"):
    if k in r:
        print(k, r[k])
