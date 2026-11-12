# -*- coding: utf-8 -*-
"""列出所有训练组的 val 最优轮指标，供交接文档引用。"""
import csv
from pathlib import Path

RT = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest")
K = "metrics/fuse/metrics/mAP50-95(B)"
KR = "metrics/rgb/metrics/mAP50-95(B)"
KI = "metrics/ir/metrics/mAP50-95(B)"

RUNS = [
    "mdp150_M0_concat_s0", "mdp150_M0_concat_s1", "mdp150_M0_concat_s2",
    "mdp150_M0mask_s0", "mdp150_M0maskDist_s0",
    "mdp150_M0CGAFusion_s0", "mdp150_M0PIM_s0", "mdp150_M0CMCE_s0",
    "mdp150_M3part3h_s0", "mdp150_M1_eo_s0", "mdp150_M2_ir_s0", "mdp150_M3_dual_s0",
]

print("%-28s %4s %7s %7s %7s" % ("run", "ep", "fuse", "rgb", "ir"))
for r in RUNS:
    p = RT / r / "results.csv"
    if not p.exists():
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    keys = rows[0].keys()
    # 三头改造前的组（M1/M2/M3part3h）没有 metrics/fuse/* 列，退回无前缀列
    kf = K if K in keys else "metrics/mAP50-95(B)"
    kr = KR if KR in keys else "-"
    ki = KI if KI in keys else "-"
    f = [float(x[kf]) for x in rows]
    bi = max(range(len(f)), key=lambda j: f[j])
    fmt = lambda k: "%7.4f" % float(rows[bi][k]) if k in keys else "      -"
    print("%-28s %4d %7.4f %s %s" % (
        r, int(float(rows[bi]["epoch"])), f[bi], fmt(kr), fmt(ki)))
