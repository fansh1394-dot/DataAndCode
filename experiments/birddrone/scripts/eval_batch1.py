#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一批同口径对比：从 results.csv 提取主指标 + 各分支指标，输出 JSON 与 Markdown。"""
import csv
import json
import os

R = "/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest"
RUNS = {
    "A1_rgbonly": "bdc3_rgbonly_yolo11n_s0",
    "A2_ironly": "bdc3_ironly_yolo26n_s0",
    "A0_nc3_s0": "bdc3_nc3_s0_150",
    "B1_unreg_s0": "bdc3_nc3_unreg_s0",
    "C1_nc3_s1": "bdc3_nc3_s1",
}
KEYS = [
    ("mAP50-95", "metrics/mAP50-95(B)"),
    ("mAP50", "metrics/mAP50(B)"),
    ("P", "metrics/precision(B)"),
    ("R", "metrics/recall(B)"),
    ("rgb_mAP50-95", "metrics/rgb/metrics/mAP50-95(B)"),
    ("rgb_mAP50", "metrics/rgb/metrics/mAP50(B)"),
    ("ir_mAP50-95", "metrics/ir/metrics/mAP50-95(B)"),
    ("ir_mAP50", "metrics/ir/metrics/mAP50(B)"),
    ("fuse_mAP50-95", "metrics/fuse/metrics/mAP50-95(B)"),
    ("fuse_ir_mAP50-95", "metrics/fuse_ir/metrics/mAP50-95(B)"),
]


def f(v):
    try:
        return float(v)
    except Exception:
        return None


def main():
    out = {}
    for tag, name in RUNS.items():
        d = os.path.join(R, name)
        csvp = os.path.join(d, "results.csv")
        if not os.path.exists(csvp):
            out[tag] = {"run": name, "error": "no results.csv"}
            continue
        rows = list(csv.DictReader(open(csvp)))
        rows = [r for r in rows if f(r.get("metrics/mAP50-95(B)")) is not None]
        if not rows:
            out[tag] = {"run": name, "error": "empty"}
            continue
        best = max(rows, key=lambda r: f(r["metrics/mAP50-95(B)"]))
        last = rows[-1]
        rec = {"run": name, "epochs": len(rows),
               "best_epoch": int(float(best["epoch"])), "last_epoch": int(float(last["epoch"]))}
        for k, col in KEYS:
            if col in best:
                rec[k + "_best"] = round(f(best[col]), 4)
                rec[k + "_last"] = round(f(last[col]), 4)
        # 收敛性：最后 10 轮 mAP50-95 的波动
        tail = [f(r["metrics/mAP50-95(B)"]) for r in rows[-10:]]
        rec["tail10_std"] = round((sum((x - sum(tail) / len(tail)) ** 2 for x in tail) / len(tail)) ** 0.5, 4)
        rec["tail10_max"] = round(max(tail), 4)
        out[tag] = rec

    # 类指标
    for tag, name in RUNS.items():
        p = os.path.join(R, name, "metrics_per_class.csv")
        if os.path.exists(p):
            out[tag]["per_class"] = list(csv.DictReader(open(p)))

    with open("/root/rivermind-fs/birddrone/batch1/eval_batch1.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    for tag, r in out.items():
        print("=====", tag, r.get("run"))
        if "error" in r:
            print("  ", r["error"]); continue
        print(f"  epochs={r['epochs']} best_ep={r['best_epoch']} last_ep={r['last_epoch']} tail10_std={r['tail10_std']}")
        print(f"  主指标 best: mAP50-95={r.get('mAP50-95_best')} mAP50={r.get('mAP50_best')} P={r.get('P_best')} R={r.get('R_best')}")
        print(f"  主指标 last: mAP50-95={r.get('mAP50-95_last')} mAP50={r.get('mAP50_last')}")
        if r.get("rgb_mAP50-95_best") is not None:
            print(f"  分支 best: rgb={r.get('rgb_mAP50-95_best')} ir={r.get('ir_mAP50-95_best')} fuse={r.get('fuse_mAP50-95_best')} fuse_ir={r.get('fuse_ir_mAP50-95_best')}")
        for c in r.get("per_class", []):
            print(f"    {c['Class']}: mAP50-95={float(c['mAP50-95']):.3f} mAP50={float(c['mAP50']):.3f} P={float(c['Precision']):.3f} R={float(c['Recall']):.3f}")


if __name__ == "__main__":
    main()
