# -*- coding: utf-8 -*-
"""盘点服务器上已完成的训练：模态、配置、数据、完成轮数与最佳指标。"""
from pathlib import Path

import yaml

PROJ = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
roots = [PROJ / "YOLO26/ResTest"] + sorted(PROJ.glob("YOLO*/ResTest")) + [PROJ / "runs/detect"]


def best_metric(d: Path):
    f = d / "results.csv"
    if not f.exists():
        return None, None
    lines = f.read_text().splitlines()
    if len(lines) < 2:
        return 0, None
    head = [h.strip() for h in lines[0].split(",")]
    idx = {h: i for i, h in enumerate(head)}
    key = "metrics/mAP50-95(B)"
    if key not in idx:
        return len(lines) - 1, None
    vals = []
    for ln in lines[1:]:
        p = ln.split(",")
        try:
            vals.append(float(p[idx[key]]))
        except Exception:
            pass
    return len(lines) - 1, (round(max(vals), 4) if vals else None)


rows, seen = [], set()
for root in roots:
    if not root.exists():
        continue
    for args in root.glob("*/args.yaml"):
        d = args.parent
        if d.name.startswith("_"):
            continue
        try:
            a = yaml.safe_load(args.read_text())
        except Exception:
            continue
        name = a.get("name") or d.name
        if str(d) in seen:
            continue
        seen.add(str(d))
        n, best = best_metric(d)
        model = str(a.get("model", ""))
        rows.append({
            "dir": str(d.relative_to(PROJ)),
            "name": name,
            "single": a.get("single_modal"),
            "modality": a.get("modality"),
            "cfg": Path(model).name if model else "",
            "data": Path(str(a.get("data", ""))).name,
            "ep": a.get("epochs"),
            "done": n,
            "best": best,
            "seed": a.get("seed"),
            "batch": a.get("batch"),
        })

rows.sort(key=lambda r: r["dir"])
for r in rows:
    print(f"{r['dir']:58s} single={str(r['single']):7s} mod={str(r['modality']):6s} "
          f"cfg={r['cfg'][:34]:34s} data={r['data'][:18]:18s} "
          f"ep={r['ep']}/{r['done']} best={r['best']} s={r['seed']} b={r['batch']}")
print("total", len(rows))
