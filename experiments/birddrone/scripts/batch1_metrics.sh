#!/bin/bash
R=/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest
for n in bdc3_rgbonly_yolo11n_s0 bdc3_ironly_yolo26n_s0 bdc3_nc3_unreg_s0 bdc3_nc3_s1 bdc3_nc3_s0_150; do
  d=$R/$n
  [ -f "$d/results.csv" ] || { echo "$n 无 csv"; continue; }
  echo "===== $n ====="
  head -1 "$d/results.csv"
  echo "--- best mAP50-95 行 ---"
  python3 - "$d/results.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("空"); sys.exit()
key = [k for k in rows[0] if 'mAP50-95' in k][0]
best = max(rows, key=lambda r: float(r[key] or 0))
last = rows[-1]
cols = ['epoch', key] + [k for k in rows[0] if 'mAP50(' in k or 'precision' in k.lower() or 'recall' in k.lower()]
print("best:", {c: best[c] for c in cols})
print("last:", {c: last[c] for c in cols})
PY
  [ -f "$d/metrics_per_class.csv" ] && { echo "--- 类指标 ---"; cat "$d/metrics_per_class.csv"; }
done
