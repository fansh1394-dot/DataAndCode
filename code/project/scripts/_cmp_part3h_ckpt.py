# -*- coding: utf-8 -*-
"""对比 part3h 各检查点在 test 集的分支指标，判断 best.pt 是否可信。"""
import json
import sys
from pathlib import Path

BASE = Path("YOLO26/ResTest/_test_eval")
TAGS = [("best(ep149)", "mdp150_M3part3h_test.json"), ("epoch140", "chk_part3h_ep140.json")]

print(f"{'权重':<12}{'mAP50':>9}{'mAP50-95':>11}{'rgb':>9}{'ir':>9}{'fuse':>9}")
print("-" * 59)
for label, fn in TAGS:
    p = BASE / fn
    if not p.is_file():
        print(f"{label:<12}  (缺失: {fn})")
        continue
    m = json.loads(p.read_text(encoding="utf-8"))["metrics"]
    print(f"{label:<12}{m.get('metrics/mAP50(B)', 0):>9.4f}"
          f"{m.get('metrics/mAP50-95(B)', 0):>11.4f}"
          f"{m.get('metrics/rgb/metrics/mAP50-95(B)', 0):>9.4f}"
          f"{m.get('metrics/ir/metrics/mAP50-95(B)', 0):>9.4f}"
          f"{m.get('metrics/fuse/metrics/mAP50-95(B)', 0):>9.4f}")
