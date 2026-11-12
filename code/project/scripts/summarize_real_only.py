# -*- coding: utf-8 -*-
"""对比全量 test 与纯真实帧 test 的指标，量化合成数据带来的虚高。"""
import json
from pathlib import Path

EV = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest/_test_eval")
KF = "metrics/fuse/metrics/mAP50-95(B)"
KR = "metrics/rgb/metrics/mAP50-95(B)"
KI = "metrics/ir/metrics/mAP50-95(B)"


def g(tag, key):
    p = EV / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"].get(key)


RUNS = [
    ("mdp150_M0_concat_s0", "M0_concat_s0"),
    ("mdp150_M0_concat_s1", "M0_concat_s1"),
    ("mdp150_M0_concat_s2", "M0_concat_s2"),
    ("mdp150_M0CGAFusion_s0", "CGAFusion_s0"),
]

print("fuse mAP50-95")
print("%-16s %8s %8s %9s" % ("组", "全量", "纯真实", "Δ虚高"))
deltas = []
for tag, name in RUNS:
    a = g(f"{tag}_test", KF)
    b = g(f"{tag}_test_real", KF)
    if a is None or b is None:
        print("%-16s %8s %8s %9s" % (name, "NA" if a is None else f"{a:.4f}",
                                     "NA" if b is None else f"{b:.4f}", "-"))
        continue
    deltas.append(a - b)
    print("%-16s %8.4f %8.4f %+9.4f" % (name, a, b, a - b))

if deltas:
    print("\n平均虚高 = %+.4f  (范围 %+.4f ~ %+.4f)"
          % (sum(deltas) / len(deltas), min(deltas), max(deltas)))

print("\n分路对照（M0_concat_s0）")
print("%-10s %8s %8s %9s" % ("分支", "全量", "纯真实", "Δ"))
for key, label in ((KF, "fuse"), (KR, "rgb"), (KI, "ir")):
    a = g("mdp150_M0_concat_s0_test", key)
    b = g("mdp150_M0_concat_s0_test_real", key)
    if a is not None and b is not None:
        print("%-10s %8.4f %8.4f %+9.4f" % (label, a, b, a - b))

print("\n纯真实帧上的种子方差")
vals = []
for tag, name in RUNS[:3]:
    v = g(f"{tag}_test_real", KF)
    if v is not None:
        vals.append(v)
        print("  %-16s %.4f" % (name, v))
if len(vals) >= 2:
    import statistics
    print("  均值=%.4f  σ=%.4f  2σ=%.4f"
          % (statistics.mean(vals), statistics.stdev(vals), 2 * statistics.stdev(vals)))
