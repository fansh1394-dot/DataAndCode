# -*- coding: utf-8 -*-
"""C 阶段收尾：用三种子的 test 结果定 σ，并重判所有组间差异是否显著。

为什么用 test 而不是 val：
  论文所有数字都走 test 口径，σ 必须和它们同源。val 最优轮还受 best.pt
  选取策略影响，会低估真实波动。
"""
import json
import statistics
from pathlib import Path

EV = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest/_test_eval")
K = "metrics/fuse/metrics/mAP50-95(B)"


def fuse(tag):
    p = EV / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"][K]


def main():
    seeds = {}
    for s in ("s0", "s1", "s2"):
        v = fuse(f"mdp150_M0_concat_{s}_test")
        if v is not None:
            seeds[s] = v

    print("=== 种子组 test fuse mAP50-95 ===")
    for k, v in seeds.items():
        print(f"  M0_concat_{k}  {v:.4f}")

    vals = list(seeds.values())
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals)          # 样本标准差（n-1），n=3
    print(f"\n  均值 μ = {mu:.4f}")
    print(f"  标准差 σ = {sd:.4f}   （n={len(vals)}, ddof=1）")
    print(f"  极差     = {max(vals) - min(vals):.4f}")
    print(f"  判定门槛 2σ = {2 * sd:.4f}")

    print("\n=== 组间差异显著性（阈值 |Δ| > 2σ 才显著）===")
    groups = [
        ("mdp150_M0CGAFusion_s0_test", "CGAFusion"),
        ("mdp150_M0PIM_s0_test", "PIM"),
        ("mdp150_M0CMCE_s0_test", "CMCE"),
        ("mdp150_M0mask_s0_test", "M0mask(屏蔽)"),
        ("mdp150_M0maskDist_s0_test", "M0maskDist(蒸馏)"),
    ]
    thr = 2 * sd
    print(f"  {'组':<22} {'fuse':>7} {'Δ vs μ':>9}  判定")
    for tag, name in groups:
        v = fuse(tag)
        if v is None:
            print(f"  {name:<22} 未评测")
            continue
        d = v - mu
        verdict = "显著" if abs(d) > thr else "不显著（噪声内）"
        print(f"  {name:<22} {v:7.4f} {d:+9.4f}  {verdict}")

    print(f"\n  参考：基线上界 = μ + 2σ = {mu + thr:.4f}")
    print(f"        基线下界 = μ - 2σ = {mu - thr:.4f}")
    print("\n  解读：任何组的 test fuse 落在 [下界, 上界] 内，都不能声称"
          "\n        与基线有真实差异；只有越界才可下结论。")


if __name__ == "__main__":
    main()
