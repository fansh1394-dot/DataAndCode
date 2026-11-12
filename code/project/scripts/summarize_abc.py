# -*- coding: utf-8 -*-
"""汇总 A/B/C 三阶段的评测结果。

用法（本地即可运行，通过 _ssh.py 拉数据；或直接在服务器上跑）：
    python scripts/summarize_abc.py

读 YOLO26/ResTest/_test_eval/*.json，输出三张表：
  A) 融合算子组：concat / CGAFusion / PIM / CMCE 的 test 指标
  B) 模态置零：IR 置黑 / IR 置中灰 / RGB 置黑 相对原样的下降
  C) 种子方差：M0_concat seed0/1/2 的离散度
"""
import json
import statistics
from pathlib import Path

import yaml

PROJ = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
EVAL_DIR = PROJ / "YOLO26/ResTest/_test_eval"

# results.csv 列名（无前缀的 fuse 指标在 metrics/ 下，带前缀的在 metrics/fuse/ 下）
K_FUSE = "metrics/fuse/metrics/mAP50-95(B)"
K_RGB = "metrics/rgb/metrics/mAP50-95(B)"
K_IR = "metrics/ir/metrics/mAP50-95(B)"


def load(tag):
    p = EVAL_DIR / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"]


def row(name, m, base=None):
    if m is None:
        return f"  {name:<38} （未评测）"
    fuse, rgb, ir = m.get(K_FUSE), m.get(K_RGB), m.get(K_IR)
    s = f"  {name:<38} fuse={fuse:.4f} rgb={rgb:.4f} ir={ir:.4f} gain={fuse - rgb:+.4f}"
    if base is not None and fuse is not None:
        s += f"  Δvs基准={fuse - base:+.4f}"
    return s


def table_a():
    print("\n=== A 融合算子组（test，同口径）===")
    base = load("mdp150_M0_concat_s0_test")
    b = base.get(K_FUSE) if base else None
    for tag, label in [
        ("mdp150_M0_concat_s0_test", "M0_concat（基线）"),
        ("mdp150_M0CGAFusion_s0_test", "CGAFusion"),
        ("mdp150_M0PIM_s0_test", "PIM"),
        ("mdp150_M0CMCE_s0_test", "CMCE"),
    ]:
        print(row(label, load(tag), b if tag != "mdp150_M0_concat_s0_test" else None))
    print("  （参考）M0mask / M0maskDist 已在 B3 消融中单独报告")


def table_b():
    print("\n=== B 模态置零敏感度（test）===")
    for name in ("mdp150_M0_concat_s0", "mdp150_M0CGAFusion_s0"):
        print(f"-- {name}")
        base = load(f"{name}_test")
        if base is None:
            print("  （基准未评测，跳过）")
            continue
        bf = base[K_FUSE]
        for suffix, label in [
            ("_test_zir", "IR 置黑(0)"),
            ("_test_zirg", "IR 置中灰(128)"),
            ("_test_zrgb", "RGB 置黑(0)"),
        ]:
            m = load(f"{name}{suffix}")
            if m is None:
                print(f"  {label:<20} （未评测）")
                continue
            d = m[K_FUSE] - bf
            print(f"  {label:<20} fuse={m[K_FUSE]:.4f}  Δ={d:+.4f}  "
                  f"{'显著依赖' if abs(d) > 0.02 else '无实质依赖'}")
        print(f"  基准原样 fuse={bf:.4f}")


def table_c():
    print("\n=== C 种子方差（val 末轮，来自 results.csv）===")
    vals = {}
    for seed in (0, 1, 2):
        name = f"mdp150_M0_concat_s{seed}"
        p = PROJ / "YOLO26/ResTest" / name / "results.csv"
        if not p.exists():
            vals[seed] = None
            continue
        rows = list(__import__("csv").DictReader(open(p, encoding="utf-8")))
        tail = [float(r[K_FUSE]) for r in rows[-10:]]
        vals[seed] = sum(tail) / len(tail)
        print(f"  seed={seed} 末10轮均值={vals[seed]:.4f}")

    got = [v for v in vals.values() if v is not None]
    if len(got) >= 2:
        sd = statistics.stdev(got) if len(got) > 1 else 0.0
        rng = max(got) - min(got)
        print(f"\n  样本数={len(got)}  均值={statistics.mean(got):.4f}  "
              f"标准差 σ={sd:.4f}  极差={rng:.4f}")
        print(f"  判定门槛（2σ）={2 * sd:.4f}")
        print("  用法：任何组间差值的绝对值 < 2σ 应视为「不显著」。")
    else:
        print("\n  种子未跑完，暂时无法给出 σ")


if __name__ == "__main__":
    print(f"评测目录: {EVAL_DIR}")
    table_a()
    table_b()
    table_c()
