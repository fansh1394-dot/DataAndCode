# -*- coding: utf-8 -*-
"""融合有效性完整分析：把 B 阶段数据组织成可写进论文的证据链。

回答三个层层递进的问题：
  Q1 融合头是否真的优于任一单路？（增益存在性）
  Q2 融合增益来自 IR 还是 RGB？（信息来源）
  Q3 下降是「信息缺失」还是「分布外伪影」？（机制排除）
"""
import json
from pathlib import Path

EV = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest/_test_eval")
KF = "metrics/fuse/metrics/mAP50-95(B)"
KR = "metrics/rgb/metrics/mAP50-95(B)"
KI = "metrics/ir/metrics/mAP50-95(B)"


def load(tag):
    p = EV / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"]


def block(name):
    base = load(f"{name}_test")
    if base is None:
        print(f"\n--- {name}: 基准缺失 ---")
        return
    f, r, i = base[KF], base[KR], base[KI]
    print(f"\n{'=' * 62}\n{name}  基准 fuse={f:.4f}  rgb={r:.4f}  ir={i:.4f}")
    print(f"{'=' * 62}")

    # Q1
    print("\n[Q1] 融合增益存在性（fuse 相对单路头的优势）")
    print(f"  fuse − rgb = {f - r:+.4f}")
    print(f"  fuse − ir  = {f - i:+.4f}")
    print(f"  fuse 相对最优单路 ({'ir' if i > r else 'rgb'}) 的净增益 = {f - max(r, i):+.4f}")

    # Q2
    print("\n[Q2] 信息来源（置零后 fuse 变化）")
    rows = []
    for suf, label in [
        ("_test_zir", "IR 置黑(0)"),
        ("_test_zirg", "IR 置中灰(128)"),
        ("_test_zrgb", "RGB 置黑(0)"),
    ]:
        m = load(f"{name}{suf}")
        if m is None:
            print(f"  {label:<18} 未评测")
            continue
        d = m[KF] - f
        rows.append((label, m[KF], d))
        print(f"  {label:<18} fuse={m[KF]:.4f}  Δ={d:+.4f}")

    if len(rows) >= 3:
        d_ir0 = rows[0][2]
        d_irg = rows[1][2]
        d_rgb = rows[2][2]
        print(f"\n  → IR 依赖度 {abs(d_ir0):.4f}  vs  RGB 依赖度 {abs(d_rgb):.4f}", end="  ")
        print("IR 更重要" if abs(d_ir0) > abs(d_rgb) else "RGB 更重要")

        # Q3
        print("\n[Q3] 机制排除：下降来自信息缺失还是分布外伪影？")
        gap = abs(d_ir0) - abs(d_irg)
        print(f"  |ΔIR置黑| − |ΔIR置中灰| = {gap:+.4f}")
        if abs(gap) < 0.05:
            print("  两者接近 → 下降源于「IR 信息缺失」本身，")
            print("  而非黑块造成的分布外偏移。这一条排除了最强的替代解释。")
        else:
            print("  两者差异较大 → 存在分布外伪影混淆，需进一步排查")


if __name__ == "__main__":
    for n in ("mdp150_M0_concat_s0", "mdp150_M0CGAFusion_s0"):
        block(n)
