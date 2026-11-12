# -*- coding: utf-8 -*-
"""诊断解绑版 CGAFusion 的幅度问题。

现象：解绑版输出 std 0.42，原版 1.22，约 1/3。
原因：原版系数和恒为 3（x·(1+p)+y·(2-p)），
      解绑版 w_x,w_y 初始化时 sigmoid(0)≈0.5，两路合计≈1。
      若 x、y 独立同分布，原版幅度 = 3σ，解绑版 = 0.5·sqrt(2)σ ≈ 0.71σ。

影响：下游 C3k2 的 BN 会自适应，但初始阶段梯度尺度不同，
      可能影响收敛。需要通过实验确认是否需要补偿。
"""
import os
import sys

import torch

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from ultralytics.nn.modules.fusion.cgafusion import CGAFusion  # noqa: E402

torch.manual_seed(0)
x = torch.randn(4, 64, 32, 32)
y = torch.randn(4, 64, 32, 32)
print(f"输入 x std={x.std():.4f}  y std={y.std():.4f}")

for label, kw in [("原版", dict(decouple=False)),
                  ("解绑-无残差", dict(decouple=True, use_residual=False)),
                  ("解绑-带残差", dict(decouple=True, use_residual=True))]:
    m = CGAFusion([64, 64], 64, 8, **kw).eval()
    with torch.no_grad():
        # 绕过后置 conv，直接看融合输出（conv 会再缩一次，掩盖问题）
        xx = m.conv_adjust[0](x)
        yy = m.conv_adjust[1](y)
        initial = xx + yy
        pattn1 = m.sa(initial) + m.ca(initial)
        if not m.decouple:
            p = m.sigmoid(m.pa(initial, pattn1))
            out = initial + p * xx + (1 - p) * yy
        else:
            w_x = m.sigmoid(m.gate_x(pattn1))
            w_y = m.sigmoid(m.gate_y(pattn1))
            out = w_x * xx + w_y * yy
            if m.use_residual:
                out = out + initial
        print(f"  {label:<12} 融合输出 std={out.std():.4f}  "
              f"w_x均值={float(w_x.mean()) if m.decouple else float(p.mean()):.4f}")

print("\n结论：")
print("  原版系数和≈3，解绑版系数和≈1（初始化时），幅度差约 2.1~3 倍。")
print("  修复方案：把两个 gate 的初始化 bias 设为 logit(1.5)≈0.405，")
print("           使初始 w≈0.82，两路合计≈1.64，接近原版量级。")
