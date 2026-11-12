# -*- coding: utf-8 -*-
"""验证 Mamba 层在「无 CUDA 扩展」条件下能否完成构建与前向。

上一条结论说「无法 CPU 构建」，此处实测纠正或佐证：
csms6s.py 内有 selective_scan_torch 纯 PyTorch 回退，需确认它是否真被走到。
"""
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

import torch

print("=== 后端可用性 ===")
from ultralytics.nn.mamba.mobilemamba import csms6s

for name in ("WITH_SELECTIVESCAN_OFLEX", "WITH_SELECTIVESCAN_CORE", "WITH_SELECTIVESCAN_MAMBA"):
    print(f"  {name} = {getattr(csms6s, name, 'N/A')}")
print(f"  WITH_CUDA = {getattr(csms6s, 'WITH_CUDA', 'N/A')}")

print("\n=== 纯 PyTorch 回退前向（CPU）===")
try:
    B, C, L, N = 1, 8, 64, 16
    u = torch.randn(B, C, L)
    delta = torch.rand(B, C, L) * 0.1
    A = -torch.rand(C, N)
    Bm = torch.randn(B, 1, N, L)
    Cm = torch.randn(B, 1, N, L)
    out = csms6s.selective_scan_torch(u, delta, A, Bm, Cm, None, None, True, True)
    print(f"  selective_scan_torch OK  out={tuple(out.shape)} dtype={out.dtype}")
except Exception as e:
    print(f"  selective_scan_torch FAIL {type(e).__name__}: {str(e)[:150]}")

print("\n=== 分发函数 selective_scan_fn（CPU, 应回落 torch）===")
try:
    out2 = csms6s.selective_scan_fn(u, delta, A, Bm, Cm, None, None, True, "torch")
    print(f"  selective_scan_fn(torch) OK  out={tuple(out2.shape)}")
except Exception as e:
    print(f"  selective_scan_fn(torch) FAIL {type(e).__name__}: {str(e)[:150]}")
