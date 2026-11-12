# -*- coding: utf-8 -*-
"""验证 CGAFusion 解绑改动：三种模式都能跑通，且输出范围符合预期。"""
import os
import sys

import torch

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from ultralytics.nn.modules.fusion.cgafusion import CGAFusion  # noqa: E402

print("=== 1. 单模块：解绑 vs 原版 ===")
for dec in (True, False):
    m = CGAFusion([64, 64], 64, 8, decouple=dec).eval()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 64, 32, 32)
    y = torch.randn(2, 64, 32, 32)
    with torch.no_grad():
        out = m([x, y])
    print(f"  decouple={dec!s:<5} 参数={n:>7,}  输出 shape={tuple(out.shape)}  "
          f"mean={out.mean():.4f} std={out.std():.4f}")

print("\n=== 2. 系数范围对比（数学层面）===")
print("  原版   : x系数∈[1,2], y系数∈[1,2], 和=3, 比值∈[0.5,2]")
print("  解绑版 : w_x∈(0,1), w_y∈(0,1), 可独立取 0 或 1")
print("           例如 w_x=1,w_y=0 -> 只用 RGB；w_x=0,w_y=1 -> 只用 IR")

print("\n=== 3. 残差选项 ===")
for res in (False, True):
    m = CGAFusion([32, 32], 32, 8, decouple=True, use_residual=res).eval()
    x = torch.randn(1, 32, 16, 16)
    y = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        out = m([x, y])
    print(f"  use_residual={res!s:<5} 输出 std={out.std():.4f}")

print("\n=== 4. 整网构建（CGA 三头 YAML）===")
from ultralytics.nn.tasks import DetectionModel, yaml_model_load  # noqa: E402

p = "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-cga3h.yaml"
try:
    m = DetectionModel(yaml_model_load(p), ch=6, nc=1, verbose=False)
    n = sum(q.numel() for q in m.parameters()) / 1e6
    print(f"  构建 OK  参数量={n:.4f} M（旧 CGA 为 5.9691 M）")
    m.eval()
    with torch.no_grad():
        o = m(torch.randn(1, 6, 256, 256))
    print(f"  前向 OK  type={type(o).__name__}")
except Exception as e:
    print(f"  ✗ 失败 {type(e).__name__}: {str(e)[:300]}")
