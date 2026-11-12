# -*- coding: utf-8 -*-
"""核查 Mamba 在本机/本工程中的实际可用性。

回答三个问题：
1) mamba_ssm / causal_conv1d 等 CUDA 依赖是否可导入；
2) 工程内的 Mamba 模块能否在 CPU 上完成构建期 stride 推断（导出前提）；
3) 这些模块是否已被 tasks.py 注册为可用层。
"""
import importlib
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

print("=== 1) 外部 CUDA 依赖 ===")
for mod in ("mamba_ssm", "causal_conv1d", "selective_scan_cuda"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<22} OK  {getattr(m, '__file__', '')}")
    except Exception as e:
        print(f"  {mod:<22} FAIL {type(e).__name__}: {e}")

print("\n=== 2) 工程内 Mamba 模块导入 ===")
for mod in (
    "ultralytics.nn.mamba",
    "ultralytics.nn.mamba.mamba_vss",
    "ultralytics.nn.mamba.evssm",
    "ultralytics.nn.mamba.mamba_vision",
):
    try:
        importlib.import_module(mod)
        print(f"  {mod:<42} OK")
    except Exception as e:
        print(f"  {mod:<42} FAIL {type(e).__name__}: {str(e)[:90]}")

print("\n=== 3) tasks.py 中 Mamba 相关注册 ===")
import re
from pathlib import Path

src = Path("ultralytics/nn/tasks.py").read_text(encoding="utf-8")
hits = [ln.strip() for ln in src.splitlines()
        if re.search(r"MAMBA|Mamba|SS2D|VSSM|EVSSM", ln) and not ln.strip().startswith("#")]
print(f"  命中行数: {len(hits)}")
for h in hits[:8]:
    print(f"    {h[:110]}")
