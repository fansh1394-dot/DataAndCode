# -*- coding: utf-8 -*-
"""单算子最小化验证：绕过整网，直接测六个融合算子的 __init__ + forward。

这样能区分「算子本身有问题」还是「我的 YAML 替换破坏了整网连接」。
输入统一为两张 [2, 256, 32, 32] 特征图（模拟 P4 尺度）。
"""
import sys
import traceback

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

import torch

from ultralytics.nn.modules.fusion.cgafusion import CGAFusion
from ultralytics.nn.modules.fusion.cmce import CMCE
from ultralytics.nn.modules.fusion.gfm import GFM
from ultralytics.nn.modules.fusion.pim import PIM
from ultralytics.nn.modules.fusion.wcmf import WCMF
from ultralytics.nn.modules.fusion.wdaf import WDAF

C = 256
x1 = torch.randn(2, C, 32, 32)
x2 = torch.randn(2, C, 32, 32)

# 按注册逻辑的调用口径构造：
#   跨模态分支: Module([c_left, c_right], ouc, ...)
#   纯特征分支: Module(c_left)   且 forward 接受 [x1, x2]
CASES = [
    ("CGAFusion", lambda: CGAFusion([C, C], C, 8), [x1, x2]),
    ("WCMF", lambda: WCMF(C), [x1, x2]),
    ("WDAF", lambda: WDAF([C, C], C), [x1, x2]),
    ("PIM", lambda: PIM(C), [x1, x2]),
    ("CMCE", lambda: CMCE(C), [x1, x2]),
    ("GFM", lambda: GFM(C), [x1, x2]),
]

for name, ctor, inp in CASES:
    print(f"\n===== {name} =====", flush=True)
    try:
        m = ctor()
        n = sum(p.numel() for p in m.parameters())
        print(f"  构造 OK  参数量={n / 1e6:.4f} M", flush=True)
        with torch.no_grad():
            out = m(inp)
        shp = tuple(out.shape) if torch.is_tensor(out) else type(out).__name__
        print(f"  前向 OK  out={shp}", flush=True)
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {str(e)[:160]}", flush=True)
        traceback.print_exc(limit=2)
