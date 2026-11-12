# -*- coding: utf-8 -*-
"""为 bd_nc2 生成 nc=3 评测 yaml（GT 只有 bird/drone，用于三类模型护栏评测）。"""
from pathlib import Path

p = Path("/root/rivermind-fs/birddrone/bd_nc2/data.yaml")
s = p.read_text(encoding="utf-8")
s = s.replace("nc: 2", "nc: 3")
s = s.replace("  1: drone", "  1: drone\n  2: other")
out = p.with_name("data_nc3_eval.yaml")
out.write_text(s, encoding="utf-8")
print(out)
