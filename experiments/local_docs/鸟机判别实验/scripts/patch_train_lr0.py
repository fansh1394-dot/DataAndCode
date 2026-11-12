# -*- coding: utf-8 -*-
"""给 train_full.py 增加 --lr0（微调用，默认不改历史行为）。"""
from pathlib import Path

p = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/train_full.py")
s = p.read_text(encoding="utf-8")
if "--lr0" in s:
    print("already patched")
    raise SystemExit(0)

anchor_arg = '''    ap.add_argument("--patience", type=int, default=None,'''
add_arg = '''    ap.add_argument("--lr0", type=float, default=None,
                    help="初始学习率覆盖（微调建议 0.001~0.003；不传则沿用默认 0.01）")
'''
assert s.count(anchor_arg) == 1
s = s.replace(anchor_arg, add_arg + anchor_arg)

anchor = '''    # 损失权重透传（box/dfl）：仅显式传入时覆盖，保持历史实验口径不变
'''
add = '''    if getattr(args, "lr0", None) is not None:
        train_kwargs["lr0"] = float(args.lr0)

'''
assert s.count(anchor) == 1
s = s.replace(anchor, add + anchor)

p.with_suffix(".py.bak_lr0").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
p.write_text(s, encoding="utf-8")
print("patched", p)
