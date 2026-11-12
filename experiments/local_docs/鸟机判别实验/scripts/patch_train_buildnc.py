# -*- coding: utf-8 -*-
"""给 train_full.py 增加 --build-nc（warm start 换类别数时用）。

背景：路线 C 会用 ckpt 的 nc 构建模型，nc=1 的 UAV 权重无法直接 warm start
一个 nc=2（bird/drone）检测器。--build-nc 允许显式指定构建用 nc，backbone/neck
照常加载，形状不匹配的检测头分类分支被跳过并重新初始化。
"""
from pathlib import Path

p = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/train_full.py")
s = p.read_text(encoding="utf-8")
if "--build-nc" in s:
    print("already patched")
    raise SystemExit(0)

old_arg = '''    ap.add_argument("--pretrained", default=None,
                    help="预训练权重路径（路线 B / C 必填）")
'''
new_arg = old_arg + '''    ap.add_argument("--build-nc", type=int, default=None, dest="build_nc",
                    help="路线 C 换类别数时显式指定构建用 nc（默认沿用 ckpt 的 nc；"
                         "形状不匹配的检测头会被跳过并重新初始化）")
'''
assert s.count(old_arg) == 1
s = s.replace(old_arg, new_arg)

old = '''        _nc = int(getattr(_ck.get("model"), "nc", 0) or 0) if isinstance(_ck, dict) else 0
'''
new = '''        _ck_nc = int(getattr(_ck.get("model"), "nc", 0) or 0) if isinstance(_ck, dict) else 0
        _nc = int(args.build_nc or _ck_nc)
        if args.build_nc:
            print(f"[nc 覆盖] --build-nc={args.build_nc}（ckpt nc={_ck_nc}，"
                  f"检测头分类分支将重新初始化）")
'''
assert s.count(old) == 1
s = s.replace(old, new)

p.with_suffix(".py.bak_buildnc").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
p.write_text(s, encoding="utf-8")
print("patched", p)
