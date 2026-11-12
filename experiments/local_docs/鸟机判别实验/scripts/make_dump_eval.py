# -*- coding: utf-8 -*-
"""生成开启 save_json 的评测副本（不改仓库文件）。"""
from pathlib import Path

src = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/scripts/eval_split.py")
s = src.read_text(encoding="utf-8")
n = s.count('"save_json": False')
s = s.replace('"save_json": False', '"save_json": True')
Path("/tmp/eval_dump.py").write_text(s, encoding="utf-8")
print("replaced", n)
