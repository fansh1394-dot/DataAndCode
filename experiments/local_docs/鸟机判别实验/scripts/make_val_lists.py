# -*- coding: utf-8 -*-
"""在 bd_nc2 下按来源生成 val 列表（相对 data.yaml 的 path）。"""
from pathlib import Path

root = Path("/root/rivermind-fs/birddrone/bd_nc2")
files = sorted((root / "images/val").iterdir())
men = [f"images/val/{p.name}" for p in files if p.name.startswith("men__")]
avid = [f"images/val/{p.name}" for p in files if p.name.startswith("avid__")]
(root / "men_val.txt").write_text("\n".join(men) + "\n", encoding="utf-8")
(root / "avid_val.txt").write_text("\n".join(avid) + "\n", encoding="utf-8")
print("men", len(men), "avid", len(avid))
