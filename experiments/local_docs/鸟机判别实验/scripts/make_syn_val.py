# -*- coding: utf-8 -*-
"""生成全部合成 val 列表与对应 yaml。"""
from pathlib import Path

R = Path("/root/rivermind-fs/birddrone/bd_nc3")
files = sorted(R.glob("images/val/*__syn*.jpg"))
(R / "syn_only.txt").write_text("\n".join(f"images/val/{p.name}" for p in files) + "\n", encoding="utf-8")
s = (R / "data_nc3.yaml").read_text(encoding="utf-8").replace("val: val_nc3.txt", "val: syn_only.txt")
(R / "data_syn.yaml").write_text(s, encoding="utf-8")
print("syn val", len(files))
