# -*- coding: utf-8 -*-
"""E28c：Mendeley:AVID = 3:1 的二阶段收尾列表。

做法：直接重复 Mendeley 训练帧 3 次（列表层面，不复制文件），
AVID 沿用 E28b 的 4 604 帧抽样，保持鸟类覆盖。
"""
from pathlib import Path

ROOT = Path("/root/rivermind-fs/birddrone/bd_nc2")
lines = [x.strip() for x in (ROOT / "train_bal.txt").read_text().splitlines() if x.strip()]
men = [x for x in lines if "men__" in x]
avid = [x for x in lines if "avid__" in x]
train = men * 3 + avid
(ROOT / "train_3to1.txt").write_text("\n".join(train) + "\n", encoding="utf-8")

base = (ROOT / "data_bal.yaml").read_text(encoding="utf-8")
out = []
for ln in base.splitlines():
    if ln.startswith("train:"):
        ln = "train: train_3to1.txt"
    out.append(ln)
(ROOT / "data_3to1.yaml").write_text("\n".join(out) + "\n", encoding="utf-8")
print("men", len(men), "avid", len(avid), "total", len(train))
