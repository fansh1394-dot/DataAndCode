# -*- coding: utf-8 -*-
"""bd_nc3 完整性检查：文件配对、类别分布、合成占比。"""
from collections import Counter
from pathlib import Path

R = Path("/root/rivermind-fs/birddrone/bd_nc3")
for split in ("train", "val"):
    lst = [x.strip() for x in (R / f"{'train' if split == 'train' else 'val'}_nc3.txt").read_text().splitlines() if x.strip()]
    missing = [x for x in lst if not (R / x).exists()]
    syn = [x for x in lst if "__syn" in x]
    base = [x for x in lst if "__syn" not in x]
    cc = Counter()
    for x in lst:
        lb = R / "labels" / split / (Path(x).stem + ".txt")
        for ln in lb.read_text().splitlines():
            t = ln.split()
            if t:
                cc[int(float(t[0]))] += 1
    print(split, "listed", len(lst), "base", len(base), "syn", len(syn),
          "missing", len(missing), "classes", dict(cc))
    ir_stems = {p.stem for p in (R / "images_ir" / split).iterdir()}
    print("  ir_missing", sum(1 for x in lst if Path(x).stem not in ir_stems))
print("syn_ratio_train", round(3166 / 21582, 4))
