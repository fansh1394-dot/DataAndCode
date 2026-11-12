# -*- coding: utf-8 -*-
"""下载 nielsr/balloon（Roboflow 热气球分割集）并导出图片与 COCO 标注。"""
from pathlib import Path

import pyarrow.parquet as pq
import requests

M = "https://hf-mirror.net/datasets/nielsr/balloon/resolve/main"
FILES = {
    "data/train-00000-of-00001-2d3dccae7b7adfb3.parquet": "train.parquet",
    "data/validation-00000-of-00001-4834160fb6a854a5.parquet": "val.parquet",
    "custom_train.json": "custom_train.json",
    "custom_val.json": "custom_val.json",
}
OUT = Path("/root/rivermind-fs/birddrone/other_raw/balloon")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research)"}

for remote, local in FILES.items():
    dst = OUT / local
    if dst.exists() and dst.stat().st_size > 0:
        continue
    r = requests.get(f"{M}/{remote}", headers=UA, timeout=120, stream=True)
    r.raise_for_status()
    with dst.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print("downloaded", local, dst.stat().st_size)

for split in ("train", "val"):
    t = pq.read_table(OUT / f"{split}.parquet")
    col = t.column("image").to_pylist()
    for i, item in enumerate(col):
        data = item["bytes"] if isinstance(item, dict) else item
        name = item.get("path") if isinstance(item, dict) else None
        ext = ".jpg"
        if name and "." in name:
            ext = "." + name.rsplit(".", 1)[-1].lower()
        (OUT / f"hotair_{split}_{i:03d}{ext}").write_bytes(data)
    print(split, len(col), "images")
