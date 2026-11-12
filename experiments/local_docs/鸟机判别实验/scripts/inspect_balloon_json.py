# -*- coding: utf-8 -*-
"""检查 balloon 标注与导出图片的对应关系。"""
import json
from pathlib import Path

import pyarrow.parquet as pq

D = Path("/root/rivermind-fs/birddrone/other_raw/balloon")
for split in ("train", "val"):
    j = json.loads((D / f"custom_{split}.json").read_text())
    t = pq.read_table(D / f"{split}.parquet").column("image").to_pylist()
    print("==", split, "json_images", len(j["images"]), "parquet", len(t),
          "annotations", len(j["annotations"]))
    for im in j["images"][:3]:
        print("  json:", im.get("id"), im.get("file_name"))
    for item in t[:3]:
        print("  pq  :", item.get("path") if isinstance(item, dict) else type(item))
    ann = j["annotations"][0]
    print("  ann keys", list(ann.keys()), "bbox", ann.get("bbox"),
          "seg_len", len(ann.get("segmentation", [[]])[0]))
