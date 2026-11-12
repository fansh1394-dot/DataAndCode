# -*- coding: utf-8 -*-
"""验证 balloon 标注与 party_*.jpg 的索引对应是否正确（bbox 越界检查）。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

D = Path("/root/rivermind-fs/birddrone/other_raw/balloon")
for split in ("train", "val"):
    j = json.loads((D / f"custom_{split}.json").read_text())
    files = sorted(D.glob(f"party_{split}_*.jpg"))
    imgs = {im["id"]: im for im in j["images"]}
    bad = 0
    for ann in j["annotations"]:
        im = imgs[ann["image_id"]]
        idx = j["images"].index(im)
        p = files[idx]
        with Image.open(p) as im_obj:
            w, h = im_obj.size
        x, y, bw, bh = ann["bbox"]
        if x < 0 or y < 0 or x + bw > w + 2 or y + bh > h + 2:
            bad += 1
    print(split, "images", len(files), "anns", len(j["annotations"]),
          "bbox_out_of_bounds", bad)
    # 打印前 3 张尺寸与 bbox，便于人工判断
    for ann in j["annotations"][:3]:
        im = imgs[ann["image_id"]]
        p = files[j["images"].index(im)]
        with Image.open(p) as im_obj:
            print("   ", p.name, im_obj.size, "bbox", [round(v) for v in ann["bbox"]])
