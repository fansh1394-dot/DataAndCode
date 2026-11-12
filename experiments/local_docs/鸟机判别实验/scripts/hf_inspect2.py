# -*- coding: utf-8 -*-
"""查看 LibreYOLO/coco-val2017 与 balloon 数据集的文件结构。"""
import json

import requests

M = "https://hf-mirror.net"
UA = {"User-Agent": "Mozilla/5.0 (research)"}
for ds in ("LibreYOLO/coco-val2017", "nielsr/balloon"):
    r = requests.get(f"{M}/api/datasets/{ds}", headers=UA, timeout=25)
    d = r.json()
    di = (d.get("cardData") or {}).get("dataset_info")
    print("==", ds, "download_size",
          (di or {}).get("download_size") if isinstance(di, dict) else di)
    for s in d.get("siblings", [])[:15]:
        print("   ", s["rfilename"], s.get("size"))
