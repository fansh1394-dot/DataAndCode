# -*- coding: utf-8 -*-
"""查找可用的 COCO（含 kite/airplane 类）与 kite 图像源。"""
import requests

UA = {"User-Agent": "Mozilla/5.0 (research)"}
M = "https://hf-mirror.net"

for q in ("coco2017", "coco val2017", "coco detection", "kite image", "kite detection"):
    try:
        r = requests.get(f"{M}/api/datasets", params={"search": q, "limit": 25},
                         headers=UA, timeout=20)
        print("==", q, r.status_code)
        for d in r.json()[:25]:
            t = d.get("tags", [])
            if any("modality:image" in x for x in t):
                print("   ", d["id"], [x for x in t if "size_categories" in x])
    except Exception as e:
        print("FAIL", q, type(e).__name__)
