# -*- coding: utf-8 -*-
"""搜索并筛选体积可控、含真实气球/风筝图像的 HF 数据集。"""
import json

import requests

UA = {"User-Agent": "Mozilla/5.0 (research)"}
M = "https://hf-mirror.net"

QUERIES = ["kites", "kite flying", "hot air balloon", "hot-air-balloon",
           "parachute", "balloon sky", "balloons", "drone bird"]

seen = {}
for q in QUERIES:
    try:
        r = requests.get(f"{M}/api/datasets", params={"search": q, "limit": 30},
                         headers=UA, timeout=20)
        if r.status_code != 200:
            continue
        for d in r.json():
            seen[d["id"]] = d
    except Exception as e:
        print("FAIL", q, type(e).__name__)

print("candidates", len(seen))
for did in seen:
    try:
        info = requests.get(f"{M}/api/datasets/{did}", headers=UA, timeout=20).json()
        di = (info.get("cardData") or {}).get("dataset_info") or {}
        if isinstance(di, list):
            di = di[0] if di else {}
        feats = [f.get("name") for f in di.get("features", [])]
        splits = [(s.get("name"), s.get("num_examples")) for s in di.get("splits", [])]
        size = di.get("download_size") or 0
        if "image" in feats and size and size < 3e9:
            print(f"{did:55s} {size/1e6:8.1f} MB  feats={feats[:4]} splits={splits[:3]}")
    except Exception:
        pass
