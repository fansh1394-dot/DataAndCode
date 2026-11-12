# -*- coding: utf-8 -*-
"""通过 HF 镜像搜索含 balloon / kite 的数据集。"""
import json

import requests

BASE = "https://hf-mirror.net"
UA = {"User-Agent": "Mozilla/5.0 (research)"}
for q in ("balloon", "kite", "open-images", "open_images"):
    try:
        r = requests.get(f"{BASE}/api/datasets", params={"search": q, "limit": 20},
                         headers=UA, timeout=20)
        print("==", q, r.status_code)
        if r.status_code == 200:
            for d in r.json()[:20]:
                print("  ", d.get("id"), d.get("downloads"), d.get("tags", [])[:3])
    except Exception as e:
        print("FAIL", q, type(e).__name__, str(e)[:80])
