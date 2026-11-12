# -*- coding: utf-8 -*-
"""检查候选 HF 数据集的结构与可下载性。"""
import json

import requests

UA = {"User-Agent": "Mozilla/5.0 (research)"}
MIRROR = "https://hf-mirror.net"


def show(url, n=1200):
    try:
        r = requests.get(url, headers=UA, timeout=25)
        print("==", r.status_code, url[:90])
        if r.status_code == 200:
            try:
                print(json.dumps(r.json(), ensure_ascii=False)[:n])
            except Exception:
                print(r.text[:n])
    except Exception as e:
        print("FAIL", url[:70], type(e).__name__, str(e)[:60])


for ds in ("nielsr/balloon", "vikhyatk/openimages-bbox", "bitmind/open-images-v7-subset"):
    show(f"{MIRROR}/api/datasets/{ds}")
show(f"{MIRROR}/api/datasets/nielsr/balloon/tree/main")
show("https://datasets-server.huggingface.co/valid")
