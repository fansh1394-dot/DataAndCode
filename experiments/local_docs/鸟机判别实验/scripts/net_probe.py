# -*- coding: utf-8 -*-
"""测试服务器出网可达性（用于选择图片源）。"""
import requests

HOSTS = [
    "https://commons.wikimedia.org/w/api.php?action=query&meta=siteinfo&format=json",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/RedDot_Balloon.jpg/320px-RedDot_Balloon.jpg",
    "https://hf-mirror.net",
    "https://huggingface.co",
    "https://raw.githubusercontent.com",
    "https://images.pexels.com",
    "https://www.google.com",
]
UA = {"User-Agent": "Mozilla/5.0 (research)"}
for u in HOSTS:
    try:
        r = requests.get(u, headers=UA, timeout=8)
        print("OK  ", r.status_code, u[:60])
    except Exception as e:
        print("FAIL", type(e).__name__, u[:60], str(e)[:80])
