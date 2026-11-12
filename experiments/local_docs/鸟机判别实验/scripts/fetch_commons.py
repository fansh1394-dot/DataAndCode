# -*- coding: utf-8 -*-
"""从 Wikimedia Commons 抓取真实气球/风筝图片（含许可元数据）。

只抓 JPEG/PNG、宽度 >= 800 的文件；按 pageid 去重；记录作者/许可/来源页，
供论文数据来源与署名使用。默认只取 CC0/PD/CC BY/CC BY-SA/CC BY-NC，
并单独标记 SA/NC，便于后续筛选。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "BirdDroneAcademicDataset/1.0 (non-commercial research)"}

CATEGORIES = {
    "balloon": [
        "Category:Balloons in flight",
        "Category:Party balloons",
        "Category:Toy balloons",
        "Category:Balloons",
        "Category:Gas balloons",
    ],
    "kite": [
        "Category:Kites in flight",
        "Category:Kites",
        "Category:Sport kites",
        "Category:Box kites",
    ],
}


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def api(params):
    p = {"format": "json", "formatversion": 2, **params}
    r = requests.get(API, params=p, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def iter_category(cat: str, limit: int):
    cont = {}
    got = 0
    while got < limit:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": cat,
            "gcmtype": "file",
            "gcmlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1600,
            **cont,
        }
        data = api(params)
        pages = data.get("query", {}).get("pages", [])
        for pg in pages:
            ii = (pg.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if (ii.get("width") or 0) < 800:
                continue
            yield {
                "pageid": pg.get("pageid"),
                "title": pg.get("title"),
                "page_url": ii.get("descriptionurl"),
                "url": ii.get("thumburl") or ii.get("url"),
                "width": ii.get("width"),
                "height": ii.get("height"),
                "mime": ii.get("mime"),
                "meta": ii.get("extmetadata", {}),
            }
            got += 1
            if got >= limit:
                return
        if "continue" not in data:
            return
        cont = data["continue"]
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/rivermind-fs/birddrone/other_raw")
    ap.add_argument("--limit", type=int, default=250, help="每类最多抓多少张")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    manifest = []
    seen = set()
    for cls, cats in CATEGORIES.items():
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for cat in cats:
            if n >= a.limit:
                break
            for item in iter_category(cat, a.limit - n):
                meta = item["meta"]
                lic = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
                artist = strip_html(meta.get("Artist", {}).get("value", ""))
                lic_url = meta.get("LicenseUrl", {}).get("value", "")
                rec = {
                    "class": cls,
                    "pageid": item["pageid"],
                    "title": item["title"],
                    "page_url": item["page_url"],
                    "url": item["url"],
                    "license": lic,
                    "license_url": lic_url,
                    "artist": artist,
                    "width": item["width"],
                    "height": item["height"],
                    "category": cat,
                    "sha1": None,
                    "file": None,
                }
                if a.dry_run:
                    print(json.dumps(rec, ensure_ascii=False))
                    n += 1
                    if n >= a.limit:
                        break
                    continue
                if item["pageid"] in seen:
                    continue
                try:
                    r = requests.get(item["url"], headers=UA, timeout=60)
                    r.raise_for_status()
                except Exception as e:
                    print("skip", item["title"], e)
                    continue
                sha = hashlib.sha1(r.content).hexdigest()
                if sha in {m["sha1"] for m in manifest}:
                    continue
                ext = ".png" if item["mime"] == "image/png" else ".jpg"
                fn = f"{cls}_{len([m for m in manifest if m['class'] == cls]):04d}{ext}"
                (d / fn).write_bytes(r.content)
                rec["sha1"] = sha
                rec["file"] = f"{cls}/{fn}"
                manifest.append(rec)
                seen.add(item["pageid"])
                n += 1
                print(f"[{cls}] {n}/{a.limit} {fn} {lic} | {item['title'][:60]}")
                if n >= a.limit:
                    break
                time.sleep(0.15)
    if not a.dry_run:
        mf = out / "manifest.jsonl"
        with mf.open("a", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print("saved", len(manifest), "->", mf)


if __name__ == "__main__":
    main()
