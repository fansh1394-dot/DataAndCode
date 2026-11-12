# -*- coding: utf-8 -*-
"""从 COCO val2017 中筛出 kite / airplane 图，并生成缩略图拼版。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

RAW = Path("/root/rivermind-fs/birddrone/other_raw")
COCO = RAW / "coco_val2017" / "coco-val2017"
IMG_DIR = COCO / "images" / "val2017"
LB_DIR = COCO / "labels" / "val2017"
OUT = RAW / "coco"
CLS = {33: "kite", 4: "airplane"}   # COCO 80 类顺序
SHEET_DIR = RAW / "sheets"


def contact_sheet(files, dst, cols=8, cell=180):
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(files):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im.thumbnail((cell, cell))
        x = (i % cols) * cell
        y = (i // cols) * (cell + 16)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        draw.text((x + 4, y + cell), p.name[:24], fill="black")
    sheet.save(dst, quality=88)
    print("sheet", dst, sheet.size)


def main():
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    picked = {"kite": [], "airplane": []}
    labels = sorted(LB_DIR.glob("*.txt"))
    print("coco images", len(list(IMG_DIR.glob("*"))), "labels", len(labels))
    for lb in labels:
        cls_ids = set()
        for ln in lb.read_text().splitlines():
            t = ln.split()
            if t:
                cls_ids.add(int(float(t[0])))
        for cid, name in CLS.items():
            if cid in cls_ids:
                img = next(IMG_DIR.glob(lb.stem + ".*"), None)
                if img is None:
                    continue
                d = OUT / name
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, d / img.name)
                shutil.copy2(lb, d / lb.name)
                picked[name].append(d / img.name)
                manifest.append({
                    "class": name, "source": "COCO val2017",
                    "image": str(d / img.name), "label": str(d / lb.name),
                    "license": "COCO annotations CC BY 4.0; images Flickr (individual licenses)",
                })
                break

    for name, files in picked.items():
        contact_sheet(files[:48], SHEET_DIR / f"coco_{name}.jpg")

    balloon_imgs = sorted((RAW / "balloon").glob("party_*.jpg"))
    contact_sheet(balloon_imgs, SHEET_DIR / "balloon_party.jpg")
    for p in balloon_imgs:
        manifest.append({
            "class": "balloon", "source": "Roboflow balloon via nielsr/balloon",
            "image": str(p), "label": "polygon in custom_train.json/custom_val.json",
            "license": "see HuggingFace dataset card (Roboflow public dataset)",
        })

    (RAW / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8")
    print("kite", len(picked["kite"]), "airplane", len(picked["airplane"]),
          "balloon", len(balloon_imgs))


if __name__ == "__main__":
    main()
