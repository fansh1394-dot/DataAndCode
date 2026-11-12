# -*- coding: utf-8 -*-
"""修正气球文件命名（该集是派对气球而非热气球），统计实例数并重建清单。"""
import json
from pathlib import Path

RAW = Path("/root/rivermind-fs/birddrone/other_raw")
BD = RAW / "balloon"

for p in sorted(BD.glob("hotair_*.jpg")):
    p.rename(p.with_name(p.name.replace("hotair_", "party_", 1)))

n_ann = 0
for j in ("custom_train.json", "custom_val.json"):
    d = json.loads((BD / j).read_text())
    n_ann += len(d.get("annotations", []))
    print(j, "images", len(d.get("images", [])), "annotations", len(d.get("annotations", [])))

kite_lb = list((RAW / "coco" / "kite").glob("*.txt"))
air_lb = list((RAW / "coco" / "airplane").glob("*.txt"))
n_kite = sum(1 for f in kite_lb for ln in f.read_text().splitlines() if ln.split() and int(float(ln.split()[0])) == 33)
n_air = sum(1 for f in air_lb for ln in f.read_text().splitlines() if ln.split() and int(float(ln.split()[0])) == 4)

manifest = []
for name, d, cid in (("kite", RAW / "coco" / "kite", 33), ("airplane", RAW / "coco" / "airplane", 4)):
    for img in sorted(d.glob("*.jpg")):
        manifest.append({
            "class": name, "source": "COCO val2017",
            "image": str(img), "label": str(img.with_suffix(".txt")),
            "license": "COCO annotations CC BY 4.0; images Flickr (individual licenses)",
        })
for img in sorted(BD.glob("party_*.jpg")):
    manifest.append({
        "class": "balloon", "source": "Roboflow balloon via nielsr/balloon",
        "image": str(img), "label": "polygon in custom_train.json/custom_val.json",
        "license": "see HuggingFace dataset card (Roboflow public dataset)",
    })
(RAW / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n", encoding="utf-8")

print("balloon_images", len(list(BD.glob("party_*.jpg"))), "balloon_ann", n_ann)
print("kite_images", len(kite_lb), "kite_instances", n_kite)
print("airplane_images", len(air_lb), "airplane_instances", n_air)
