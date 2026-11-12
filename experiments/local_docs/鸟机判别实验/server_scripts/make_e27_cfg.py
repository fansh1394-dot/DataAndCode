import json
from pathlib import Path

ROOT = Path("/root/rivermind-fs/birddrone")
AVID = Path("/root/rivermind-fs/AVID")
split = json.loads((ROOT / "split.json").read_text())
test = split["sources"]["avid"]["splits"]["test"]

yaml_text = f"""path: {AVID}
train: images/rgb/train
val: images/rgb/val
test: images/rgb/val
modality:
  rgb: images/rgb
  ir: images/ir
modality_used:
- rgb
- ir
labels:
  rgb: labels
  ir: labels
pair_policy: any
align: false
channels: 3
Xch: 3
nc: 1
names:
  0: UAV
"""
(ROOT / "data_avid_nc1.yaml").write_text(yaml_text, encoding="utf-8")

paths = []
for v in test:
    if v["class"] != "bird":
        continue
    paths += sorted((AVID / "images/rgb/val" / v["video"]).glob("*.jpg"))
(ROOT / "avid_bird_val.txt").write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
print("bird_val_imgs", len(paths))
print("videos", [v["video"] for v in test if v["class"] == "bird"])
