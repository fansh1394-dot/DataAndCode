import os
from pathlib import Path

ROOT = Path("/root/rivermind-fs/birddrone")
AVID = Path("/root/rivermind-fs/AVID")
Y = ROOT / "avid_yolo"
for d in ("images", "images_ir", "labels", "labels_ir"):
    (Y / d).mkdir(parents=True, exist_ok=True)

links = {
    "images/train": AVID / "images/rgb/train",
    "images/val": AVID / "images/rgb/val",
    "images_ir/train": AVID / "images/ir/train",
    "images_ir/val": AVID / "images/ir/val",
    "labels/train": AVID / "labels/train",
    "labels/val": AVID / "labels/val",
    "labels_ir/train": AVID / "labels/train",
    "labels_ir/val": AVID / "labels/val",
}
for rel, target in links.items():
    p = Y / rel
    if p.is_symlink() or p.exists():
        if p.is_dir() and not p.is_symlink():
            raise SystemExit(f"refuse to replace real dir {p}")
        p.unlink()
    p.symlink_to(target, target_is_directory=True)

# 用包装层路径重写 val 列表
old = (ROOT / "avid_bird_val.txt").read_text().splitlines()
new = [str(Y / "images/val" / Path(x).parent.name / Path(x).name) for x in old if x.strip()]
(Y / "avid_bird_val.txt").write_text("\n".join(new) + "\n", encoding="utf-8")

yaml_text = f"""path: {Y}
train: images/train
val: images/val
test: images/val
modality:
  rgb: images
  ir: images_ir
modality_used:
- rgb
- ir
labels:
  rgb: labels
  ir: labels_ir
pair_policy: any
align: false
channels: 3
Xch: 3
nc: 1
names:
  0: UAV
"""
(ROOT / "data_avid_nc1.yaml").write_text(yaml_text, encoding="utf-8")
print("wrapper", Y)
print("val_list", len(new), new[0] if new else None)
for rel in links:
    p = Y / rel
    print(rel, "->", os.readlink(p), p.exists())
