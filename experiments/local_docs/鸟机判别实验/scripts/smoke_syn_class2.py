# -*- coding: utf-8 -*-
"""只加载合成帧，确认 class 2 标签能进入 batch。"""
import sys
from pathlib import Path

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

import yaml  # noqa: E402
from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator  # noqa: E402

R = Path("/root/rivermind-fs/birddrone/bd_nc3")
syn = sorted(R.glob("images/val/*__syn*.jpg"))[:16]
(R / "syn_only.txt").write_text("\n".join(f"images/val/{p.name}" for p in syn) + "\n")
y = str(R / "data_nc3.yaml")
(R / "data_syn.yaml").write_text(
    (R / "data_nc3.yaml").read_text().replace("val: val_nc3.txt", "val: syn_only.txt"),
    encoding="utf-8")
y = str(R / "data_syn.yaml")
args = {"task": "detect", "mode": "val", "data": y, "imgsz": 640, "batch": 4}
v = MultiModalDetectionValidator(args=args)
data = check_det_dataset(y)
for k, val in yaml.safe_load(open(y)).items():
    data.setdefault(k, val)
v.data = data
v.stride = 32
ds = v.build_dataset(data["val"], mode="val", batch=4)
print("selected", len(ds), "files")
loader = build_dataloader(ds, batch=4, workers=0, shuffle=False, rank=-1)
b = next(iter(loader))
print("img", tuple(b["img"].shape), "cls", b["cls"].view(-1).tolist()[:16])
print("labels_ir_cls", b.get("cls_ir", b["cls"]).view(-1).tolist()[:16])
print("file", b["im_file"][0])
