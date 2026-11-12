# -*- coding: utf-8 -*-
"""bd_nc3 加载冒烟：确认 6 通道、双标签、含 class 2。"""
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator  # noqa: E402

y = "/root/rivermind-fs/birddrone/bd_nc3/data_nc3.yaml"
args = {"task": "detect", "mode": "val", "data": y, "imgsz": 640, "batch": 4}
v = MultiModalDetectionValidator(args=args)
data = check_det_dataset(y)
for k, val in yaml.safe_load(open(y)).items():
    data.setdefault(k, val)
v.data = data
v.stride = 32
ds = v.build_dataset(data["val"], mode="val", batch=4)
print("dataset", len(ds), "nc", ds.data.get("nc"), "names", ds.data.get("names"))
loader = build_dataloader(ds, batch=4, workers=0, shuffle=False, rank=-1)
b = next(iter(loader))
print("img", tuple(b["img"].shape), "cls", b["cls"].view(-1).tolist()[:12])
print("has_ir", tuple(b.get("img_ir", torch.empty(0)).shape))
print("file", b["im_file"][0])
