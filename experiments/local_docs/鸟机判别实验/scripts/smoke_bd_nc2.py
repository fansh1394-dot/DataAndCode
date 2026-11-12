# -*- coding: utf-8 -*-
"""E28 数据冒烟：确认 bd_nc2 能被多模态 loader 正确读成 6 通道 + 双标签。"""
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics.data.build import build_dataloader  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator  # noqa: E402

yaml_path = "/root/rivermind-fs/birddrone/bd_nc2/data.yaml"
args = {"task": "detect", "mode": "val", "data": yaml_path, "imgsz": 640, "batch": 4}
v = MultiModalDetectionValidator(args=args)
data = check_det_dataset(yaml_path)
for k, val in yaml.safe_load(open(yaml_path)).items():
    data.setdefault(k, val)
v.data = data
v.stride = 32
ds = v.build_dataset(data["val"], mode="val", batch=4)
print("dataset", type(ds).__name__, len(ds))
loader = build_dataloader(ds, batch=4, workers=0, shuffle=False, rank=-1)
batch = next(iter(loader))
print("img", tuple(batch["img"].shape), batch["img"].dtype)
print("keys", sorted(batch.keys()))
print("cls", batch["cls"].view(-1).tolist()[:20])
print("bboxes", tuple(batch["bboxes"].shape))
print("has_ir", tuple(batch.get("img_ir", torch.empty(0)).shape))
print("im_file0", batch["im_file"][0])
