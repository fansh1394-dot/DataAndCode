# -*- coding: utf-8 -*-
"""检查 E28c 检测头的 cv3（分类分支）形状，用于 nc=2 -> 3 的权重迁移。"""
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")
import torch  # noqa: E402

ck = torch.load("/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/"
                "ResTest/mdp150_M0BD_nc2_3to1_s0/weights/last.pt",
                map_location="cpu", weights_only=False)
m = ck["model"]
print(type(m).__name__, "nc", getattr(m, "nc", None), "yaml nc", m.yaml.get("nc"),
      "ch", m.yaml.get("ch"))
sd = m.state_dict()
for k, v in sd.items():
    if "cv3" in k and v.ndim == 4:
        print(k, tuple(v.shape))
for k, v in sd.items():
    if ".cv3." in k and v.ndim == 1:
        print("bias", k, tuple(v.shape))
