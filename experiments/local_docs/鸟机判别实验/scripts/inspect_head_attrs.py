# -*- coding: utf-8 -*-
"""查看检测头属性，为 nc=2->3 扩头做准备。"""
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")
import torch  # noqa: E402

ck = torch.load("/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/"
                "ResTest/mdp150_M0BD_nc2_3to1_s0/weights/last.pt",
                map_location="cpu", weights_only=False)
m = ck["model"]
print("root nc", m.nc, "yaml ch", m.yaml.get("ch"), "yaml nc", m.yaml.get("nc"))
head = m.model[-1]
print("head", type(head).__name__)
for k, v in vars(head).items():
    if not isinstance(v, torch.nn.Module):
        print("  attr", k, v if not isinstance(v, (list, tuple)) else type(v).__name__)
for name, mod in head.named_children():
    print("  branch", name, type(mod).__name__,
          "nc", getattr(mod, "nc", None), "no", getattr(mod, "no", None))
