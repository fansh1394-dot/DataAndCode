# -*- coding: utf-8 -*-
"""把 E28c 的 nc=2 权重扩成 nc=3：保留 bird/drone 分类通道，other 新增随机初始化。

不修改项目代码，只产出一个新的 .pt 供训练入口使用。
"""
import sys
from copy import deepcopy

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

SRC = ("/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest/"
       "mdp150_M0BD_nc2_3to1_s0/weights/last.pt")
DST = "/root/rivermind-fs/birddrone/nc3_init.pt"
OLD, NEW = 2, 3

ck = torch.load(SRC, map_location="cpu", weights_only=False)
m = deepcopy(ck["model"])
head = m.model[-1]

n_repl = 0
for br_name in ("rgb_detect", "ir_detect", "fuse_detect", "fuse_ir_detect"):
    br = getattr(head, br_name)
    for lvl in range(len(br.cv3)):
        seq = br.cv3[lvl]
        assert isinstance(seq[-1], nn.Conv2d) and seq[-1].out_channels == OLD
        old = seq[-1]
        new = nn.Conv2d(old.in_channels, NEW, old.kernel_size, old.stride,
                        old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight[:OLD] = old.weight
            new.weight[OLD:] = old.weight[:1] * 0 + torch.randn_like(old.weight[:1]) * 0.01
            if old.bias is not None:
                new.bias[:OLD] = old.bias
                new.bias[OLD:] = 0
        seq[-1] = new
        n_repl += 1
    br.nc = NEW
    br.no = NEW + 4 * getattr(br, "reg_max", 1)

head.nc = NEW
head.no = NEW + 4 * getattr(head, "reg_max", 1)
m.nc = NEW
m.yaml["nc"] = NEW
m.yaml["names"] = {0: "bird", 1: "drone", 2: "other"}

torch.save({"model": m, "epoch": -1, "best_fitness": 0.0,
            "optimizer": None, "train_args": ck.get("train_args", {})}, DST)
print("replaced", n_repl, "convs; head no=", head.no, "->", DST)
