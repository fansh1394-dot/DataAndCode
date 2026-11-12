# -*- coding: utf-8 -*-
"""列出主干各层的输出形状，用于挑选低层 RGB/IR 配对。"""
import sys
sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")
import torch
from ultralytics import YOLOMM

CKPT = ("/3250410011/root/rivermind-data/MutilModel_3485988898/"
        "YOLO26/ResTest/mdp1_G1probe/weights/best.pt")
m = YOLOMM(CKPT)
model = m.model.float().cuda().eval()

shapes = {}
for i, mod in enumerate(model.model):
    mod.register_forward_hook(lambda mm, inp, out, idx=i: shapes.__setitem__(idx, tuple(out.shape) if torch.is_tensor(out) else type(out).__name__))

with torch.no_grad():
    model(torch.zeros(1, 6, 640, 640).cuda())

for i, s in shapes.items():
    if i <= 30:
        print(i, model.model[i].__class__.__name__, s)
