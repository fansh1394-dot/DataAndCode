# -*- coding: utf-8 -*-
"""检查 1 epoch 探针后门控 alpha 是否随 IR 位移下降（前向即可，不训练）。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)

from ultralytics import YOLOMM  # noqa: E402
from ultralytics.nn.mm.misalign_gate import MisalignGatedConcat  # noqa: E402

CKPT = f"{PROJ}/YOLO26/ResTest/mdp1_G1probe/weights/best.pt"
ROOT = Path("/3250410011/root/rivermind-data/mendeley_split")
IM = 640

m = YOLOMM(CKPT)
model = m.model.float().cuda().eval()
gates = [mod for mod in model.modules() if isinstance(mod, MisalignGatedConcat)]
print(f"gates={len(gates)}")

rgb_dir, ir_dir = ROOT / "images/val", ROOT / "images_ir/val"
rgb_files = sorted(rgb_dir.glob("*.png"))[:8]
print("frames:", len(rgb_files))


def load6(rgb_p, dx):
    a = cv2.imread(str(rgb_p))
    b = cv2.imread(str(ir_dir / rgb_p.name))
    if b is None:
        b = a.copy()
    a = cv2.resize(a, (IM, IM))
    b = cv2.resize(b, (IM, IM))
    if dx:
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        b = cv2.warpAffine(b, M, (IM, IM), borderValue=(114, 114, 114))
    x = np.ascontiguousarray(np.concatenate([a, b], axis=2)[:, :, ::-1])  # BGR->RGB
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0


for dx in (0, 2, 4, 8):
    x = torch.stack([load6(p, dx) for p in rgb_files]).cuda()
    with torch.no_grad():
        model(x)
    vals = [g.last_alpha_mean for g in gates]
    stds = [g.last_alpha_std for g in gates]
    print(f"shift={dx:2d}px  alpha(46/49/52)=" +
          "  ".join(f"{v:.4f}" for v in vals) +
          "  mean=%.4f" % float(np.mean(vals)))
