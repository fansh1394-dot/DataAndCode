# -*- coding: utf-8 -*-
"""验证 stride-2 上的 peak_lag_x 是否是真实错位信号：对称性检验 (+8 / 0 / -8 px)。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
from ultralytics import YOLOMM  # noqa: E402

CKPT = f"{PROJ}/YOLO26/ResTest/mdp1_G1probe/weights/best.pt"
ROOT = Path("/3250410011/root/rivermind-data/mendeley_split")
IM, NF, BS = 640, 64, 4

m = YOLOMM(CKPT)
model = m.model.float().cuda().eval()
CAP = {}
model.model[0].register_forward_hook(lambda mm, inp, out: CAP.__setitem__("rgb", out.detach()))
model.model[11].register_forward_hook(lambda mm, inp, out: CAP.__setitem__("ir", out.detach()))

files = sorted((ROOT / "images/val").glob("*.png"))[:NF]
LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def load6(rgb_p, dx):
    a = cv2.imread(str(rgb_p))
    b = cv2.imread(str(ROOT / "images_ir/val" / rgb_p.name))
    if b is None:
        b = a.copy()
    a = cv2.resize(a, (IM, IM))
    b = cv2.resize(b, (IM, IM))
    if dx:
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        b = cv2.warpAffine(b, M, (IM, IM), borderValue=(114, 114, 114))
    x = np.ascontiguousarray(np.concatenate([a, b], axis=2)[:, :, ::-1])
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0


def ncc_curve(a, b, dw):
    b = torch.roll(b, shifts=(0, dw), dims=(2, 3)) if dw else b
    af, bf = a.flatten(2), b.flatten(2)
    num = (af * bf).sum(2)
    den = af.pow(2).sum(2).clamp_min(1e-8).sqrt() * bf.pow(2).sum(2).clamp_min(1e-8).sqrt()
    return (num / den).mean(1)  # 通道平均后的逐样本相关


for dx in (+8, 0, -8):
    peaks, tops = [], []
    with torch.no_grad():
        for s in range(0, len(files), BS):
            x = torch.stack([load6(p, dx) for p in files[s:s + BS]]).cuda()
            model(x)
            cur = torch.stack([ncc_curve(CAP["rgb"], CAP["ir"], l) for l in LAGS], 1)
            peaks += (cur.argmax(1).float() + 0).cpu().tolist()
            tops += cur.max(1).values.cpu().tolist()
    peaks = np.array(peaks)
    print(f"shift={dx:+d}px  峰值滞后(cell) mean={peaks.mean():.2f} med={np.median(peaks):.1f} "
          f"峰相关={np.mean(tops):.4f}  argmax 分布={np.bincount(peaks.astype(int), minlength=len(LAGS)).tolist()}")
