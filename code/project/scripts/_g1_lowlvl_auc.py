# -*- coding: utf-8 -*-
"""低层特征（stride 2/4/8）上复测「错位可分性」：0px vs 8px。

配对：RGB 骨干 layer 0/1/3 与 IR 骨干 layer 11/12/14（通道数一致）。
对每对特征做逐通道归一化互相关在滞后网格上求峰，输出多种描述子并算 AUC。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
from ultralytics import YOLOMM  # noqa: E402

CKPT = f"{PROJ}/YOLO26/ResTest/mdp1_G1probe/weights/best.pt"
ROOT = Path("/3250410011/root/rivermind-data/mendeley_split")
IM, NF, BS = 640, 64, 4
PAIRS = [(0, 11, 2), (1, 12, 4), (3, 14, 8)]

m = YOLOMM(CKPT)
model = m.model.float().cuda().eval()
CAP = {}
for idx in (0, 1, 3, 11, 12, 14):
    model.model[idx].register_forward_hook(
        lambda mm, inp, out, i=idx: CAP.__setitem__(i, out.detach()))

files = sorted((ROOT / "images/val").glob("*.png"))[:NF]


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


LAGS = [0, 1, 2, 3, 4]


def ncc_at(a, b, dh, dw):
    """逐通道归一化互相关（b 相对 a 平移 dh,dw）。"""
    if dh or dw:
        b = torch.roll(b, shifts=(dh, dw), dims=(2, 3))
    af, bf = a.flatten(2), b.flatten(2)
    num = (af * bf).sum(2)
    den = af.pow(2).sum(2).clamp_min(1e-8).sqrt() * bf.pow(2).sum(2).clamp_min(1e-8).sqrt()
    return num / den  # (N, C)


def desc_pair(r, ir):
    n0 = ncc_at(r, ir, 0, 0).mean(1)
    xs = {l: ncc_at(r, ir, 0, l).mean(1) for l in LAGS}
    ys = {l: ncc_at(r, ir, l, 0).mean(1) for l in LAGS}
    # 峰值滞后（每样本独立取值）
    cxs = torch.stack([xs[l] for l in LAGS], 1)
    cys = torch.stack([ys[l] for l in LAGS], 1)
    # 用滞后 1..4 相对滞后 0 的下降量作"错位"指标
    dropx = n0 - torch.stack([xs[l] for l in LAGS[1:]], 1).max(1).values
    dropy = n0 - torch.stack([ys[l] for l in LAGS[1:]], 1).max(1).values
    # 峰位置（argmax 滞后）
    ax = cxs[:, 1:].argmax(1).float() + 1
    ay = cys[:, 1:].argmax(1).float() + 1
    res = (r - ir).pow(2).mean((1, 2, 3)).sqrt() / ir.pow(2).mean((1, 2, 3)).clamp_min(1e-8).sqrt()
    gx_r = (r[:, :, 1:] - r[:, :, :-1]).abs().mean((1, 2, 3))
    gx_i = (ir[:, :, 1:] - ir[:, :, :-1]).abs().mean((1, 2, 3))
    return {
        "ncc0": n0, "drop_x": dropx, "drop_y": dropy, "drop_xy": (dropx + dropy) / 2,
        "peak_lag_x": ax, "peak_lag_y": ay, "res": res, "grad_ratio": gx_r / gx_i.clamp_min(1e-8),
    }


def collect(dx):
    out = {p: {k: [] for k in ("ncc0", "drop_x", "drop_y", "drop_xy", "peak_lag_x", "peak_lag_y", "res", "grad_ratio")} for p in PAIRS}
    with torch.no_grad():
        for s in range(0, len(files), BS):
            x = torch.stack([load6(p, dx) for p in files[s:s + BS]]).cuda()
            model(x)
            for (lr, li, st) in PAIRS:
                d = desc_pair(CAP[lr], CAP[li])
                for k, v in d.items():
                    out[(lr, li, st)][k] += v.cpu().tolist()
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


A, B = collect(0), collect(8)
for (lr, li, st), d in A.items():
    print(f"--- RGB L{lr} / IR L{li} (stride {st}, {NF} 帧)")
    for k in d:
        a = auc(B[(lr, li, st)][k], A[(lr, li, st)][k])
        m0, m8 = np.mean(A[(lr, li, st)][k]), np.mean(B[(lr, li, st)][k])
        print(f"    {k:12s} AUC={a:.3f}  mean0={m0:.4f} mean8={m8:.4f}")
