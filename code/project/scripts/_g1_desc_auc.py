# -*- coding: utf-8 -*-
"""判定：融合颈特征里是否存在「可区分 0px vs 8px 错位」的廉价标量描述子。

做法：64 帧 × 两个条件（shift=0 / shift=8），在门控输入（rgb/ir 两半特征）上
计算若干候选描述子，用 AUC（shift8 判为正类）衡量单帧可分性。
AUC≈0.5 表示该描述子无信息；这是门控能否成立的前提，与训练无关。
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
from ultralytics.nn.mm.misalign_gate import MisalignGatedConcat  # noqa: E402

CKPT = f"{PROJ}/YOLO26/ResTest/mdp1_G1probe/weights/best.pt"
ROOT = Path("/3250410011/root/rivermind-data/mendeley_split")
IM, NF, BS = 640, 64, 8

m = YOLOMM(CKPT)
model = m.model.float().cuda().eval()
gates = [mod for mod in model.modules() if isinstance(mod, MisalignGatedConcat)]
CAP = {}
for i, g in enumerate(gates):
    g.register_forward_hook(lambda mod, inp, out, idx=i: CAP.__setitem__(idx, (inp[0][0].detach(), inp[0][1].detach())))

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


def desc(r, ir):
    """逐样本候选描述子（全部只用点乘/池化，端侧代价可忽略）。"""
    g_r = F.adaptive_avg_pool2d(r, 1).flatten(1)
    g_i = F.adaptive_avg_pool2d(ir, 1).flatten(1)
    cos_gap = F.cosine_similarity(g_r, g_i, dim=1, eps=1e-6)
    ad_gap = (g_r - g_i).abs().mean(1)

    rf, irf = r.flatten(2), ir.flatten(2)
    den = (rf.pow(2).sum(2).clamp_min(1e-8).sqrt() * irf.pow(2).sum(2).clamp_min(1e-8).sqrt())

    def ncc(dh, dw):
        b = torch.roll(ir, shifts=(dh, dw), dims=(2, 3)).flatten(2)
        return ((rf * b).sum(2) / den).mean(1)

    n0 = ncc(0, 0)
    # 特征图上的滞后相关（1~2 个 cell ≈ stride×1~2 像素位移）
    nx1, nx2 = ncc(0, 1), ncc(0, 2)
    ny1, ny2 = ncc(1, 0), ncc(2, 0)
    lag_ratio = (nx1 + ny1) / 2 / n0.clamp_min(1e-6).abs()
    lag_drop = n0 - (nx1 + ny1 + nx2 + ny2) / 4

    res = (r - ir).pow(2).mean((1, 2, 3)).sqrt() / ir.pow(2).mean((1, 2, 3)).clamp_min(1e-8).sqrt()
    gx_r = (r[:, :, 1:] - r[:, :, :-1]).abs().mean((1, 2, 3))
    gx_i = (ir[:, :, 1:] - ir[:, :, :-1]).abs().mean((1, 2, 3))
    grad_r = gx_r / gx_i.clamp_min(1e-8)
    return {
        "cos_gap": cos_gap, "absdiff_gap": ad_gap, "ncc0": n0,
        "lag_ratio": lag_ratio, "lag_drop": lag_drop, "res": res, "grad_ratio": grad_r,
    }


def collect(dx):
    out = {i: {k: [] for k in ("cos_gap", "absdiff_gap", "ncc0", "lag_ratio", "lag_drop", "res", "grad_ratio")}
           for i in range(len(gates))}
    with torch.no_grad():
        for s in range(0, len(files), BS):
            x = torch.stack([load6(p, dx) for p in files[s:s + BS]]).cuda()
            model(x)
            for i in range(len(gates)):
                d = desc(*CAP[i])
                for k, v in d.items():
                    out[i][k] += v.cpu().tolist()
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


A = collect(0)
B = collect(8)
for i in range(len(gates)):
    print(f"--- L{i} (AUC shift8 vs shift0, 1.0=完全可分, 0.5=无信息)")
    for k in A[i]:
        a = auc(B[i][k], A[i][k])
        print(f"    {k:12s} AUC={a:.3f}  mean0={np.mean(A[i][k]):.4f} mean8={np.mean(B[i][k]):.4f}")
