# -*- coding: utf-8 -*-
"""测量融合拼接输入（rgb/ir 两半特征）对 IR 位移的响应强度。

目的：判断门控输入描述子（GAP 余弦 / 逐像素相关）中，哪一个真的能区分 0/2/4/8px 错位。
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
IM = 640

m = YOLOMM(CKPT)
model = m.model.float().cuda().eval()
gates = [mod for mod in model.modules() if isinstance(mod, MisalignGatedConcat)]
CAP = {}

for i, g in enumerate(gates):
    def hook(mod, inp, out, idx=i):
        r, ir = inp[0]
        CAP[idx] = (r.detach(), ir.detach())
    g.register_forward_hook(hook)

rgb_files = sorted((ROOT / "images/val").glob("*.png"))[:8]


def load6(rgb_p, dx, dy=0):
    a = cv2.imread(str(rgb_p))
    b = cv2.imread(str(ROOT / "images_ir/val" / rgb_p.name))
    if b is None:
        b = a.copy()
    a = cv2.resize(a, (IM, IM))
    b = cv2.resize(b, (IM, IM))
    if dx or dy:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        b = cv2.warpAffine(b, M, (IM, IM), borderValue=(114, 114, 114))
    x = np.ascontiguousarray(np.concatenate([a, b], axis=2)[:, :, ::-1])
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0


def stats(r, ir):
    """返回一组对位移敏感的标量。"""
    g_r = F.adaptive_avg_pool2d(r, 1).flatten(1)
    g_i = F.adaptive_avg_pool2d(ir, 1).flatten(1)
    cos_gap = F.cosine_similarity(g_r, g_i, dim=1, eps=1e-6).mean()
    absdiff_gap = (g_r - g_i).abs().mean()

    # 逐通道零延迟归一化互相关（相对能量），比全局余弦更能反映空间错位
    rf = r.flatten(2)
    irf = ir.flatten(2)
    num = (rf * irf).sum(2)
    den = (rf.pow(2).sum(2) * irf.pow(2).sum(2)).clamp_min(1e-8).sqrt()
    ncc0 = (num / den).mean()

    # 逐像素残差能量（相对幅度）
    res = (r - ir).pow(2).mean().sqrt() / ir.pow(2).mean().sqrt().clamp_min(1e-8)

    # 梯度幅值差（对位移敏感的结构一致度）
    gx_r = r[:, :, 1:] - r[:, :, :-1]
    gx_i = ir[:, :, 1:] - ir[:, :, :-1]
    grad_diff = (gx_r - gx_i).abs().mean() / gx_r.abs().mean().clamp_min(1e-8)
    return cos_gap.item(), absdiff_gap.item(), ncc0.item(), res.item(), grad_diff.item()


for dx in (0, 2, 4, 8):
    x = torch.stack([load6(p, dx) for p in rgb_files]).cuda()
    with torch.no_grad():
        model(x)
    rows = []
    for idx, (r, ir) in CAP.items():
        c_gap, ad, ncc, res, gd = stats(r, ir)
        a_mean = gates[idx].last_alpha_mean
        rows.append((idx, a_mean, c_gap, ad, ncc, res, gd))
    print(f"--- shift={dx}px")
    for idx, a, c_gap, ad, ncc, res, gd in rows:
        print(f"  L{idx}: alpha={a:.4f} cos_gap={c_gap:.4f} absdiff_gap={ad:.5f} "
              f"ncc0={ncc:.4f} res={res:.4f} grad_diff={gd:.4f}")
