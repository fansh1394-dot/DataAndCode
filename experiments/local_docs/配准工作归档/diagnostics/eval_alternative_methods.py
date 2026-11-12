# -*- coding: utf-8 -*-
r"""替代方案可行性评估。

核心问题：能否**自动**建立两模态的对应关系，绕开人工标点？

三条候选路线：
  A. 用模型在 RGB 上的检测框中心 作为 RGB 侧真值（现用方案）
  B. 用红外热斑质心 作为 IR 侧真值，自动配准
  C. 相位相关 / 互信息 直接配准两幅图（无需目标可见）

本脚本评估 A 与 B 的稳定性，并直接给出"用热斑自动定位"的可行性与精度上界。
"""
import glob
import json
from pathlib import Path

import cv2
import numpy as np

RGB_DIR = Path(r'D:\uav_data\images\val')
IR_DIR = Path(r'D:\uav_data\images_ir\val')


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def hotspot_centroid(ir):
    """红外热斑质心（归一化）。用背景减除找最热区域。"""
    h, w = ir.shape[:2]
    g = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    res = np.clip(g - bg, 0, None)
    # 限制在画面中心区域，避开边框/HUD
    mask = np.zeros((h, w), bool)
    mask[int(h * 0.15):int(h * 0.95), int(w * 0.15):int(w * 0.95)] = True
    res = np.where(mask, res, 0)
    idx = np.argsort(res.ravel())[::-1][:80]
    ys, xs = np.unravel_index(idx, res.shape)
    # 只保留最热点的连通近邻，避免多个热源平均
    cx0, cy0 = xs[0], ys[0]
    d = np.hypot(xs - cx0, ys - cy0)
    sel = d < 40
    return xs[sel].mean() / w, ys[sel].mean() / h


def rgb_salient(rgb):
    """可见光目标候选：用暗/亮对比找小目标（天空背景下的小暗点）。"""
    h, w = rgb.shape[:2]
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    # 双向：既可能是暗目标也可能亮目标
    res = np.abs(g - bg)
    mask = np.zeros((h, w), bool)
    mask[int(h * 0.15):int(h * 0.95), int(w * 0.15):int(w * 0.95)] = True
    res = np.where(mask, res, 0)
    idx = np.argsort(res.ravel())[::-1][:80]
    ys, xs = np.unravel_index(idx, res.shape)
    cx0, cy0 = xs[0], ys[0]
    d = np.hypot(xs - cx0, ys - cy0)
    sel = d < 40
    return xs[sel].mean() / w, ys[sel].mean() / h


stems = sorted(p.stem for p in RGB_DIR.iterdir() if p.suffix.lower() == '.jpg')
print(f'验证集共 {len(stems)} 对')

rng = np.random.RandomState(0)
pick = rng.choice(len(stems), size=min(400, len(stems)), replace=False)

print('\n' + '=' * 70)
print('方案 B 可行性：自动提取"可见光目标" 与 "红外热斑" 的对应关系')
print('=' * 70)

rows = []
for k in pick:
    st = stems[k]
    rp = RGB_DIR / f'{st}.jpg'
    ip = IR_DIR / f'{st}.jpg'
    r = rd(rp)
    i = rd(ip)
    if r is None or i is None:
        continue
    # 亮度过滤：跳过夜间
    if cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).mean() < 80:
        continue
    rx, ry = rgb_salient(r)
    ix, iy = hotspot_centroid(i)
    rows.append((st, rx, ry, ix, iy))

print(f'白天可用样本: {len(rows)}')

if rows:
    A = np.array([[r[1], r[2]] for r in rows])
    B = np.array([[r[3], r[4]] for r in rows])
    d = B - A
    print(f'\n自动提取点对 的位移统计（IR - RGB，归一化）:')
    print(f'  x: 中位 {np.median(d[:,0])*100:+7.2f} pp   std {d[:,0].std()*100:6.2f} pp')
    print(f'  y: 中位 {np.median(d[:,1])*100:+7.2f} pp   std {d[:,1].std()*100:6.2f} pp')
    print(f'  距离: 中位 {np.median(np.hypot(d[:,0]*640, d[:,1]*512)):6.2f} px  '
          f'p90 {np.percentile(np.hypot(d[:,0]*640, d[:,1]*512), 90):6.2f} px')

    # 拟合仿射后残差
    Ab = np.hstack([A, np.ones((len(A), 1))])
    cx, *_ = np.linalg.lstsq(Ab, B[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(Ab, B[:, 1], rcond=None)
    pred = np.stack([Ab @ cx, Ab @ cy], axis=1)
    e = pred - B
    epx = np.hypot(e[:, 0] * 640, e[:, 1] * 512)
    print(f'\n  拟合仿射后残差: 中位 {np.median(epx):.2f} px  '
          f'p90 {np.percentile(epx, 90):.2f} px')
    print(f'  仿射_x = {np.round(cx, 5).tolist()}')
    print(f'  仿射_y = {np.round(cy, 5).tolist()}')

    # 5 折 CV
    idx = np.arange(len(rows))
    np.random.RandomState(1).shuffle(idx)
    allp = np.zeros_like(B)
    for g in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, g)
        if len(tr) < 6:
            continue
        Ab2 = np.hstack([A[tr], np.ones((len(tr), 1))])
        c1, *_ = np.linalg.lstsq(Ab2, B[tr, 0], rcond=None)
        c2, *_ = np.linalg.lstsq(Ab2, B[tr, 1], rcond=None)
        Ag = np.hstack([A[g], np.ones((len(g), 1))])
        allp[g] = np.stack([Ag @ c1, Ag @ c2], axis=1)
    ecv = allp - B
    epcv = np.hypot(ecv[:, 0] * 640, ecv[:, 1] * 512)
    print(f'  5折CV 留出集残差: 中位 {np.median(epcv):.2f} px  '
          f'p90 {np.percentile(epcv, 90):.2f} px')
