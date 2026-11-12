# -*- coding: utf-8 -*-
"""验证：序列级 A + 解析 B 的对齐，是否优于 per-frame 单帧拟合（原 oracle 做法）。

背景：
  - 原 oracle（v2c_R1_reg_early_s0）用 mode=per-frame + oracle=True，
    即每帧用该帧自身标注拟合仿射，构造性完美（frame_err=0），
    但逐序列固定仿射达标率仅 12.4%，且早期融合仍 -0.0010。
  - 本轮分析发现真实偏移机制为：dy = A(seq) + B·cy + ε
      B ≈ -120~-130 px/单位 cy（跨序列稳定，可解析计算）
      A(seq)（序列级常数，用前 N 帧在线估计即收敛）
      ε ≈ 5 px 帧内随机抖动（不可消除）

本脚本对应三个「对齐方案」，在同一批帧上比较对 IR 框的预测精度：
  P0 per-frame 单帧拟合：用该帧自身的 RGB/IR 框对拟合仿射，再预测（原 oracle）
  P1 序列级 A + 解析 B：Δ = A_seq + B·(cy - 0.5)，纯平移预测
  P2 序列级仿射：用该序列全部 both 帧的共轭点对拟合一个 2x3 仿射（非 oracle，纯标签）

评价指标：预测 IR 框中心到真实 IR 框中心的误差（px），以及派生 IoU。
注意：P0 用的是「同一帧的标签」，严格说不是预测而是重拟合；
      P1/P2 用的是「序列级信息」，更接近真实可用的对齐先验。

用法：python scripts/_verify_align_oracle.py [--split test] [--fit-n N]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/rivermind-data/uav_data_v2')
W, H = 640.0, 512.0


def read_txt(p):
    if not p.exists():
        return None
    out = []
    for line in p.read_text().splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        out.append([float(x) for x in s[1:5]])
    return np.array(out, dtype=np.float64).reshape(-1, 4)


def hungarian_pair(a, b):
    from scipy.optimize import linear_sum_assignment
    ca, cb = a[:, :2], b[:, :2]
    cost = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    sa, sb = a[:, 2:] * a[:, 2:], b[:, 2:] * b[:, 2:]
    cost = cost + 0.5 * ((np.sqrt(sa)[:, None, :] - np.sqrt(sb)[None, :, :]) ** 2).sum(-1)
    return list(zip(*[x.tolist() for x in linear_sum_assignment(cost)]))


def to_px(boxes):
    """归一化 cxcywh -> 像素 x1y1x2y2"""
    cx, cy, w, h = boxes[:, 0] * W, boxes[:, 1] * H, boxes[:, 2] * W, boxes[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def iou_px(a, b):
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def fit_affine(src, dst):
    """src,dst: (N,2)。返回 2x3。

    注意：若点集退化（如全部共线/近共线），最小二乘会得到病态解。
    这里做条件数检查，退化时返回 None 由调用方降级。
    """
    A = np.hstack([src, np.ones((len(src), 1))])
    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > 1e9:
        return None
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return M.T


def apply_affine(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--fit-n', type=int, default=8, help='P1/P2 在线估计所用的前 N 帧数（模拟部署）')
    args = ap.parse_args()

    meta = json.loads((ROOT / 'metadata' / f'{args.split}.json').read_text())['frames']
    rgb_dir, ir_dir = ROOT / 'labels' / args.split, ROOT / 'labels_ir' / args.split
    stems = [p.stem for p in rgb_dir.glob('*.txt')]

    # 收集每个序列的帧（按 frame_index 排序）
    per_seq = defaultdict(list)
    for stem in stems:
        m = meta.get(stem, {})
        if m.get('kind') != 'both':
            continue
        r = read_txt(rgb_dir / f'{stem}.txt')
        i = read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None or len(r) == 0 or len(r) != len(i):
            continue
        pairs = hungarian_pair(r, i)
        per_seq[m['sequence']].append((m.get('frame_index', 0), r, i, pairs))

    print(f'=== split={args.split} 序列 {len(per_seq)}  在线估计窗口 N={args.fit_n} ===\n')

    # 先估计全局解析 B（用全部序列的中心化回归）
    all_cy, all_dy = [], []
    for seq, frames in per_seq.items():
        for _, r, i, pairs in frames:
            for ri, ii in pairs:
                all_cy.append(r[ri][1])
                all_dy.append((i[ii][1] - r[ri][1]) * H)
    all_cy, all_dy = np.array(all_cy), np.array(all_dy)
    # 序列内中心化，剔除序列常数
    resid_cy, resid_dy = [], []
    for seq, frames in per_seq.items():
        c, d = [], []
        for _, r, i, pairs in frames:
            for ri, ii in pairs:
                c.append(r[ri][1]); d.append((i[ii][1] - r[ri][1]) * H)
        c, d = np.array(c), np.array(d)
        if len(c) > 20:
            resid_cy.extend((c - c.mean()).tolist())
            resid_dy.extend((d - d.mean()).tolist())
    rc, rd = np.array(resid_cy), np.array(resid_dy)
    B = float(np.cov(rc, rd)[0, 1] / np.var(rc)) if rc.std() > 1e-9 else 0.0
    print(f'解析视差斜率 B = {B:.2f} px / 单位 cy（序列内中心化回归）\n')

    schemes = ['P0 per-frame 单帧拟合', 'P1 A(seq)+B·cy', 'P2 序列级仿射']
    errs = {s: [] for s in schemes}
    ious = {s: [] for s in schemes}
    n_eval = 0
    diag = []
    n_affine_ok = 0

    for seq, frames in per_seq.items():
        frames.sort(key=lambda x: x[0])
        if len(frames) <= args.fit_n:
            continue
        fit_part, eval_part = frames[:args.fit_n], frames[args.fit_n:]

        # --- 序列级参数：只用前 N 帧（模拟在线估计） ---
        A_est = []
        src_pts, dst_pts = [], []
        for _, r, i, pairs in fit_part:
            for ri, ii in pairs:
                A_est.append((i[ii][1] - r[ri][1]) * H - B * (r[ri][1] - 0.5))
                src_pts.append([r[ri][0] * W, r[ri][1] * H])
                dst_pts.append([i[ii][0] * W, i[ii][1] * H])
        A_seq = float(np.mean(A_est)) if A_est else 0.0
        # 序列级仿射只在点分布充分（x/y 都有跨度）时可用；否则病态，降级为纯平移
        sp = np.array(src_pts)
        usable = (len(src_pts) >= 6 and sp[:, 0].std() > 8.0 and sp[:, 1].std() > 8.0)
        M_seq = fit_affine(sp, np.array(dst_pts)) if usable else None
        n_affine_ok += int(M_seq is not None)
        # 诊断：序列级仿射在「估计用帧」上的残差，以及 A 的离散度
        if M_seq is not None:
            fit_res = np.linalg.norm(apply_affine(M_seq, np.array(src_pts)) - np.array(dst_pts), axis=1)
            diag.append((seq, len(fit_part), float(np.std(A_est)), float(np.median(fit_res)),
                         float(sp[:, 0].std()), float(sp[:, 1].std())))
        else:
            diag.append((seq, len(fit_part), float(np.std(A_est)), float('nan'),
                         float(sp[:, 0].std()), float(sp[:, 1].std())))

        for _, r, i, pairs in eval_part:
            nb_r, nb_i = to_px(r), to_px(i)

            # P0：原 oracle 做法 = per-frame 用本帧自身拟合 warp。
            # 注意每帧只有 1 个目标（多目标帧数 = 0），单点无法拟合仿射；
            # 原 oracle 实际是「用该帧标签算出精确位移」，故这里取该帧的真实位移，
            # 即构造性完美（误差恒为 0），这是它的定义性质，作为上界参照。
            for idx, (ri, ii) in enumerate(pairs):
                n_eval += 1
                rgb_c = np.array([[r[ri][0] * W, r[ri][1] * H]])
                ir_c_true = np.array([[i[ii][0] * W, i[ii][1] * H]])

                # P0：oracle 完美对齐（误差 0）
                pred0 = ir_c_true.copy()
                # P1：序列级 A + 解析 B
                pred1 = np.array([[rgb_c[0][0], rgb_c[0][1] + A_seq + B * (r[ri][1] - 0.5)]])
                # P2：序列级仿射（用前 N 帧共轭点对拟合）
                pred2 = apply_affine(M_seq, rgb_c) if M_seq is not None else pred1

                for name, pred in zip(schemes, (pred0, pred1, pred2)):
                    errs[name].append(float(np.linalg.norm(pred[0] - ir_c_true[0])))
                    # 用预测中心 + 原 RGB 框尺寸构造预测框，算与真实 IR 框的 IoU
                    pb = nb_r[idx:idx + 1].copy()
                    pw = pb[0, 2] - pb[0, 0]; ph = pb[0, 3] - pb[0, 1]
                    predbox = np.array([[pred[0][0] - pw / 2, pred[0][1] - ph / 2,
                                         pred[0][0] + pw / 2, pred[0][1] + ph / 2]])
                    ious[name].append(float(iou_px(predbox, nb_i[idx:idx + 1])[0]))

    print(f'评估目标数 {n_eval}（仅用每序列后 {100}% 帧，前 {args.fit_n} 帧用于估计）\n')
    print(f'  {"方案":24s} {"center_err 中位":>14s} {"mean":>8s} {"p90":>8s} {"IoU 中位":>10s} {"IoU>0.5":>8s}')
    for s in schemes:
        e, u = np.array(errs[s]), np.array(ious[s])
        if e.size == 0:
            continue
        print(f'  {s:24s} {np.median(e):14.2f} {e.mean():8.2f} '
              f'{np.percentile(e, 90):8.2f} {np.median(u):10.3f} {np.mean(u > 0.5)*100:7.1f}%')

    print('\n差值（相对 P1）:')
    for s in schemes:
        if not errs[s]:
            continue
        d = np.median(errs[s]) - np.median(errs['P1 A(seq)+B·cy'])
        print(f'  {s:24s} 中位误差 {d:+.2f} px')

    if diag:
        print('\n--- 诊断（序列级仿射拟合质量）---')
        print(f'  可用仿射的序列 {n_affine_ok}/{len(diag)}'
              f'（其余因前 {args.fit_n} 帧目标分布退化而降级为纯平移）')
        print(f'  {"seq":24s} {"n_fit":>6s} {"std(A)":>8s} {"fit残差中位":>11s} {"std(x)":>8s} {"std(y)":>8s}')
        ok = [d for d in diag if not np.isnan(d[3])]
        for d in sorted(ok, key=lambda x: -x[3])[:8]:
            print(f'  {d[0][:24]:24s} {d[1]:6d} {d[2]:8.2f} {d[3]:11.2f} {d[4]:8.1f} {d[5]:8.1f}')
        if ok:
            print(f'\n  可用仿射序列：拟合残差中位 {np.median([d[3] for d in ok]):.2f} px')
        print(f'  std(A) 中位 {np.median([d[2] for d in diag]):.2f} px'
              f'（前 {args.fit_n} 帧内序列偏置的离散度，越小说明在线估计越容易收敛）')


if __name__ == '__main__':
    main()
