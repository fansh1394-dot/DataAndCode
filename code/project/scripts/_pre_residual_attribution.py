# -*- coding: utf-8 -*-
"""预实验 Step 2：残差归因实验。

目的：判定 A(seq)+B·cy+ε 模型扣除后的剩余残差，是
  (H1) 几何项——与目标位置/尺寸/序列/时间等可观测变量相关，可继续建模
  (H2) 随机项——与上述变量无关，属标注不一致或部位差异

方法：
  1. 拟合 dy = A(seq) + B·cy + C·cx + D·h + E·t  多元线性模型
  2. 检验各项系数是否显著（t 检验），报告 R²
  3. 对残差做多因素相关性分析（位置、尺寸、时间、序列）
  4. 残差的正态性、自相关、跨序列稳定性
  5. 关键判定：残差可解释方差占比

若可解释部分 > 50% → H1（几何项，C 不成立）
若可解释部分 < 20% → H2（随机项，C 有空间）
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
        if len(s) >= 5:
            out.append([float(x) for x in s[1:5]])
    return np.array(out, dtype=np.float64).reshape(-1, 4)


def pair_one(r, i):
    from scipy.optimize import linear_sum_assignment
    ca, cb = r[:, :2], i[:, :2]
    cost = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    sa, sb = r[:, 2:] * r[:, 2:], i[:, 2:] * i[:, 2:]
    cost = cost + 0.5 * ((np.sqrt(sa)[:, None, :] - np.sqrt(sb)[None, :, :]) ** 2).sum(-1)
    rr, cc = linear_sum_assignment(cost)
    return list(zip(rr.tolist(), cc.tolist()))


def build(split, sample):
    meta = json.loads((ROOT / 'metadata' / f'{split}.json').read_text())['frames']
    rgb_dir, ir_dir = ROOT / 'labels' / split, ROOT / 'labels_ir' / split
    stems = sorted(p.stem for p in rgb_dir.glob('*.txt'))
    stems = [s for s in stems if meta.get(s, {}).get('kind') == 'both']
    if sample and len(stems) > sample:
        stems = sorted(np.random.default_rng(0).choice(stems, sample, replace=False).tolist())

    rows = []
    for stem in stems:
        r, i = read_txt(rgb_dir / f'{stem}.txt'), read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None or len(r) == 0 or len(r) != len(i):
            continue
        m = meta[stem]
        seq, fi = m['sequence'], m.get('frame_index', 0)
        for ri, ii in pair_one(r, i):
            rb, ib = r[ri], i[ii]
            rows.append(dict(
                stem=stem, seq=seq, fi=fi,
                cx=rb[0], cy=rb[1], h=rb[3] * H, w=rb[2] * W,
                aspect=rb[2] / max(rb[3], 1e-9),
                dx=(ib[0] - rb[0]) * W, dy=(ib[1] - rb[1]) * H,
                ir_cy=ib[1], ir_h=ib[3] * H,
            ))
    return rows


def ols(X, y, names):
    """带标准误与 t 值的 OLS。"""
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / max(n - k, 1)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 1e-12))
    t = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
    print(f'\n  {"项":16s} {"系数":>12s} {"标准误":>10s} {"t值":>8s} {"显著":>6s}')
    for nm, b, s, tv in zip(names, beta, se, t):
        sig = '***' if abs(tv) > 3.29 else ('**' if abs(tv) > 2.58 else ('*' if abs(tv) > 1.96 else ''))
        print(f'  {nm:16s} {b:12.5f} {s:10.5f} {tv:8.2f} {sig:>6s}')
    print(f'  R² = {r2:.4f}  调整R² = {adj:.4f}  残差std = {resid.std():.2f} px  n={n}')
    return beta, resid, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--sample', type=int, default=8000)
    args = ap.parse_args()

    rows = build(args.split, args.sample)
    n = len(rows)
    seqs = sorted({r['seq'] for r in rows})
    print(f'=== 残差归因实验  split={args.split}  目标数={n}  序列数={len(seqs)} ===')

    cy = np.array([r['cy'] for r in rows])
    cx = np.array([r['cx'] for r in rows])
    hh = np.array([r['h'] for r in rows])
    dy = np.array([r['dy'] for r in rows])
    fi = np.array([r['fi'] for r in rows], dtype=float)

    # ---------- 模型 M0：仅序列截距 + cy ----------
    seq_idx = {s: k for k, s in enumerate(seqs)}
    D = np.zeros((n, len(seqs)))
    for k, r in enumerate(rows):
        D[k, seq_idx[r['seq']]] = 1.0

    print('\n--- M0: dy = A(seq) + B·cy ---')
    X0 = np.hstack([D, (cy - 0.5).reshape(-1, 1)])
    names0 = [f'A[{i}]' for i in range(len(seqs))] + ['B·cy']
    beta0, res0, r2_0 = ols(X0, dy, names0)

    # ---------- 模型 M1：加位置、尺寸、时间 ----------
    print('\n--- M1: dy = A(seq) + B·cy + C·cx + D·h + E·aspect + F·t ---')
    X1 = np.hstack([
        D,
        (cy - 0.5).reshape(-1, 1),
        (cx - cx.mean()).reshape(-1, 1),
        ((hh - hh.mean()) / 10.0).reshape(-1, 1),
        (hh / 28.0).reshape(-1, 1) ** 2,
        (fi - fi.mean()).reshape(-1, 1),
    ])
    names1 = [f'A[{i}]' for i in range(len(seqs))] + ['B·cy', 'C·cx', 'D·h', 'D2·h²', 'F·t']
    beta1, res1, r2_1 = ols(X1, dy, names1)

    # ---------- 模型 M2：序列内二次 + 交互 ----------
    print('\n--- M2: 再加 cy²、cy·cx、cy·h ---')
    X2 = np.hstack([X1, ((cy - 0.5) ** 2).reshape(-1, 1),
                    ((cy - 0.5) * (cx - cx.mean())).reshape(-1, 1),
                    ((cy - 0.5) * (hh / 28.0)).reshape(-1, 1)])
    names2 = names1 + ['B2·cy²', 'B3·cy·cx', 'B4·cy·h']
    beta2, res2, r2_2 = ols(X2, dy, names2)

    # ---------- 残差诊断 ----------
    print('\n=== 残差诊断（基于 M2） ===')
    print(f'  残差均值 {res2.mean():+.3f} px   std {res2.std():.3f} px')
    print(f'  残差分位 p5={np.percentile(res2,5):+.2f}  p25={np.percentile(res2,25):+.2f}  '
          f'中位={np.median(res2):+.2f}  p75={np.percentile(res2,75):+.2f}  p95={np.percentile(res2,95):+.2f}')
    from scipy import stats as sps
    sk, ku = sps.skew(res2), sps.kurtosis(res2)
    print(f'  偏度 {sk:+.3f}  峰度 {ku:+.3f}（正态为 0）')

    # 残差与候选变量相关性
    print('\n  残差与各变量的相关性：')
    for nm, v in [('cx', cx), ('cy', cy), ('h', hh), ('aspect', np.array([r['aspect'] for r in rows])),
                  ('frame_index', fi), ('dx', np.array([r['dx'] for r in rows]))]:
        c = np.corrcoef(v, res2)[0, 1]
        print(f'    {nm:14s} r = {c:+.4f}')

    # 序列间残差均值差异（方差分解）
    grand = res2.mean()
    ss_between = ss_within = 0.0
    per_seq = []
    for s in seqs:
        m = np.array([r['seq'] == s for r in rows])
        if m.sum() < 10:
            continue
        v = res2[m]
        ss_between += len(v) * (v.mean() - grand) ** 2
        ss_within += ((v - v.mean()) ** 2).sum()
        per_seq.append((s, len(v), v.mean(), v.std()))
    tot = ss_between + ss_within
    print(f'\n  残差方差分解：序列间 {ss_between/tot*100:.1f}% | 序列内 {ss_within/tot*100:.1f}%')
    per_seq.sort(key=lambda x: -abs(x[2]))
    print('  残差均值最大的 5 个序列：')
    for s, cnt, mu, sd in per_seq[:5]:
        print(f'    {s:26s} n={cnt:5d}  mean={mu:+8.3f}  std={sd:6.3f}')

    # 一阶自相关（同序列内按帧序）
    acf = []
    for s in seqs:
        sub = sorted([r for r in rows if r['seq'] == s], key=lambda r: r['fi'])
        if len(sub) < 20:
            continue
        m = np.array([r['seq'] == s for r in rows])
        v = res2[m]
        v = v - v.mean()
        if v.std() < 1e-9:
            continue
        acf.append(float((v[:-1] * v[1:]).sum() / (v * v).sum()))
    if acf:
        print(f'\n  序列内一阶自相关 ACF(1)：均值 {np.mean(acf):+.4f} 中位 {np.median(acf):+.4f}')

    # ---------- 最终判定 ----------
    print('\n' + '=' * 62)
    print('判定')
    print('=' * 62)
    d_r2 = r2_2 - r2_0
    resid_ratio = res2.std() / dy.std() * 100
    print(f'  M0（仅序列+cy）R² = {r2_0:.4f}')
    print(f'  M2（全模型）  R² = {r2_2:.4f}   增量 {d_r2:+.4f}')
    print(f'  残差 std {res2.std():.2f} px，占 dy 总 std 的 {resid_ratio:.1f}%')
    print()
    if r2_2 > 0.70:
        print('  → H1（可继续建模的几何项）占主导：残差大部分可由可观测变量解释')
        print('    C 的「不可约残差」前提不成立，建议转向 A+B')
    elif r2_2 > 0.50:
        print('  → 混合情形：几何项占多数，但存在不可忽略的未解释部分')
        print('    需进一步看残差是否与序列/时间相关，再决定 C 是否可用')
    else:
        print('  → H2（随机项）占比较高：残差大部分无法由可观测变量解释')
        print('    C 的「不可约残差」前提有支撑，可考虑立项')

    out = Path('/3250410011/root/rivermind-data/MutilModel_3485988898/scripts/_pre_resid_attr.json')
    out.write_text(json.dumps({
        'split': args.split, 'n': n, 'n_seq': len(seqs),
        'R2_M0': r2_0, 'R2_M2': r2_2,
        'resid_std': float(res2.std()), 'dy_std': float(dy.std()),
        'var_between_pct': float(ss_between / tot * 100),
        'acf1_mean': float(np.mean(acf)) if acf else None,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n[SAVED] {out}')


if __name__ == '__main__':
    main()
