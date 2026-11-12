# -*- coding: utf-8 -*-
"""预实验 Step 3：追查 55% 未解释残差的成因。

在 M2 模型（R²=0.693, 残差 std 11.56, ACF(1)=0.45）基础上，把残差拆成四块：

  R_lag   : 相邻帧可预测部分（滞后回归，代表连续物理过程，如姿态漂移）
  R_seq   : 序列内低频趋势（去趋势后剩余）
  R_size  : 与目标尺寸/位置的非线性交互
  R_tail  : 重尾离群（峰度 35 的来源）
  R_white : 真正白噪声

判定：若 R_lag 占比高 → 是未观测的连续自由度（可建模，C 不成立）
      若 R_white 占比高 → 是真随机（不可约，C 有空间）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--sample', type=int, default=8000)
    args = ap.parse_args()

    meta = json.loads((ROOT / 'metadata' / f'{args.split}.json').read_text())['frames']
    rgb_dir, ir_dir = ROOT / 'labels' / args.split, ROOT / 'labels_ir' / args.split
    stems = sorted(p.stem for p in rgb_dir.glob('*.txt'))
    stems = [s for s in stems if meta.get(s, {}).get('kind') == 'both']
    if args.sample and len(stems) > args.sample:
        stems = sorted(np.random.default_rng(0).choice(stems, args.sample, replace=False).tolist())

    rows = []
    for stem in stems:
        r, i = read_txt(rgb_dir / f'{stem}.txt'), read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None or len(r) == 0 or len(r) != len(i):
            continue
        m = meta[stem]
        pr = pair_one(r, i)
        if len(pr) != 1:
            continue
        ri, ii = pr[0]
        rb, ib = r[ri], i[ii]
        rows.append(dict(seq=m['sequence'], fi=m.get('frame_index', 0),
                         cx=rb[0], cy=rb[1], h=rb[3] * H,
                         dy=(ib[1] - rb[1]) * H, dx=(ib[0] - rb[0]) * W))

    print(f'=== Step 3 残差成因追查  split={args.split}  单目标帧={len(rows)} ===')
    seqs = sorted({r['seq'] for r in rows})
    per_seq = defaultdict(list)
    for r in rows:
        per_seq[r['seq']].append(r)
    for s in per_seq:
        per_seq[s].sort(key=lambda r: r['fi'])

    # ---- 重新拟合 M2，得到残差 ----
    seq_idx = {s: k for k, s in enumerate(seqs)}
    n = len(rows)
    D = np.zeros((n, len(seqs)))
    cy = np.array([r['cy'] for r in rows])
    cx = np.array([r['cx'] for r in rows])
    hh = np.array([r['h'] for r in rows])
    dy = np.array([r['dy'] for r in rows])
    fi = np.array([r['fi'] for r in rows], dtype=float)
    for k, r in enumerate(rows):
        D[k, seq_idx[r['seq']]] = 1.0

    X = np.hstack([D, (cy - .5).reshape(-1, 1), (cx - cx.mean()).reshape(-1, 1),
                   ((hh - hh.mean()) / 10).reshape(-1, 1), (hh / 28).reshape(-1, 1) ** 2,
                   (fi - fi.mean()).reshape(-1, 1), ((cy - .5) ** 2).reshape(-1, 1),
                   ((cy - .5) * (cx - cx.mean())).reshape(-1, 1),
                   ((cy - .5) * (hh / 28)).reshape(-1, 1)])
    beta = np.linalg.pinv(X.T @ X) @ X.T @ dy
    resid = dy - X @ beta
    print(f'M2 残差: std={resid.std():.2f}px  total_var={resid.var():.1f}')

    # ---- 1. 滞后回归：用同序列相邻帧的残差预测当前帧 ----
    print('\n--- 1. 时间结构（滞后回归）---')
    print('    假设：若相邻帧残差可预测当前帧，说明残差来自连续物理过程')
    pairs = []
    for s, rs in per_seq.items():
        idxs = [k for k, r in enumerate(rows) if r['seq'] == s]
        if len(idxs) < 10:
            continue
        for a, b in zip(idxs[:-1], idxs[1:]):
            if rows[b]['fi'] - rows[a]['fi'] > 20:  # 跳帧过大则不连
                continue
            pairs.append((resid[a], resid[b]))
    if pairs:
        A = np.array([p[0] for p in pairs])
        Bv = np.array([p[1] for p in pairs])
        r_lag = np.corrcoef(A, Bv)[0, 1]
        coef = np.cov(A, Bv)[0, 1] / max(np.var(A), 1e-9)
        pred = coef * A
        r2_lag = 1 - ((Bv - pred) ** 2).sum() / ((Bv - Bv.mean()) ** 2).sum()
        print(f'    相邻帧对 = {len(pairs)}')
        print(f'    corr(r_t, r_t+1) = {r_lag:+.4f}')
        print(f'    用 r_t 预测 r_t+1 的 R² = {r2_lag:.4f}')
        print(f'    → 可解释当前残差方差约 {r2_lag*100:.1f}%')
    else:
        r2_lag = 0.0
        print('    相邻帧对不足')

    # ---- 2. 序列内低频趋势 ----
    print('\n--- 2. 序列内低频趋势（滑动均值可解释比例）---')
    k = 5
    smooth_var = 0.0
    total_var = 0.0
    for s, rs in per_seq.items():
        idxs = [kk for kk, r in enumerate(rows) if r['seq'] == s]
        if len(idxs) < 2 * k:
            continue
        v = resid[idxs]
        sm = np.convolve(v, np.ones(k) / k, mode='valid')
        # 平滑项方差近似
        smooth_var += sm.var() * (len(sm) - 1)
        total_var += v.var() * (len(v) - 1)
    print(f'    滑动窗口 k={k} 的低频成分占总方差 {smooth_var/max(total_var,1e-9)*100:.1f}%')
    print('    → 低频成分代表序列内的缓慢漂移（如姿态渐变）')

    # ---- 3. 尺寸/位置的残余非线性 ----
    print('\n--- 3. 尺寸与位置的残余依赖（去M2后）---')
    for nm, v in [('h', hh), ('cy', cy), ('cx', cx)]:
        bins = np.percentile(v, [0, 25, 50, 75, 100])
        print(f'    {nm} 分档残差 std：', end='')
        for j in range(4):
            m = (v >= bins[j]) & (v <= bins[j + 1] if j == 3 else v < bins[j + 1])
            if m.sum() > 20:
                print(f'Q{j+1}={resid[m].std():5.2f}  ', end='')
        print()

    # ---- 4. 重尾分离 ----
    print('\n--- 4. 重尾与主体分离 ---')
    sigma = resid.std()
    for thr in (2, 3, 4, 5):
        m = np.abs(resid) > thr * sigma
        print(f'    |r| > {thr}σ: {m.sum():5d} 个 ({m.sum()/n*100:5.2f}%)  '
              f'贡献方差 {((resid[m]**2).sum()/(resid**2).sum())*100:5.1f}%')
    core = resid[np.abs(resid) <= 3 * sigma]
    print(f'    剔除 3σ 外离群后：std = {core.std():.2f}px（原 {sigma:.2f}）')
    print(f'    主体残差占原方差 {core.var()*(len(core))/(resid.var()*n)*100:.1f}%')

    # ---- 5. 序列贡献 ----
    print('\n--- 5. 各序列残差离散度（前 8）---')
    stats = []
    for s, rs in per_seq.items():
        idxs = [kk for kk, r in enumerate(rows) if r['seq'] == s]
        if len(idxs) >= 20:
            v = resid[idxs]
            stats.append((s, len(v), v.std()))
    stats.sort(key=lambda x: -x[2])
    for s, cnt, sd in stats[:8]:
        print(f'    {s:26s} n={cnt:5d}  std={sd:6.2f}')

    # ---- 6. 结论 ----
    print('\n' + '=' * 60)
    print('成因分解（占原残差方差的近似比例）')
    print('=' * 60)
    r2_lag_v = max(r2_lag, 0)
    print(f'  ① 帧间连续性可解释 : {r2_lag_v*100:5.1f}%    ← 连续物理过程，可建模')
    print(f'  ② 序列内低频趋势   : {smooth_var/max(total_var,1e-9)*100:5.1f}%    ← 缓慢漂移，可建模')
    print(f'  ③ 重尾离群         : {((resid[np.abs(resid)>3*sigma]**2).sum()/(resid**2).sum())*100:5.1f}%    ← 少数异常帧')
    print(f'  ④ 其余（近似白噪声）: 由 100% 减去上述')
    print()
    if r2_lag_v > 0.30:
        print('  → 主要成因是「未观测的连续自由度」（如姿态），不是随机误差')
        print('    C 的不可约前提不成立；A 应表述为「未观测姿态自由度导致的偏移」')
    elif r2_lag_v > 0.15:
        print('  → 连续自由度与随机成分各占一部分，需进一步分离')
    else:
        print('  → 以随机成分为主，C 的不可约前提有支撑')


if __name__ == '__main__':
    main()
