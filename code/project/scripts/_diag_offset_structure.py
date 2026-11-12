# -*- coding: utf-8 -*-
"""分析 labels/ 与 labels_ir/ 两套标注框的相对位移结构。

前提（已由 _chk_cls_consistency.py 证实）：both 帧两套标注框数量 100% 一致，
因此可用「一对一」推定对应。本脚本不用贪心 IoU 配对（97% 帧 IoU<0.5 会配错），
改用 Hungarian 最小代价匹配，代价 = 中心距离 + 尺寸差惩罚。

回答四个问题：
  Q1 位移的分布形态（dx/dy 的均值、中位、分位数、同向性）
  Q2 位移是否随目标在画幅中的位置系统变化（与 x/y 的相关性）
  Q3 位移是否随目标尺度变化（与框高/宽的相关性）
  Q4 位移是否与「热斑位置」相关（机体 vs 热源的垂直间距）
  Q5 逐序列一致性：序列内位移是否稳定（可建模）还是帧间乱跳（不可建模）

用法：python scripts/_diag_offset_structure.py [--split test] [--limit N]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/rivermind-data/uav_data_v2')
W, H = 640.0, 512.0  # 仅用于把归一化坐标转成像素；若分辨率不同，报告里同时给归一化值


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
    """a,b: (N,4) 归一化 cxcywh。返回索引对列表。"""
    from scipy.optimize import linear_sum_assignment
    ca, cb = a[:, :2], b[:, :2]
    cost = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    # 尺寸差异也计入代价，避免同位置大小目标错配
    sa, sb = a[:, 2:] * a[:, 2:], b[:, 2:] * b[:, 2:]
    cost = cost + 0.5 * ((np.sqrt(sa)[:, None, :] - np.sqrt(sb)[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def stats(name, v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        print(f'  {name:22s} (空)')
        return
    q = np.percentile(v, [5, 25, 50, 75, 95])
    print(f'  {name:22s} n={v.size:6d} mean={v.mean():8.2f} std={v.std():7.2f} '
          f'med={q[2]:8.2f} | p5={q[0]:7.2f} p25={q[1]:7.2f} p75={q[3]:7.2f} p95={q[4]:7.2f}')


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--spot', action='store_true', help='追加热斑定位分析（读 IR 图，慢）')
    ap.add_argument('--spot-n', type=int, default=1500, help='热斑分析抽样帧数')
    ap.add_argument('--scale', action='store_true', help='追加目标尺度分档验证')
    ap.add_argument('--seq', action='store_true', help='追加跨序列分布验证')
    ap.add_argument('--temporal', action='store_true', help='追加帧内抖动的时间相关性验证')
    args = ap.parse_args()

    meta = json.loads((ROOT / 'metadata' / f'{args.split}.json').read_text())['frames']
    rgb_dir, ir_dir = ROOT / 'labels' / args.split, ROOT / 'labels_ir' / args.split
    stems = sorted(p.stem for p in rgb_dir.glob('*.txt'))
    if args.limit:
        stems = stems[:args.limit]

    # 逐框记录
    rec_dx, rec_dy, rec_dw, rec_dh = [], [], [], []
    rec_cx_rgb, rec_cy_rgb, rec_rgb_h = [], [], []
    rec_dist = []
    seq_of = []
    fidx_of = []                  # 帧序号，用于时间轴排序
    seq_dxy = defaultdict(list)   # seq -> [(dx,dy)]
    n_frames = n_pairs = n_unmatched_frame = 0
    multi_per_frame = 0

    for stem in stems:
        kind = meta.get(stem, {}).get('kind')
        if kind != 'both':
            continue
        r = read_txt(rgb_dir / f'{stem}.txt')
        i = read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None or len(r) == 0 or len(r) != len(i):
            n_unmatched_frame += 1
            continue
        n_frames += 1
        if len(r) > 1:
            multi_per_frame += 1
        mrec = meta.get(stem, {})
        seq = mrec.get('seq') or mrec.get('sequence') or stem.split('_')[0]
        fidx = mrec.get('frame_index')
        if fidx is None:
            try:
                fidx = int(stem.rsplit('_', 1)[-1])
            except ValueError:
                fidx = -1

        for ri, ii in hungarian_pair(r, i):
            rb, ib = r[ri], i[ii]
            dx = (ib[0] - rb[0]) * W
            dy = (ib[1] - rb[1]) * H
            rec_dx.append(dx); rec_dy.append(dy)
            rec_dw.append((ib[2] - rb[2]) * W)
            rec_dh.append((ib[3] - rb[3]) * H)
            rec_cx_rgb.append(rb[0]); rec_cy_rgb.append(rb[1])
            rec_rgb_h.append(rb[3] * H)
            rec_dist.append(np.hypot(dx, dy))
            seq_of.append(seq)
            fidx_of.append(fidx)
            seq_dxy[seq].append((dx, dy))
            n_pairs += 1

    print(f'=== split={args.split} both帧={n_frames} 配对={n_pairs} '
          f'多目标帧={multi_per_frame} 跳过={n_unmatched_frame} ===\n')

    dx, dy = np.array(rec_dx), np.array(rec_dy)
    print('--- Q1 位移分布（像素，IR 减 RGB）---')
    stats('dx', dx); stats('dy', dy)
    stats('|d| 欧氏距离', rec_dist)
    stats('dw (宽差)', rec_dw); stats('dh (高差)', rec_dh)
    print(f'  dy>0 占比 {np.mean(dy > 0)*100:.2f}%   dy<0 占比 {np.mean(dy < 0)*100:.2f}%')
    print(f'  |dx|<2px 占比 {np.mean(np.abs(dx) < 2)*100:.2f}%   '
          f'|dy|<2px 占比 {np.mean(np.abs(dy) < 2)*100:.2f}%')
    print(f'  距离<10px 占比 {np.mean(np.array(rec_dist) < 10)*100:.2f}%')

    print('\n--- Q2 位移 vs 画幅位置（相关系数）---')
    print(f'  corr(dx, cx_rgb)   = {corr(rec_cx_rgb, dx):+.4f}')
    print(f'  corr(dy, cy_rgb)   = {corr(rec_cy_rgb, dy):+.4f}')
    print(f'  corr(dx, cy_rgb)   = {corr(rec_cy_rgb, dx):+.4f}')
    print(f'  corr(dy, cx_rgb)   = {corr(rec_cx_rgb, dy):+.4f}')

    print('\n--- Q3 位移 vs 目标尺度（相关系数）---')
    print(f'  corr(dx, rgb_h_px) = {corr(rec_rgb_h, dx):+.4f}')
    print(f'  corr(dy, rgb_h_px) = {corr(rec_rgb_h, dy):+.4f}')
    print(f'  corr(|d|, rgb_h_px)= {corr(rec_rgb_h, rec_dist):+.4f}')

    print('\n--- Q4 位移归一化到目标尺寸（等价于「偏移了几个目标身位」）---')
    hh = np.array(rec_rgb_h)
    ok = hh > 1e-6
    stats('dy / rgb_h', np.array(rec_dy)[ok] / hh[ok])
    stats('dx / rgb_h', np.array(rec_dx)[ok] / hh[ok])

    print('\n--- Q5 逐序列一致性（序列内位移是否稳定）---')
    rows = []
    for seq, v in seq_dxy.items():
        if len(v) < 20:
            continue
        a = np.array(v)
        # 序列内「帧间抖动」= 去均值后的 std；「系统偏移」= 均值
        rows.append((seq, len(v), a[:, 0].mean(), a[:, 1].mean(),
                     a[:, 0].std(), a[:, 1].std()))
    rows.sort(key=lambda x: -x[1])
    print(f'  {"seq":28s} {"n":>5s} {"mean dx":>9s} {"mean dy":>9s} {"std dx":>8s} {"std dy":>8s}')
    for r in rows[:15]:
        print(f'  {str(r[0])[:28]:28s} {r[1]:5d} {r[2]:9.2f} {r[3]:9.2f} {r[4]:8.2f} {r[5]:8.2f}')
    if rows:
        mdy = np.array([r[3] for r in rows])
        sdy = np.array([r[5] for r in rows])
        print(f'\n  跨序列 mean(dy) 的均值 {mdy.mean():.2f}，跨序列 std {mdy.std():.2f}'
              f'（跨序列差异大 = 偏移随序列/视角变化）')
        print(f'  序列内 std(dy) 的中位数 {np.median(sdy):.2f}'
              f'（序列内抖动小 = 偏移在序列内稳定，可建模）')
        print(f'  同向序列占比 {np.mean(mdy > 0 if mdy.mean() > 0 else mdy < 0)*100:.1f}%')

    if args.temporal:
        run_temporal_analysis(rec_dy, rec_cy_rgb, rec_rgb_h, seq_of, fidx_of)

    if args.seq:
        run_seq_analysis(rec_dx, rec_dy, rec_cx_rgb, rec_cy_rgb, rec_rgb_h, seq_of, meta)

    if args.scale:
        run_scale_analysis(rec_dx, rec_dy, rec_cx_rgb, rec_cy_rgb, rec_rgb_h, seq_of)

    if args.spot:
        run_spot_analysis(seq_dxy, args.split, args.spot_n)


def run_temporal_analysis(rec_dy, rec_cy, rec_h, seq_of, fidx_of):
    """帧内抖动的时间相关性：决定在线估计序列偏置需要多长的窗口。

    做法：
      1) 每个序列内按 frame_index 排序，得到时间序列 dy(t)
      2) 计算滞后 k 帧的自相关 ACF(k)，k=1..20
      3) 去除 cy 的视差项后（dy − B·cy）再算一次，分离「真实偏移抖动」与「目标运动」
      4) 用 ACF 首次跌破 1/e 的滞后量估计有效时间常数 tau
      5) 报告「用前 N 帧均值估计序列偏置」的误差曲线
    """
    dy = np.array(rec_dy); cy = np.array(rec_cy)
    seq = np.array(seq_of); fidx = np.array(fidx_of)

    print('\n--- Q9 帧内抖动的时间相关性 ---')

    # 用全局斜率去除视差项（B 用序列内去均值的中心化回归估计）
    bh = []
    for s in set(seq.tolist()):
        m = seq == s
        if m.sum() < 50:
            continue
        x, y = cy[m], dy[m]
        if x.std() < 1e-9:
            continue
        bh.append(np.cov(x, y)[0, 1] / np.var(x))
    B = float(np.median(bh)) if bh else 0.0
    print(f'  视差斜率 B 的中位估计 = {B:.2f} px / 单位归一化 cy'
          f'（cy 跨全幅≈1.0，即约 {B:.1f} px 的变化幅度）')

    for tag, arr in (('原始 dy', dy), ('去视差后 dy−B·cy', dy - B * cy)):
        acfs, taus, ns = [], [], 0
        for s in sorted(set(seq.tolist())):
            m = seq == s
            if m.sum() < 60:
                continue
            idx = np.argsort(fidx[m], kind='stable')
            v = arr[m][idx]
            v = v - v.mean()
            if v.std() < 1e-9:
                continue
            ns += 1
            denom = (v * v).sum()
            cur = []
            for k in range(1, 21):
                if len(v) - k < 10:
                    break
                cur.append(float((v[:-k] * v[k:]).sum() / denom))
            if len(cur) >= 5:
                acfs.append(cur)
        if not acfs:
            print(f'  {tag}: 可用序列不足')
            continue
        L = min(len(a) for a in acfs)
        mean_acf = np.array([np.mean([a[i] for a in acfs]) for i in range(L)])
        print(f'\n  [{tag}] 序列数 {ns}，平均自相关 ACF(k):')
        print('   ' + '  '.join(f'k={i+1}:{mean_acf[i]:+.3f}' for i in range(min(L, 10))))
        if L > 10:
            print('   ' + '  '.join(f'k={i+1}:{mean_acf[i]:+.3f}' for i in range(10, L)))
        tau = next((i + 1 for i, a in enumerate(mean_acf) if a < np.exp(-1)), None)
        print(f'   ACF 首次跌破 1/e 的滞后 = {tau if tau else f">{L}"} 帧')

    # 用前 N 帧均值估计「序列偏置」的误差（去视差后，仅剩序列级常数 + 帧内抖动）
    r = dy - B * cy
    print('\n  用前 N 帧均值估计序列偏置的误差（去视差后）：')
    print(f'  {"N":>4s} {"median_abs_err":>15s} {"p90_abs_err":>12s} {"相对帧内std":>12s}')
    seqs = []
    for s in set(seq.tolist()):
        m = seq == s
        if m.sum() < 80:
            continue
        idx = np.argsort(fidx[m], kind='stable')
        seqs.append(r[m][idx])
    if seqs:
        within = np.median([np.std(v) for v in seqs])
        for N in (1, 2, 4, 8, 16, 32, 64):
            errs = []
            for v in seqs:
                if len(v) <= N + 20:
                    continue
                mu = v[:N].mean()
                errs.extend(np.abs(v[N:] - mu).tolist())
            if errs:
                e = np.array(errs)
                print(f'  {N:4d} {np.median(e):15.2f} {np.percentile(e, 90):12.2f} '
                      f'{np.median(e)/max(within,1e-9):12.2f}')
        print(f'  参考：序列内 dy 的 std 中位 = {within:.2f} px')


def run_seq_analysis(rec_dx, rec_dy, rec_cx, rec_cy, rec_h, seq_of, meta):
    """跨序列分布验证：37 px 是全局常数，还是序列相关？

    分解 dy 的方差：总方差 = 序列间方差 + 序列内方差。
    若序列间占比高 → 需序列自适应；若序列内占主导 → 全局先验足够。
    并检验序列均 dy 与「序列平均尺度 / 平均画幅位置 / 序列时间戳」是否相关。
    """
    from scipy import stats as sps

    dx = np.array(rec_dx); dy = np.array(rec_dy)
    cx = np.array(rec_cx); cy = np.array(rec_cy); h = np.array(rec_h)
    seq = np.array(seq_of).astype(str)

    uniq = sorted(set(seq.tolist()))
    print(f'\n--- Q8 跨序列分布验证（序列数 {len(uniq)}）---')

    rows = []
    for s in uniq:
        m = seq == s
        if m.sum() < 30:
            continue
        rows.append(dict(seq=s, n=int(m.sum()), mean_dy=dy[m].mean(), med_dy=np.median(dy[m]),
                         std_dy=dy[m].std(), mean_dx=dx[m].mean(), med_h=np.median(h[m]),
                         mean_cy=cy[m].mean(), mean_cx=cx[m].mean()))
    if len(rows) < 3:
        print('  可用序列过少')
        return

    md = np.array([r['mean_dy'] for r in rows])
    sd = np.array([r['std_dy'] for r in rows])
    nn = np.array([r['n'] for r in rows])

    print(f'  序列 mean(dy): 全局均值 {md.mean():.2f} px, 跨序列 std {md.std(ddof=1):.2f} px, '
          f'范围 [{md.min():.2f}, {md.max():.2f}]')
    print(f'  序列内 std(dy): 中位 {np.median(sd):.2f} px, 范围 [{sd.min():.2f}, {sd.max():.2f}]')
    print(f'  序列内 / 跨序列 波动比 = {np.median(sd)/max(md.std(ddof=1),1e-9):.2f}'
          f'（>1 表示帧间抖动大于序列间差异）')

    # 方差分解（按样本数加权）
    grand = dy.mean()
    ss_between = float((nn * (md - grand) ** 2).sum())
    ss_within = float(sum((r['std_dy'] ** 2) * (r['n'] - 1) for r in rows))
    tot = ss_between + ss_within
    print(f'\n  ANOVA 方差分解: 序列间 {ss_between/tot*100:.2f}% | 序列内 {ss_within/tot*100:.2f}%')
    eta2 = ss_between / tot
    print(f'  eta² = {eta2:.4f}（序列可解释的 dy 方差占比）')

    # 单因素 F 检验（序列间均值是否显著不同）
    groups = [dy[seq == s] for s in uniq if (seq == s).sum() >= 30]
    if len(groups) >= 2:
        F, p = sps.f_oneway(*groups)
        print(f'  单因素 ANOVA: F={F:.2f}, p={p:.3e}'
              f'（p 小表示序列间均值确实不同，但这可能只反映微小差异）')
        # 效应量：最大组均值差
        print(f'  最大序列均值差 = {md.max()-md.min():.2f} px，'
              f'相当于 dy 全局 std({dy.std():.2f}) 的 {(md.max()-md.min())/dy.std()*100:.1f}%')

    # 序列内中心化后，若仍能完美预测序列归属 → 可用序列常量
    resid = dy.copy()
    for s in uniq:
        m = seq == s
        if m.sum() >= 30:
            resid[m] = dy[m] - dy[m].mean()
    print(f'  用序列均值校正后剩余 std = {resid.std():.2f} px '
          f'（原 {dy.std():.2f}，降低 {(1-resid.std()/dy.std())*100:.1f}%）')

    print(f'\n  {"seq":26s} {"n":>5s} {"mean_dy":>8s} {"std_dy":>7s} {"med_h":>6s} {"mean_cy":>8s} {"mean_cx":>8s}')
    rows.sort(key=lambda r: -r['mean_dy'])
    for r in rows:
        print(f'  {r["seq"][:26]:26s} {r["n"]:5d} {r["mean_dy"]:8.2f} {r["std_dy"]:7.2f} '
              f'{r["med_h"]:6.1f} {r["mean_cy"]:8.3f} {r["mean_cx"]:8.3f}')

    print('\n  序列 mean(dy) 与序列级因素的相关系数:')
    print(f'    vs 序列中位目标高度  {corr([r["med_h"] for r in rows], md):+.4f}')
    print(f'    vs 序列平均 cy       {corr([r["mean_cy"] for r in rows], md):+.4f}')
    print(f'    vs 序列平均 cx       {corr([r["mean_cx"] for r in rows], md):+.4f}')

    # 时间戳（序列名首段 20190925_111757 → 拍摄批次）是否解释序列间差异
    day = np.array([r['seq'].split('_')[0] for r in rows])
    uniq_day = sorted(set(day.tolist()))
    if 1 < len(uniq_day) <= 6:
        print(f'\n  按拍摄日分组（{len(uniq_day)} 天）:')
        for d in uniq_day:
            m = day == d
            print(f'    {d}: 序列 {m.sum()} 个, mean(dy) 均值 {md[m].mean():.2f} px, '
                  f'std {md[m].std(ddof=1) if m.sum()>1 else 0:.2f}')

    # 逐帧「同序列内 cy 效应」是否跨序列一致
    print('\n  序列内 corr(dy, cy)（检验 cy 效应是否跨序列稳定）:')
    cs = []
    for s in uniq:
        m = seq == s
        if m.sum() >= 100:
            c = corr(cy[m], dy[m])
            if not np.isnan(c):
                cs.append((s, c))
    if cs:
        vals = np.array([c for _, c in cs])
        print(f'    共 {len(cs)} 个序列, 均值 {vals.mean():+.4f}, '
              f'负相关占比 {np.mean(vals < 0)*100:.1f}%, 范围 [{vals.min():+.3f}, {vals.max():+.3f}]')


def run_scale_analysis(rec_dx, rec_dy, rec_cx, rec_cy, rec_h, seq_of):
    """按目标高度分档，检验偏移是否为「绝对像素量」而非「相对目标尺寸的比例」。

    判据：
      - 绝对像素假说 → 各档 dy 均值接近，dy/h 随 h 单调下降（∝1/h）
      - 相对尺寸假说 → 各档 dy/h 接近，dy 随 h 单调上升（∝h）
      - 混合          → 两者都不恒定，需报告 dy = a + b·h 的拟合
    """
    dx = np.array(rec_dx); dy = np.array(rec_dy)
    cx = np.array(rec_cx); cy = np.array(rec_cy); h = np.array(rec_h)
    seq = np.array(seq_of)

    print('\n--- Q7 目标尺度分档（按 RGB 框高 h 的 33/66 分位切三档）---')
    q1, q2 = np.percentile(h, [33.33, 66.67])
    edges = [0.0, q1, q2, np.inf]
    names = [f'S (h<{q1:.1f})', f'M ({q1:.1f}~{q2:.1f})', f'L (h>{q2:.1f})']
    print(f'  所有目标高度: med={np.median(h):.1f} p5={np.percentile(h,5):.1f} '
          f'p95={np.percentile(h,95):.1f}')
    print(f'  {"档":20s} {"n":>6s} {"med_h":>8s} {"mean_dy":>9s} {"med_dy":>8s} '
          f'{"std_dy":>8s} {"med_dy/h":>9s} {"mean_dx":>9s}')
    for k in range(3):
        m = (h >= edges[k]) & (h < edges[k + 1])
        if m.sum() < 10:
            print(f'  {names[k]:20s} 样本过少')
            continue
        print(f'  {names[k]:20s} {m.sum():6d} {np.median(h[m]):8.1f} {dy[m].mean():9.2f} '
              f'{np.median(dy[m]):8.2f} {dy[m].std():8.2f} '
              f'{np.median(dy[m] / h[m]):9.2f} {dx[m].mean():9.2f}')

    # 线性拟合 dy = a + b·h（最小二乘），看 b 是否显著非零
    A = np.stack([np.ones_like(h), h], 1)
    coef, *_ = np.linalg.lstsq(A, dy, rcond=None)
    pred = A @ coef
    ss_res = ((dy - pred) ** 2).sum()
    ss_tot = ((dy - dy.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f'\n  线性拟合 dy = {coef[0]:.2f} + {coef[1]:.4f}·h   R²={r2:.4f}')
    print(f'  → 若 b·(p95(h)−p5(h)) 远小于 dy 的 std，说明尺度贡献可忽略（绝对像素假说成立）')
    span = np.percentile(h, 95) - np.percentile(h, 5)
    print(f'     b·h跨度 = {coef[1]*span:.2f} px，dy 的 std = {dy.std():.2f} px，'
          f'占比 {abs(coef[1]*span)/max(dy.std(),1e-9)*100:.1f}%')

    # 逐档内「是否仍单向」以及「档间均值差是否显著」（Welch t 近似）
    ms = [(h >= edges[k]) & (h < edges[k + 1]) for k in range(3)]
    if all(m.sum() > 30 for m in ms):
        print(f'\n  档间 dy 均值差: S→M {dy[ms[1]].mean()-dy[ms[0]].mean():+.2f} px, '
              f'M→L {dy[ms[2]].mean()-dy[ms[1]].mean():+.2f} px')
        for k in range(3):
            print(f'  档 {names[k]} 内 dy>0 占比 {np.mean(dy[ms[k]] > 0)*100:.2f}%')

    # 分档后再看与画幅纵向位置的相关性（检验 cy 效应是否跨档稳定）
    print('\n  各档内 corr(dy, cy)：')
    for k in range(3):
        m = ms[k]
        if m.sum() > 30:
            print(f'    {names[k]:20s} {corr(cy[m], dy[m]):+.4f}')

    # 以序列为单位检查分档效应是否被序列混淆
    print('\n  序列级核对（同序列内部是否有尺度依赖，排除跨序列混淆）：')
    uniq = [s for s in np.unique(seq) if (seq == s).sum() >= 100]
    ratios = []
    for s in uniq[:12]:
        m = seq == s
        hs, ds = h[m], dy[m]
        if np.percentile(hs, 90) - np.percentile(hs, 10) < 2:
            continue
        mm = hs > np.median(hs)
        r = ds[mm].mean() - ds[~mm].mean()
        ratios.append(r)
    if ratios:
        ratios = np.array(ratios)
        print(f'    各序列内「大目标档 − 小目标档」的 dy 差: 均值 {ratios.mean():+.2f} px, '
              f'std {ratios.std():.2f}, 共 {len(ratios)} 个序列')
        print(f'    一致方向占比 {np.mean(np.sign(ratios) == np.sign(ratios.mean()))*100:.1f}%')


def run_spot_analysis(seq_dxy, split, spot_n):
    """热斑定位：证明「IR 框指向热源、RGB 框指向机体」而非配对误差。

    对抽样 both 帧，在 IR 图上找每个 IR 框内的最亮点（热斑质心），
    拆成「IR 框中心 → 热斑」与「RGB 框中心 → 热斑」两段残差。
    若 IR 残差显著小于 RGB 残差，则两套标注确实指向不同物理区域。
    """
    import cv2

    meta = json.loads((ROOT / 'metadata' / f'{split}.json').read_text())['frames']
    rgb_dir, ir_dir = ROOT / 'labels' / split, ROOT / 'labels_ir' / split
    img_ir = ROOT / 'images_ir' / split
    stems = [s for s in sorted(p.stem for p in rgb_dir.glob('*.txt'))
             if meta.get(s, {}).get('kind') == 'both']

    rng = np.random.default_rng(0)
    if len(stems) > spot_n:
        stems = [stems[i] for i in rng.choice(len(stems), spot_n, replace=False)]

    res_ir, res_rgb, dy_ir_spot, dy_rgb_spot = [], [], [], []
    n_used = 0
    for stem in stems:
        r = read_txt(rgb_dir / f'{stem}.txt')
        i = read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None or len(r) == 0 or len(r) != len(i):
            continue
        img = cv2.imread(str(img_ir / f'{stem}.jpg'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        n_used += 1
        for ri, ii in hungarian_pair(r, i):
            rb, ib = r[ri], i[ii]
            x1 = max(0, int((ib[0] - ib[2] / 2) * W)); x2 = min(img.shape[1], int(np.ceil((ib[0] + ib[2] / 2) * W)))
            y1 = max(0, int((ib[1] - ib[3] / 2) * H)); y2 = min(img.shape[0], int(np.ceil((ib[1] + ib[3] / 2) * H)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            patch = cv2.GaussianBlur(img[y1:y2, x1:x2], (5, 5), 0)
            py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
            sx, sy = x1 + px, y1 + py  # 热斑坐标（IR 画幅像素）

            icx, icy = ib[0] * W, ib[1] * H
            rcx, rcy = rb[0] * W, rb[1] * H
            res_ir.append(np.hypot(sx - icx, sy - icy))
            res_rgb.append(np.hypot(sx - rcx, sy - rcy))
            dy_ir_spot.append(sy - icy)
            dy_rgb_spot.append(sy - rcy)

    print('\n--- Q6 热斑定位（IR 图上最亮点 vs 两个标注框中心）---')
    print(f'  使用帧 {n_used}，目标 {len(res_ir)}')
    stats('|热斑 − IR框中心|', res_ir)
    stats('|热斑 − RGB框中心|', res_rgb)
    stats('热斑y − IR框中心y', dy_ir_spot)
    stats('热斑y − RGB框中心y', dy_rgb_spot)
    a, b = np.array(res_ir), np.array(res_rgb)
    if a.size and b.size:
        print(f'  IR 框更靠近热斑的比例: {np.mean(a < b)*100:.2f}%')
        print(f'  RGB 框更靠近热斑的比例: {np.mean(b < a)*100:.2f}%')
        print(f'  中位残差比 (RGB/IR): {np.median(b)/max(np.median(a), 1e-9):.2f}×')


if __name__ == '__main__':
    main()
