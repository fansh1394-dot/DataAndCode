# -*- coding: utf-8 -*-
"""诊断：热斑归属 o 的实际分布，以及 o 与尺寸比 s 的关系。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import cv2

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


def hot_spot_in(img, box):
    x1 = max(0, int((box[0] - box[2] / 2) * W))
    x2 = min(img.shape[1], int(np.ceil((box[0] + box[2] / 2) * W)))
    y1 = max(0, int((box[1] - box[3] / 2) * H))
    y2 = min(img.shape[0], int(np.ceil((box[1] + box[3] / 2) * H)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = cv2.GaussianBlur(img[y1:y2, x1:x2], (5, 5), 0)
    py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return x1 + px, y1 + py


def inside(pt, box):
    cx, cy = box[0] * W, box[1] * H
    hw, hh = box[2] * W / 2, box[3] * H / 2
    return (cx - hw) <= pt[0] <= (cx + hw) and (cy - hh) <= pt[1] <= (cy + hh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--sample', type=int, default=2000)
    args = ap.parse_args()

    meta = json.loads((ROOT / 'metadata' / f'{args.split}.json').read_text())['frames']
    rgb_dir = ROOT / 'labels' / args.split
    ir_dir = ROOT / 'labels_ir' / args.split
    img_ir_dir = ROOT / 'images_ir' / args.split

    stems = [p.stem for p in rgb_dir.glob('*.txt')]
    stems = [s for s in stems if meta.get(s, {}).get('kind') == 'both']
    stems.sort()
    rng = np.random.default_rng(0)
    if args.sample and len(stems) > args.sample:
        stems = [stems[i] for i in rng.choice(len(stems), args.sample, replace=False)]

    o_cnt = Counter()
    s_by_o = defaultdict(list)
    # s 的分位数
    s_all = []

    for stem in stems:
        r, i = read_txt(rgb_dir / f'{stem}.txt'), read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None or len(r) == 0 or len(r) != len(i):
            continue
        img = cv2.imread(str(img_ir_dir / f'{stem}.jpg'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for ri, ii in pair_one(r, i):
            rb, ib = r[ri], i[ii]
            sp = hot_spot_in(img, ib)
            if sp is None:
                o_cnt['no_spot'] += 1
                continue
            s = (ib[3] * H) / max(rb[3] * H, 1e-6)
            s_all.append(s)
            in_ir, in_rgb = inside(sp, ib), inside(sp, rb)
            if in_ir and in_rgb:
                o = 'both'
            elif in_ir:
                o = 'IR_only'
            elif in_rgb:
                o = 'RGB_only'
            else:
                # 热斑在两个框之外：说明热斑定位失败或目标极小
                o = 'neither'
            o_cnt[o] += 1
            s_by_o[o].append(s)

    total = sum(o_cnt.values())
    print(f'=== split={args.split} 样本帧={len(stems)} 目标={total} ===')
    print('\n热斑归属 o 的分布：')
    for k, n in o_cnt.most_common():
        print(f'  {k:10s} {n:6d}  {n/total*100:5.1f}%')

    print('\n各 o 类别下的尺寸比 s（h_ir / h_rgb）：')
    for k, v in s_by_o.items():
        a = np.array(v)
        print(f'  {k:10s} n={len(a):5d}  mean={a.mean():.3f}  med={np.median(a):.3f}  '
              f'p10={np.percentile(a,10):.3f}  p90={np.percentile(a,90):.3f}')

    a = np.array(s_all)
    print(f'\ns 全分布：mean={a.mean():.3f} med={np.median(a):.3f} '
          f'p5={np.percentile(a,5):.3f} p25={np.percentile(a,25):.3f} '
          f'p75={np.percentile(a,75):.3f} p95={np.percentile(a,95):.3f}')
    print(f'  s<0.9 占比 {np.mean(a<0.9)*100:.1f}%   s>1.1 占比 {np.mean(a>1.1)*100:.1f}%')


if __name__ == '__main__':
    main()
