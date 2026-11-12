# -*- coding: utf-8 -*-
"""预实验 Step 1：把 both 帧细分为 semantic / ambiguous / geometric 三桶。

判定依据（全部用已有标注+IR图像即可算，不需要训练）：
  o = 热斑质心落在哪个框内        （IR 框内 = 语义级强证据）
  s = h_ir / h_rgb               （热源框通常系统性小于机体框）
  r = ‖Δc − A(seq) − B·cy‖       （几何级则残差小）

分桶：
  semantic  : o=IR  且 s < 0.90
  ambiguous : o=IR  且 0.90 ≤ s ≤ 1.10
  geometric : o=RGB 或 s > 1.10

用法：python scripts/_pre_semantic_buckets.py [--split test] [--sample N]
"""
import argparse
import json
from collections import defaultdict
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
    """both 帧框数一致，按中心距离+尺寸差做一对一匹配。"""
    from scipy.optimize import linear_sum_assignment
    ca, cb = r[:, :2], i[:, :2]
    cost = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    sa, sb = r[:, 2:] * r[:, 2:], i[:, 2:] * i[:, 2:]
    cost = cost + 0.5 * ((np.sqrt(sa)[:, None, :] - np.sqrt(sb)[None, :, :]) ** 2).sum(-1)
    rr, cc = linear_sum_assignment(cost)
    return list(zip(rr.tolist(), cc.tolist()))


def hot_spot_in(ir_img, box):
    """返回 (sx, sy) 热斑像素坐标，或 None。"""
    x1 = max(0, int((box[0] - box[2] / 2) * W))
    x2 = min(ir_img.shape[1], int(np.ceil((box[0] + box[2] / 2) * W)))
    y1 = max(0, int((box[1] - box[3] / 2) * H))
    y2 = min(ir_img.shape[0], int(np.ceil((box[1] + box[3] / 2) * H)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = cv2.GaussianBlur(ir_img[y1:y2, x1:x2], (5, 5), 0)
    py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return x1 + px, y1 + py


def inside(pt, box):
    cx, cy = box[0] * W, box[1] * H
    hw, hh = box[2] * W / 2, box[3] * H / 2
    return (cx - hw) <= pt[0] <= (cx + hw) and (cy - hh) <= pt[1] <= (cy + hh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--sample', type=int, default=0, help='0=全量')
    args = ap.parse_args()

    meta = json.loads((ROOT / 'metadata' / f'{args.split}.json').read_text())['frames']
    rgb_dir = ROOT / 'labels' / args.split
    ir_dir = ROOT / 'labels_ir' / args.split
    img_ir_dir = ROOT / 'images_ir' / args.split

    stems = [p.stem for p in rgb_dir.glob('*.txt')]
    stems = [s for s in stems if meta.get(s, {}).get('kind') == 'both']
    stems.sort()
    if args.sample and len(stems) > args.sample:
        rng = np.random.default_rng(0)
        stems = [stems[i] for i in rng.choice(len(stems), args.sample, replace=False)]

    # 先用全量样本估计 A(seq) 与 B
    seq_pts = defaultdict(list)
    records = []
    for stem in stems:
        r, i = read_txt(rgb_dir / f'{stem}.txt'), read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None or len(r) == 0 or len(r) != len(i):
            continue
        seq = meta[stem]['sequence']
        for ri, ii in pair_one(r, i):
            rb, ib = r[ri], i[ii]
            dx = (ib[0] - rb[0]) * W
            dy = (ib[1] - rb[1]) * H
            seq_pts[seq].append((rb[1], dy))
            records.append((stem, seq, rb, ib, dx, dy))

    # 序列内中心化估 B
    rc, rd = [], []
    for seq, pts in seq_pts.items():
        a = np.array(pts)
        if len(a) > 20:
            rc.extend((a[:, 0] - a[:, 0].mean()).tolist())
            rd.extend((a[:, 1] - a[:, 1].mean()).tolist())
    B = float(np.cov(rc, rd)[0, 1] / np.var(rc)) if len(rc) > 10 and np.var(rc) > 0 else 0.0

    # 每序列 A
    A_of = {}
    for seq, pts in seq_pts.items():
        a = np.array(pts)
        A_of[seq] = float(np.mean(a[:, 1] - B * (a[:, 0] - 0.5)))

    buckets = defaultdict(int)
    bucket_rows = []
    spot_stats = []

    for stem, seq, rb, ib, dx, dy in records:
        img = cv2.imread(str(img_ir_dir / f'{stem}.jpg'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        sp = hot_spot_in(img, ib)
        if sp is None:
            buckets['no_spot'] += 1
            continue
        s = (ib[3] * H) / max(rb[3] * H, 1e-6)
        r_res = abs(dy - (A_of[seq] + B * (rb[1] - 0.5)))

        in_ir = inside(sp, ib)
        in_rgb = inside(sp, rb)
        if in_ir and not in_rgb:
            o = 'IR_only'
        elif in_rgb and not in_ir:
            o = 'RGB_only'
        elif in_ir and in_rgb:
            o = 'both'
        else:
            o = 'neither'

        if o == 'IR_only' and s < 0.90:
            b = 'semantic'
        elif o == 'IR_only' and s <= 1.10:
            b = 'ambiguous'
        elif o in ('RGB_only', 'both', 'neither') or s > 1.10:
            b = 'geometric'
        else:
            b = 'ambiguous'
        buckets[b] += 1
        bucket_rows.append((stem, b, s, r_res, o))
        spot_stats.append((s, r_res, o == 'IR_only'))

    total = sum(buckets.values())
    print(f'=== split={args.split}  样本帧={len(stems)}  可用目标={total} ===')
    print(f'B = {B:.2f}  |  序列数 {len(A_of)}')
    print()
    print(f'{"桶":12s} {"目标数":>8s} {"占比":>8s}')
    for k in ('semantic', 'ambiguous', 'geometric', 'no_spot'):
        n = buckets.get(k, 0)
        pct = n / total * 100 if total else 0
        print(f'{k:12s} {n:8d} {pct:7.1f}%')

    print()
    print('--- 桶内特征均值 ---')
    for k in ('semantic', 'ambiguous', 'geometric'):
        rows = [r for r in bucket_rows if r[1] == k]
        if not rows:
            continue
        s_arr = np.array([r[2] for r in rows])
        r_arr = np.array([r[3] for r in rows])
        print(f'{k:12s} n={len(rows):5d}  s均值={s_arr.mean():.3f}  r均值={r_arr.mean():.1f}px  '
              f's中位={np.median(s_arr):.3f}  r中位={np.median(r_arr):.1f}px')

    out = Path('/3250410011/root/rivermind-data/MutilModel_3485988898/scripts/_pre_buckets.json')
    out.write_text(json.dumps({
        'split': args.split, 'B': B, 'buckets': dict(buckets),
        'stems_by_bucket': {
            k: sorted({r[0] for r in bucket_rows if r[1] == k}) for k in ('semantic', 'ambiguous', 'geometric')
        },
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n[SAVED] {out}')


if __name__ == '__main__':
    main()
