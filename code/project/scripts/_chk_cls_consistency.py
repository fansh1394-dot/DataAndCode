# -*- coding: utf-8 -*-
"""核查 labels/ 与 labels_ir/ 两套标注的分类标签是否一致。

不做几何判断，只回答四个问题：
  Q1 两套标注各自出现的 class id 及计数
  Q2 同名文件里"框的数量"是否一致（类别数一致但框数不同=可见性差异）
  Q3 同名文件里逐框的 class id 多重集是否一致
  Q4 both 帧上，按几何配对后的类别是否一致（配对失败单列）

用法：python scripts/_chk_cls_consistency.py [--split test] [--limit N]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/root/rivermind-data/uav_data_v2')


def read_txt(p):
    """返回 [(cls, cx, cy, w, h)]，缺失/空文件返回 []。"""
    if not p.exists():
        return None  # 用 None 区分"文件不存在"与"空文件"
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((int(float(parts[0])), *[float(x) for x in parts[1:5]]))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=0, help='0 表示全量')
    args = ap.parse_args()

    meta_p = ROOT / 'metadata' / f'{args.split}.json'
    meta = json.loads(meta_p.read_text())['frames']
    kinds = Counter(v.get('kind') for v in meta.values())

    rgb_dir, ir_dir = ROOT / 'labels' / args.split, ROOT / 'labels_ir' / args.split
    stems = sorted(p.stem for p in rgb_dir.glob('*.txt'))
    if args.limit:
        stems = stems[:args.limit]

    cls_rgb, cls_ir = Counter(), Counter()
    stat = defaultdict(int)
    mismatch_examples, pair_iou_cls = [], []

    for stem in stems:
        r = read_txt(rgb_dir / f'{stem}.txt')
        i = read_txt(ir_dir / f'{stem}.txt')
        if r is None or i is None:
            stat['missing_one_side'] += 1
            continue
        for b in r:
            cls_rgb[b[0]] += 1
        for b in i:
            cls_ir[b[0]] += 1

        nr, ni = len(r), len(i)
        if nr == ni == 0:
            stat['both_empty'] += 1
            continue
        if nr == 0 or ni == 0:
            stat['one_side_empty'] += 1
            stat['one_side_empty_rgb' if nr == 0 else 'one_side_empty_ir'] += 1
            continue
        if nr != ni:
            stat['box_count_diff'] += 1
            continue

        stat['box_count_same'] += 1
        # 逐框类别多重集（类别数相同则必然一致，因当前仅单类；仍显式计算）
        if Counter(b[0] for b in r) == Counter(b[0] for b in i):
            stat['cls_multiset_same'] += 1
        else:
            stat['cls_multiset_diff'] += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(stem)

        # 几何配对后的类别一致性（贪心最近中心）
        used = set()
        paired = 0
        for rb in r:
            best, bj = -1.0, -1
            for j, ib in enumerate(i):
                if j in used:
                    continue
                v = iou(rb, ib)
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best > 0:
                used.add(bj)
                paired += 1
                pair_iou_cls.append((best, rb[0] == i[bj][0]))
        stat['paired_boxes'] += paired

    print(f'=== split={args.split}  frames={len(stems)} ===')
    print('frame kind 分布:', dict(kinds))
    print('RGB 标签 class 计数:', dict(sorted(cls_rgb.items())))
    print('IR  标签 class 计数:', dict(sorted(cls_ir.items())))
    print('\n--- 文件级统计 ---')
    for k in sorted(stat):
        print(f'  {k:26s} {stat[k]}')
    print('\n--- 配对框类别一致性 ---')
    if pair_iou_cls:
        same = sum(1 for _, s in pair_iou_cls if s)
        print(f'  配对框 {len(pair_iou_cls)}，类别相同 {same} ({same/len(pair_iou_cls)*100:.2f}%)')
        hi = [s for v, s in pair_iou_cls if v >= 0.5]
        if hi:
            print(f'  IoU>=0.5 的配对 {len(hi)}，类别相同 {sum(hi)} ({sum(hi)/len(hi)*100:.2f}%)')
    else:
        print('  无可配对框')
    if mismatch_examples:
        print('\n  类别多重集不一致样例:', mismatch_examples)


if __name__ == '__main__':
    main()
