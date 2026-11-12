# -*- coding: utf-8 -*-
"""双光谱基准数据集的独立 DataLoader 与评测协议（不依赖 ultralytics）。

设计要点：
  1. 从 manifest.jsonl 读取样本，支持按 source / split / kind / scene 过滤
  2. 逐模态独立标注：rgb_boxes 与 ir_boxes 各自独立，不合并
  3. letterbox 到统一画幅，同时返回逆变换参数（便于把预测框映射回原图）
  4. 三种评测分桶：both / only_rgb / only_ir（基于标注是否为空，而非 metadata）
  5. 提供可视化与一致性自检

用法：
  python dualspec_dataset.py --stats
  python dualspec_dataset.py --verify --n 20
  python dualspec_dataset.py --visualize --out /tmp/viz
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

BENCH = Path('/3250410011/root/rivermind-data/dualspec_bench')
DEFAULT_SIZE = (640, 512)   # (W, H)


# =========================================================
#  基础工具
# =========================================================
def load_manifest(path=None):
    p = Path(path) if path else BENCH / 'manifest.jsonl'
    recs = []
    with p.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def letterbox(img, new_size=DEFAULT_SIZE, color=(114, 114, 114)):
    """返回 (img_lb, scale, pad_x, pad_y)。"""
    h0, w0 = img.shape[:2]
    nw, nh = new_size
    scale = min(nw / w0, nh / h0)
    rw, rh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    top = (nh - rh) // 2
    bottom = nh - rh - top
    left = (nw - rw) // 2
    right = nw - rw - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, scale, left, top


def boxes_to_lb(boxes, scale, pad_x, pad_y):
    """归一化 cxcywh -> letterbox 后的归一化 cxcywh。"""
    if not boxes:
        return []
    out = []
    for b in boxes:
        cls, cx, cy, w, h = b[0], b[1], b[2], b[3], b[4]
        px = cx * (scale * 0 + 1)  # 占位，下面直接换算式
        out.append([cls, cx, cy, w, h])
    return out


def norm_to_px(boxes, w, h):
    """归一化 cxcywh -> 像素 x1y1x2y2。"""
    out = []
    for b in boxes:
        cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
        out.append([b[0], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return out


def px_to_norm(boxes, w, h):
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b[1], b[2], b[3], b[4]
        out.append([b[0], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                    (x2 - x1) / w, (y2 - y1) / h])
    return out


def lb_boxes(boxes, src_w, src_h, scale, pad_x, pad_y, dst_w, dst_h):
    """把源图归一化框映射到 letterbox 后的归一化框。"""
    if not boxes:
        return []
    px = norm_to_px(boxes, src_w, src_h)
    out = []
    for b in px:
        x1 = b[1] * scale + pad_x
        y1 = b[2] * scale + pad_y
        x2 = b[3] * scale + pad_x
        y2 = b[4] * scale + pad_y
        out.append([b[0], (x1 + x2) / 2 / dst_w, (y1 + y2) / 2 / dst_h,
                    (x2 - x1) / dst_w, (y2 - y1) / dst_h])
    return out


def unletterbox_box(box, scale, pad_x, pad_y):
    """letterbox 后的归一化框 -> 源图归一化框（逆变换）。"""
    cx = (box[1] * DEFAULT_SIZE[0] - pad_x) / scale
    cy = (box[2] * DEFAULT_SIZE[1] - pad_y) / scale
    w = box[3] * DEFAULT_SIZE[0] / scale
    h = box[4] * DEFAULT_SIZE[1] / scale
    return [box[0], cx, cy, w, h]


def visibility_bucket(rec):
    """按标注是否为空判定可见性桶（不依赖 metadata 的 kind 字段）。

    both      : 两模态都有框
    only_rgb  : 仅 rgb 有框
    only_ir   : 仅 ir 有框
    empty     : 两模态都无框（纯背景）
    """
    nr = len(rec.get('rgb_boxes') or [])
    ni = len(rec.get('ir_boxes') or [])
    if nr > 0 and ni > 0:
        return 'both'
    if nr > 0:
        return 'only_rgb'
    if ni > 0:
        return 'only_ir'
    return 'empty'


# =========================================================
#  Dataset
# =========================================================
class DualSpecDataset:
    """最小可用数据集：返回 letterbox 后的双模态图像 + 两套独立标注。"""

    def __init__(self, source=None, split=None, size=DEFAULT_SIZE,
                 with_images=True, max_items=None, seed=0):
        self.size = size
        self.with_images = with_images
        recs = load_manifest()
        if source:
            srcs = source if isinstance(source, (list, tuple)) else [source]
            recs = [r for r in recs if r['source'] in srcs]
        if split:
            sps = split if isinstance(split, (list, tuple)) else [split]
            recs = [r for r in recs if r['split'] in sps]
        # 负样本池不参与双光谱样本
        recs = [r for r in recs if r['source'] != 'drone_bird']
        # 必须有 IR 图
        recs = [r for r in recs if r.get('ir_path')]
        if max_items and len(recs) > max_items:
            random.Random(seed).shuffle(recs)
            recs = recs[:max_items]
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        item = dict(rec=r, bucket=visibility_bucket(r))
        if not self.with_images or cv2 is None:
            return item

        img_r = cv2.imread(r['rgb_path'], cv2.IMREAD_COLOR)
        img_i = cv2.imread(r['ir_path'], cv2.IMREAD_COLOR)
        if img_r is None or img_i is None:
            item['error'] = 'read_failed'
            return item

        rw, rh = r['rgb_size']
        iw, ih = r['ir_size']
        lb_r, sc_r, px_r, py_r = letterbox(img_r, self.size)
        lb_i, sc_i, px_i, py_i = letterbox(img_i, self.size)

        item.update({
            'img_rgb': lb_r,
            'img_ir': lb_i,
            'boxes_rgb': lb_boxes(r.get('rgb_boxes') or [], rw, rh, sc_r, px_r, py_r,
                                  self.size[0], self.size[1]),
            'boxes_ir': lb_boxes(r.get('ir_boxes') or [], iw, ih, sc_i, px_i, py_i,
                                 self.size[0], self.size[1]),
            'meta': {
                'rgb_size': (rw, rh), 'ir_size': (iw, ih),
                'scale_rgb': sc_r, 'pad_rgb': (px_r, py_r),
                'scale_ir': sc_i, 'pad_ir': (px_i, py_i),
            },
        })
        return item


# =========================================================
#  自检与统计
# =========================================================
def cmd_stats(dataset_filter=None):
    recs = load_manifest()
    by_src = Counter(r['source'] for r in recs)
    print('=== 数据集总览 ===')
    for s, c in sorted(by_src.items()):
        print(f'  {s:16s} {c:7d}')

    print('\n=== 可见性分桶（按标注是否为空判定）===')
    print(f'  {"source":16s} {"both":>8s} {"only_rgb":>9s} {"only_ir":>8s} {"empty":>7s} {"合计":>8s}')
    for s in sorted(by_src):
        if s == 'drone_bird':
            continue
        sub = [r for r in recs if r['source'] == s]
        c = Counter(visibility_bucket(r) for r in sub)
        print(f'  {s:16s} {c.get("both",0):8d} {c.get("only_rgb",0):9d} '
              f'{c.get("only_ir",0):8d} {c.get("empty",0):7d} {len(sub):8d}')

    print('\n=== metadata kind 与标注桶的一致性 ===')
    for s in ('anti_uav300',):
        sub = [r for r in recs if r['source'] == s]
        cross = Counter((r.get('kind', '?'), visibility_bucket(r)) for r in sub)
        for (k, b), n in sorted(cross.items(), key=lambda x: -x[1])[:8]:
            print(f'  {s}: kind={k:10s} -> bucket={b:9s}  n={n}')

    print('\n=== 各数据源场景数 ===')
    for s in sorted(by_src):
        if s == 'drone_bird':
            continue
        scenes = {r['scene'] for r in recs if r['source'] == s}
        print(f'  {s:16s} 场景/序列 {len(scenes):5d}')


def cmd_verify(n=20, seed=0):
    """抽样验证：读取图像、letterbox、映射框，检查是否落在图内。"""
    if cv2 is None:
        print('cv2 不可用，跳过')
        return
    print(f'=== 抽样验证 n={n} ===\n')
    ok = bad = 0
    problems = []
    for src in ('anti_uav300', 'uav_cb', 'mendeley'):
        ds = DualSpecDataset(source=src, max_items=n, seed=seed)
        if not len(ds):
            print(f'[{src}] 空数据集')
            continue
        print(f'[{src}] {len(ds)} 样本')
        stat = Counter()
        for i in range(len(ds)):
            it = ds[i]
            if 'error' in it:
                bad += 1
                problems.append((src, it['rec']['id'], it['error']))
                continue
            ok += 1
            for key in ('boxes_rgb', 'boxes_ir'):
                for b in it[key]:
                    x1 = b[1] - b[3] / 2
                    y1 = b[2] - b[4] / 2
                    x2 = b[1] + b[3] / 2
                    y2 = b[2] + b[4] / 2
                    inb = (-0.02 <= x1 and -0.02 <= y1 and x2 <= 1.02 and y2 <= 1.02)
                    stat[f'{key}_in'] += int(inb)
                    stat[f'{key}_out'] += int(not inb)
                    if not inb:
                        problems.append((src, it['rec']['id'], key, [round(v, 3) for v in b]))
            stat[f'bucket_{it["bucket"]}'] += 1
            stat[f'img_{it["img_rgb"].shape[:2]}'] += 1
        for k in sorted(stat):
            print(f'    {k}: {stat[k]}')
        print()

    print(f'总读取成功 {ok}，失败 {bad}')
    if problems:
        print('问题样例（最多 10 条）：')
        for p in problems[:10]:
            print('  ', p)


def cmd_visualize(out_dir, n=6, seed=0):
    """把 letterbox 后的双模态图像与两套框画出来，人工确认。"""
    if cv2 is None:
        print('cv2 不可用')
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for src in ('anti_uav300', 'uav_cb', 'mendeley'):
        ds = DualSpecDataset(source=src, max_items=n, seed=seed)
        for i in range(len(ds)):
            it = ds[i]
            if 'error' in it:
                continue
            a = it['img_rgb'].copy()
            b = it['img_ir'].copy()
            for box, img, col in ((it['boxes_rgb'], a, (0, 200, 0)),
                                  (it['boxes_ir'], b, (0, 160, 255))):
                for bb in box:
                    x1 = int((bb[1] - bb[3] / 2) * DEFAULT_SIZE[0])
                    y1 = int((bb[2] - bb[4] / 2) * DEFAULT_SIZE[1])
                    x2 = int((bb[1] + bb[3] / 2) * DEFAULT_SIZE[0])
                    y2 = int((bb[2] + bb[4] / 2) * DEFAULT_SIZE[1])
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(a, f'{src} RGB  {it["bucket"]}', (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(b, f'{src} IR  {it["bucket"]}', (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            canvas = np.hstack([a, np.full((a.shape[0], 8, 3), 60, np.uint8), b])
            fn = out / f'{src}_{i:02d}.jpg'
            cv2.imwrite(str(fn), canvas)
        print(f'[{src}] 已输出 {min(n, len(ds))} 张到 {out}')


def cmd_consistency():
    """检查 anti_uav300 的 metadata kind 与实际标注是否一致。"""
    recs = [r for r in load_manifest() if r['source'] == 'anti_uav300']
    print('=== anti_uav300: metadata kind vs 标注实际状态 ===')
    cross = Counter()
    for r in recs:
        k = r.get('kind', '?')
        b = visibility_bucket(r)
        cross[(k, b)] += 1
    print(f'  {"kind":12s} {"标注桶":10s} {"数量":>8s}')
    for (k, b), n in sorted(cross.items(), key=lambda x: -x[1]):
        flag = '' if (k == b or (k == 'both' and b == 'both')) else '  ← 不一致'
        print(f'  {k:12s} {b:10s} {n:8d}{flag}')

    # 图像缺失检查
    miss = [r['id'] for r in recs if not Path(r['rgb_path']).exists()
            or not Path(r['ir_path']).exists()]
    print(f'\n  图像缺失: {len(miss)} 条')
    if miss[:5]:
        print('  样例:', miss[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--visualize', action='store_true')
    ap.add_argument('--consistency', action='store_true')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default='/tmp/dualspec_viz')
    args = ap.parse_args()

    if args.stats:
        cmd_stats()
    if args.consistency:
        cmd_consistency()
    if args.verify:
        cmd_verify(args.n)
    if args.visualize:
        cmd_visualize(args.out, min(args.n, 8))


if __name__ == '__main__':
    main()
