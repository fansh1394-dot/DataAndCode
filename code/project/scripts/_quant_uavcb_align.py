# -*- coding: utf-8 -*-
"""量化 UAV-CB 的配准程度。

关键：两边分辨率与宽高比都不同（visible 1920x1080 16:9，thermal 640x512 5:4），
不能直接比较像素坐标。这里采用两条路线：

  Route A：归一化坐标比较（除以各自宽高），假设画面覆盖同一视场
  Route B：letterbox 到共同画幅 640x512 后再比较（可与其他数据集统一）

并输出：IoU 分布、中心位移（归一化/像素）、逐场景统计、配对成功率。
"""
import json
import math
import random
import statistics
import sys
import zipfile

Z = '/3250410011/root/rivermind-fs/UAV-CB.zip'
TARGET_W, TARGET_H = 640, 512
SCENES = ['building', 'cloud', 'ground', 'powerline', 'vegetation']


def boxes_of(js):
    out = []
    for s in js.get('shapes', []):
        pts = s.get('points', [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((min(xs), min(ys), max(xs), max(ys), s.get('label')))
    return out


def letterbox_params(w, h, tw, th):
    """返回 (scale, new_w, new_h, pad_x, pad_y)，与 Ultralytics 的 letterbox 一致。"""
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x = (tw - nw) / 2
    pad_y = (th - nh) / 2
    return scale, nw, nh, pad_x, pad_y


def to_lb(box, w, h):
    s, nw, nh, px, py = letterbox_params(w, h, TARGET_W, TARGET_H)
    x1, y1, x2, y2, lb = box
    return (x1 * s + px, y1 * s + py, x2 * s + px, y2 * s + py, lb)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def main():
    zf = zipfile.ZipFile(Z)
    names = set(zf.namelist())

    print(f'=== UAV-CB 配准程度量化  target={TARGET_W}x{TARGET_H} (letterbox) ===\n')
    print(f'{"场景":12s} {"配对数":>7s} {"单目标对":>8s} {"IoU均值":>8s} {"IoU中位":>8s} '
          f'{"IoU>0.5占比":>10s} {"位移中位px":>10s} {"归一化位移中位":>12s}')

    all_iou, all_dist, all_dist_n = [], [], []
    scene_stats = {}

    for sc in SCENES:
        vj = sorted(n for n in names if f'/{sc}/visible/' in n and n.endswith('.json'))
        pairs = 0
        single = 0
        ious, dists, dists_n = [], [], []

        for n in vj:
            tn = n.replace('/visible/', '/thermal/').replace('_visible_', '_thermal_')
            if tn not in names:
                continue
            try:
                dv = json.loads(zf.read(n))
                dt = json.loads(zf.read(tn))
            except Exception:
                continue
            pairs += 1
            bv_raw = boxes_of(dv)
            bt_raw = boxes_of(dt)
            if not bv_raw or not bt_raw:
                continue
            wv, hv = dv.get('imageWidth', 1920), dv.get('imageHeight', 1080)
            wt, ht = dt.get('imageWidth', 640), dt.get('imageHeight', 512)

            # Route A: 归一化
            def norm(b, w, h):
                return (b[0] / w, b[1] / h, b[2] / w, b[3] / h, b[4])
            bvn = [norm(b, wv, hv) for b in bv_raw]
            btn = [norm(b, wt, ht) for b in bt_raw]

            # Route B: letterbox
            bvl = [to_lb(b, wv, hv) for b in bv_raw]
            btl = [to_lb(b, wt, ht) for b in bt_raw]

            if len(bvn) == 1 and len(btn) == 1:
                single += 1
                # 归一化 IoU（宽高比不同，只作参考）
                ious.append(iou(bvn[0], btn[0]))
                cv, ct = center(bvn[0]), center(btn[0])
                dists_n.append(math.hypot(cv[0] - ct[0], cv[1] - ct[1]))
                # letterbox 像素位移
                cvl, ctl = center(bvl[0]), center(btl[0])
                dists.append(math.hypot(cvl[0] - ctl[0], cvl[1] - ctl[1]))
            else:
                # 多目标：按最近中心贪心配对取平均 IoU
                for b in bvn:
                    best = max((iou(b, t) for t in btn), default=0.0)
                    ious.append(best)

        all_iou += ious
        all_dist += dists
        all_dist_n += dists_n
        if pairs:
            scene_stats[sc] = dict(pairs=pairs, single=single, iou_med=statistics.median(ious) if ious else None)

        def _med(x):
            return statistics.median(x) if x else float('nan')
        print(f'{sc:12s} {pairs:7d} {single:8d} '
              f'{(statistics.mean(ious) if ious else float("nan")):8.3f} {_med(ious):8.3f} '
              f'{(sum(1 for x in ious if x>0.5)/len(ious)*100 if ious else float("nan")):9.1f}% '
              f'{_med(dists):10.1f} {_med(dists_n):12.4f}')

    print('\n--- 汇总 ---')
    if all_iou:
        print(f'  总配对数       : {sum(v["pairs"] for v in scene_stats.values())}')
        print(f'  IoU 均值/中位   : {statistics.mean(all_iou):.3f} / {statistics.median(all_iou):.3f}')
        print(f'  IoU > 0.5 占比  : {sum(1 for x in all_iou if x>0.5)/len(all_iou)*100:.1f}%')
        print(f'  IoU > 0.3 占比  : {sum(1 for x in all_iou if x>0.3)/len(all_iou)*100:.1f}%')
        print(f'  IoU < 0.1 占比  : {sum(1 for x in all_iou if x<0.1)/len(all_iou)*100:.1f}%')
    if all_dist:
        d = sorted(all_dist)
        print(f'  letterbox 像素位移: 中位 {statistics.median(d):.1f}px  '
              f'均值 {statistics.mean(d):.1f}px  p90 {d[int(len(d)*0.9)]:.1f}px  最大 {d[-1]:.1f}px')
    if all_dist_n:
        dn = sorted(all_dist_n)
        print(f'  归一化位移        : 中位 {statistics.median(dn):.4f}  '
              f'p90 {dn[int(len(dn)*0.9)]:.4f}  最大 {dn[-1]:.4f}')

    print('\n--- 判读 ---')
    if all_iou:
        m = statistics.median(all_iou)
        if m > 0.5:
            print('  → UAV-CB 基本已配准（中位 IoU > 0.5）')
        elif m > 0.2:
            print('  → UAV-CB 属弱配准，存在可观偏移')
        else:
            print('  → UAV-CB 明显未配准（中位 IoU < 0.2），可用于非对齐研究')


if __name__ == '__main__':
    main()
