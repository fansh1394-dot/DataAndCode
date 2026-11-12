# -*- coding: utf-8 -*-
"""核实 uav_data_v2 是否真为跨模态双光谱：
  1) images 与 images_ir 是否来自不同视频（像素统计 + 相关性）
  2) labels 与 labels_ir 是否分别来自 visible.json / infrared.json
  3) 两路视频帧数是否一致（only_ir / only_rgb 的来源）
"""
import json
import subprocess
from pathlib import Path

import numpy as np

RAW = Path('/3250410011/root/rivermind-data/Anti-UAV300')
DATA = Path('/root/rivermind-data/uav_data_v2')


def json_boxes(p):
    """Anti-UAV json 可能为 {frame_id: [x1,y1,x2,y2]} 或带 exist 标记。"""
    d = json.loads(p.read_text())
    return d


def main():
    import cv2

    # 选一个 test 序列
    seq = '20190925_111757_1_10'
    sdir = None
    for sp in ('test', 'train', 'val'):
        c = RAW / sp / seq
        if c.exists():
            sdir = c
            break
    print(f'=== 核实序列: {sdir} ===\n')

    # ---------- 1. 视频帧数 ----------
    print('--- 1. 两路视频信息 ---')
    for name in ('visible.mp4', 'infrared.mp4'):
        p = sdir / name
        if not p.exists():
            print(f'  {name}: 不存在')
            continue
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f'  {name}: {n} 帧  {w}x{h}  {fps:.1f} fps')

    # ---------- 2. 图像是否同源 ----------
    print('\n--- 2. images vs images_ir 是否同源 ---')
    stem = f'{seq}_00000'
    pv = DATA / 'images' / 'test' / f'{stem}.jpg'
    pi = DATA / 'images_ir' / 'test' / f'{stem}.jpg'
    for tag, p in (('images   ', pv), ('images_ir', pi)):
        if not p.exists():
            print(f'  {tag}: 缺失 {p}')
            continue
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        print(f'  {tag}: shape={im.shape} mean={im.mean():.2f} std={im.std():.2f} '
              f'min={im.min()} max={im.max()}')
    if pv.exists() and pi.exists():
        a = cv2.imread(str(pv), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        b = cv2.imread(str(pi), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if a.shape == b.shape:
            c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            diff = np.abs(a - b).mean()
            print(f'  两图相关系数 = {c:.4f}   平均绝对差 = {diff:.2f}')
            print('  判读：相关系数接近 1 且差≈0 → 同源；明显 <1 → 不同模态')

    # ---------- 3. 标注来源 ----------
    print('\n--- 3. 标注文件结构 ---')
    for name in ('visible.json', 'infrared.json'):
        p = sdir / name
        if not p.exists():
            continue
        d = json_boxes(p)
        ks = list(d)[:3]
        print(f'  {name}: 顶层类型={type(d).__name__}, 条目数={len(d)}, 前3键={ks}')
        for k in ks:
            print(f'     {k} -> {json.dumps(d[k], ensure_ascii=False)[:160]}')

    # ---------- 4. 与 labels / labels_ir 对照 ----------
    print('\n--- 4. uav_data_v2 的标签 vs 原始 json ---')
    for tag, lp in (('labels   ', DATA / 'labels' / 'test' / f'{stem}.txt'),
                    ('labels_ir', DATA / 'labels_ir' / 'test' / f'{stem}.txt')):
        if lp.exists():
            print(f'  {tag}: {lp.read_text().strip()[:200]}')
        else:
            print(f'  {tag}: 缺失')

    # ---------- 5. 全量序列帧数一致性 ----------
    print('\n--- 5. 抽样 8 个序列：两路视频帧数对照 ---')
    cnt = 0
    for sp in ('test', 'train', 'val'):
        for sd in sorted((RAW / sp).iterdir()):
            if not sd.is_dir():
                continue
            v, i = sd / 'visible.mp4', sd / 'infrared.mp4'
            if not (v.exists() and i.exists()):
                continue
            nv = ni = -1
            for p, which in ((v, 'v'), (i, 'i')):
                cap = cv2.VideoCapture(str(p))
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if which == 'v':
                    nv = n
                else:
                    ni = n
            print(f'  {sp}/{sd.name}: visible={nv} infrared={ni} '
                  f'{"一致" if nv == ni else "不一致"}')
            cnt += 1
            if cnt >= 8:
                break
        if cnt >= 8:
            break


if __name__ == '__main__':
    main()
