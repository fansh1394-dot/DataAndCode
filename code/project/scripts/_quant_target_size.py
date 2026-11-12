# -*- coding: utf-8 -*-
"""统计三份双光谱数据在 640x512 letterbox 下的目标尺寸分布。

三份数据：
  A. uav_data_v2   640x512，YOLO 标注（labels/ 与 labels_ir/）
  B. UAV-CB        visible 1920x1080 / thermal 640x512，LabelMe JSON
  C. Mendeley      454x580，YOLO 标注（UAV_eo_Dataset / UAV_ir_Dataset）

输出：各数据源在原始分辨率与 640x512 letterbox 后的目标高度/面积分布，
用于判断统一画幅是否会把目标压到不可检测。
"""
import json
import statistics
import zipfile
from pathlib import Path

import numpy as np

TW, TH = 640, 512


def lb_scale(w, h):
    return min(TW / w, TH / h)


def report(name, hs_raw, hs_lb, note=''):
    if not hs_raw:
        print(f'{name:22s} 无数据')
        return
    a = np.array(hs_raw)
    b = np.array(hs_lb)
    q = lambda v: np.percentile(v, [5, 25, 50, 75, 95])
    qa, qb = q(a), q(b)
    print(f'{name:22s} n={len(a):6d}  {note}')
    print(f'  原始高度      : 中位 {np.median(a):6.1f}  均值 {a.mean():6.1f}  '
          f'p5={qa[0]:5.1f} p25={qa[1]:5.1f} p75={qa[3]:5.1f} p95={qa[4]:5.1f}')
    print(f'  letterbox 后  : 中位 {np.median(b):6.1f}  均值 {b.mean():6.1f}  '
          f'p5={qb[0]:5.1f} p25={qb[1]:5.1f} p75={qb[3]:5.1f} p95={qb[4]:5.1f}')
    for thr in (2, 4, 8, 16, 32):
        print(f'    高度 < {thr:2d}px : {np.mean(b < thr) * 100:5.1f}%')


def main():
    print(f'=== 三份数据在 {TW}x{TH} letterbox 下的目标尺寸 ===\n')

    # ---------- A. uav_data_v2 ----------
    root = Path('/root/rivermind-data/uav_data_v2')
    for tag, sub in (('A. uav_data_v2 RGB', 'labels'), ('A. uav_data_v2 IR', 'labels_ir')):
        hs_raw, hs_lb = [], []
        for split in ('train', 'val', 'test'):
            d = root / sub / split
            if not d.exists():
                continue
            for p in d.glob('*.txt'):
                for line in p.read_text().splitlines():
                    s = line.split()
                    if len(s) >= 5:
                        h = float(s[4]) * 512
                        hs_raw.append(h)
                        hs_lb.append(h)  # 原生即 640x512，尺度 1.0
        report(tag, hs_raw, hs_lb, '原生 640x512，无缩放')

    # ---------- B. UAV-CB ----------
    Z = '/3250410011/root/rivermind-fs/UAV-CB.zip'
    zf = zipfile.ZipFile(Z)
    names = zf.namelist()
    for tag, side in (('B. UAV-CB visible', 'visible'), ('B. UAV-CB thermal', 'thermal')):
        js = [n for n in names if f'/{side}/' in n and n.endswith('.json')]
        hs_raw, hs_lb = [], []
        for n in js:
            try:
                d = json.loads(zf.read(n))
            except Exception:
                continue
            w, h = d.get('imageWidth', 0), d.get('imageHeight', 0)
            if not w or not h:
                continue
            sc = lb_scale(w, h)
            for s in d.get('shapes', []):
                pts = s.get('points', [])
                if len(pts) < 2:
                    continue
                ys = [p[1] for p in pts]
                hh = max(ys) - min(ys)
                hs_raw.append(hh)
                hs_lb.append(hh * sc)
        w_note = '原生 1920x1080' if side == 'visible' else '原生 640x512'
        report(tag, hs_raw, hs_lb, w_note)

    # ---------- C. Mendeley ----------
    base = Path('/3250410011/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/'
                'Labelled_UAV_Images/Labelled EO_IR Dataset')
    for tag, sub in (('C. Mendeley EO', 'UAV_eo_Dataset'), ('C. Mendeley IR', 'UAV_ir_Dataset')):
        hs_raw, hs_lb = [], []
        d0 = base / sub
        if not d0.exists():
            print(f'{tag:22s} 目录不存在: {d0}')
            continue
        # 推断原始分辨率：从图片读取一张
        img = None
        for cand in (d0 / 'images' / 'train').glob('*'):
            img = cand
            break
        if img is None:
            print(f'{tag:22s} 无图片')
            continue
        try:
            import cv2
            im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            H0, W0 = im.shape[:2]
        except Exception:
            W0, H0 = 580, 454
        sc = lb_scale(W0, H0)
        for lab in (d0 / 'labels').rglob('*.txt'):
            for line in lab.read_text().splitlines():
                s = line.split()
                if len(s) >= 5:
                    h = float(s[4]) * H0
                    hs_raw.append(h)
                    hs_lb.append(h * sc)
        report(tag, hs_raw, hs_lb, f'原生 {W0}x{H0}，scale={sc:.3f}')

    print('\n--- 判读参考 ---')
    print('  可检测下限经验值：目标高度 ≥ 8px 才较可靠，< 4px 基本不可检')
    print('  若某数据集 letterbox 后 p25 低于 8px，说明该缩放不可接受')


if __name__ == '__main__':
    main()
