# -*- coding: utf-8 -*-
"""仅统计 Mendeley 的目标尺寸分布（容错解码 + 格式自适应）。"""
import sys
from pathlib import Path

import numpy as np

TW, TH = 640, 512
base = Path('/3250410011/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/'
            'Labelled_UAV_Images/Labelled EO_IR Dataset')


def read_lines(p):
    b = p.read_bytes()
    for enc in ('utf-8', 'gbk', 'latin-1'):
        try:
            return b.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return b.decode('latin-1', 'ignore').splitlines()


def get_wh(d0):
    try:
        import cv2
        for sub in ('images/train', 'images/val', 'images/test', 'images'):
            d = d0 / sub
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                        if im is not None:
                            return im.shape[1], im.shape[0]
    except Exception:
        pass
    return 580, 454


def main():
    print(f'=== Mendeley 在 {TW}x{TH} letterbox 下的目标尺寸 ===\n')
    for tag, sub in (('C. Mendeley EO', 'UAV_eo_Dataset'), ('C. Mendeley IR', 'UAV_ir_Dataset')):
        d0 = base / sub
        if not d0.exists():
            print(f'{tag:22s} 目录不存在')
            continue
        W0, H0 = get_wh(d0)
        sc = min(TW / W0, TH / H0)
        hs_raw, hs_lb = [], []
        n_files = 0
        bad = 0
        lab_dir = d0 / 'labels'
        for lab in lab_dir.rglob('*.txt'):
            n_files += 1
            for line in read_lines(lab):
                s = line.replace(',', ' ').split()
                if len(s) < 5:
                    continue
                try:
                    h = float(s[4])
                except ValueError:
                    bad += 1
                    continue
                # 兼容归一化与像素两种
                if h <= 1.5:
                    h = h * H0
                hs_raw.append(h)
                hs_lb.append(h * sc)
        if not hs_raw:
            print(f'{tag:22s} 标注目录 {lab_dir} 无有效框（文件 {n_files}，解析失败 {bad}）')
            continue
        a, b = np.array(hs_raw), np.array(hs_lb)
        print(f'{tag:22s} n={len(a):6d}  原生 {W0}x{H0}  scale={sc:.3f}  标注文件={n_files}')
        print(f'  原始高度     : 中位 {np.median(a):6.1f}  均值 {a.mean():6.1f}  '
              f'p5={np.percentile(a,5):5.1f} p25={np.percentile(a,25):5.1f} '
              f'p75={np.percentile(a,75):5.1f} p95={np.percentile(a,95):5.1f}')
        print(f'  letterbox 后 : 中位 {np.median(b):6.1f}  均值 {b.mean():6.1f}  '
              f'p5={np.percentile(b,5):5.1f} p25={np.percentile(b,25):5.1f} '
              f'p75={np.percentile(b,75):5.1f} p95={np.percentile(b,95):5.1f}')
        for thr in (2, 4, 8, 16, 32):
            print(f'    高度 < {thr:2d}px : {np.mean(b < thr) * 100:5.1f}%')
        print()


if __name__ == '__main__':
    main()
