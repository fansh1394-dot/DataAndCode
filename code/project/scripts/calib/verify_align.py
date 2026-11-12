# -*- coding: utf-8 -*-
# 最终验收：对比 原始 / 仅修根因 / 修根因+配准补偿 三种结果
import os
import glob
import numpy as np
import cv2

RGB_DIR = r'D:\uav_data\demo_frames\rgb'
IR_DIR = r'D:\uav_data\demo_frames\ir'
OUT = r'D:\uav_data\_vis'

RESTEST = r'c:\Users\不如喝茶去\Desktop\双光谱毕设\MutilModel_3485988898\ResTest'
CASES = [
    ('0-原始(未修)', os.path.join(RESTEST, 'predict_video_dual')),
    ('1-修根因(无配准)', os.path.join(RESTEST, 'predict_video_dual_fixed')),
    ('2-修根因+平移补偿', os.path.join(RESTEST, 'predict_video_dual_fixed_reg')),
    ('3-修根因+仿射补偿', os.path.join(RESTEST, 'predict_video_dual_fixed_aff')),
]


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def ir_target(img):
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    res = np.clip(g - bg, 0, None)
    mask = np.zeros((h, w), bool)
    mask[120:460, 120:560] = True
    res = np.where(mask, res, 0)
    idx = np.argsort(res.ravel())[::-1][:60]
    ys, xs = np.unravel_index(idx, res.shape)
    return xs.mean() / w, ys.mean() / h


def find_box(img):
    b, g, r = img[:, :, 0].astype(np.int16), img[:, :, 1].astype(np.int16), img[:, :, 2].astype(np.int16)
    m = (b > 150) & (g < 100) & (r < 100)
    if m.sum() == 0:
        return None
    ys, xs = np.where(m)
    h, w = img.shape[:2]
    return (xs.min(), ys.min(), xs.max(), ys.max()), xs.mean() / w, ys.mean() / h


rgb_n = sorted(glob.glob(os.path.join(RGB_DIR, '*.jpg')))
ir_n = sorted(glob.glob(os.path.join(IR_DIR, '*.jpg')))
N = min(len(rgb_n), len(ir_n))

summary = []
for tag, d in CASES:
    dxl, dyl, hit, tot, sizes, pxs = [], [], 0, 0, [], []
    for i in range(N):
        fir = os.path.join(d, os.path.basename(ir_n[i]).replace('.jpg', '_ir.jpg'))
        if not os.path.exists(fir):
            continue
        im = imread_u(fir)
        if im is None:
            continue
        bx = find_box(im)
        if bx is None:
            continue
        g = imread_u(ir_n[i])
        if g is None:
            continue
        tx, ty = ir_target(g)
        cx, cy = bx[1], bx[2]
        ih, iw = im.shape[:2]
        dxl.append(cx - tx)
        dyl.append(cy - ty)
        pxs.append((cx * iw - tx * iw, cy * ih - ty * ih))
        tot += 1
        x0, y0, x1, y1 = bx[0]
        sizes.append((x1 - x0, y1 - y0))
        if x0 <= tx * iw <= x1 and y0 <= ty * ih <= y1:
            hit += 1
    if tot == 0:
        print(f'{tag}: 无可评估样本')
        continue
    dxl, dyl = np.array(dxl), np.array(dyl)
    sx, sy = np.median(sizes, axis=0)
    px, py = np.median(np.array(pxs), axis=0)
    hw = (im.shape[1], im.shape[0])
    print(f'\n=== {tag} ===')
    print(f'  样本 {tot}   框画幅 {hw[0]}x{hw[1]}   框尺寸中位 {sx:.0f}x{sy:.0f}px')
    print(f'  框中心-热斑 归一化偏移 中位 ({np.median(dxl)*100:+.2f},{np.median(dyl)*100:+.2f}) pp  '
          f'std ({dxl.std()*100:.2f},{dyl.std()*100:.2f})')
    print(f'  换算像素 ({px:+.1f},{py:+.1f}) px')
    print(f'  >>> 热斑被框住率: {hit}/{tot} = {hit/tot*100:.1f}%')
    summary.append((tag, hw[0], sx, sy, np.median(dxl) * 100, np.median(dyl) * 100, px, py, hit / tot * 100))

print('\n\n########## 汇总 ##########')
print(f"{'方案':<22}{'画幅':<12}{'框尺寸':<12}{'偏移(pp)':<18}{'偏移(px)':<18}{'框住率'}")
for t, w, sx, sy, dx, dy, px, py, rate in summary:
    print(f'{t:<22}{w:<12}{f"{sx:.0f}x{sy:.0f}":<12}{f"({dx:+.2f},{dy:+.2f})":<18}'
          f'{f"({px:+.1f},{py:+.1f})":<18}{rate:.1f}%')
