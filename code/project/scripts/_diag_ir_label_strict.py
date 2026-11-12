"""严格核验：labels_ir 的框是否真的圈住 images_ir 里的无人机热斑。

判定不依赖目测，用三个互相独立的证据：
  E1 框内是否存在显著热斑（框内峰值 - 框外背景中位数）
  E2 该帧最显著热斑的质心是否落在框内（容差 0 像素）
  E3 热斑连通域被框覆盖的面积比
同时在 train / val 两个 split、全量随机抽样（不只 400 帧）上给出比例。
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

D = Path('/root/rivermind-data/uav_data_v2')
N_SAMPLE = 3000


def analyze(ir, box):
    """ir: HxW uint8; box: (cx,cy,w,h) 归一化。返回 (contrast, inside, cover)。"""
    H, W = ir.shape
    cx, cy, bw, bh = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    x1, y1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    x2, y2 = int(min(W, cx + bw / 2)), int(min(H, cy + bh / 2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    g = cv2.GaussianBlur(ir, (5, 5), 0).astype(np.float32)
    # 局部窗口（框外扩 45px）
    r = 45
    wy0, wy1 = int(max(0, cy - bh / 2 - r)), int(min(H, cy + bh / 2 + r))
    wx0, wx1 = int(max(0, cx - bw / 2 - r)), int(min(W, cx + bw / 2 + r))
    win = g[wy0:wy1, wx0:wx1]
    if win.size < 64:
        return None
    bg = float(np.median(win))
    pmax = float(win.max())
    contrast = pmax - bg
    if contrast < 40:            # 无显著热斑（可能目标很弱/被云遮挡）
        return (contrast, None, None, 'weak')
    # 热斑连通域：取最亮像素所在块
    thr = bg + 0.35 * contrast
    mask = (win >= thr).astype(np.uint8)
    num, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return (contrast, None, None, 'weak')
    py, px = np.unravel_index(int(np.argmax(win)), win.shape)
    k = lab[py, px]
    if k == 0:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bwd, bht = stats[k, :4]
    gx1, gy1 = bx + wx0, by + wy0
    gx2, gy2 = gx1 + bwd, gy1 + bht
    tc = cents[k]
    tcx, tcy = tc[0] + wx0, tc[1] + wy0
    inside = (x1 <= tcx <= x2) and (y1 <= tcy <= y2)
    inter = max(0, min(x2, gx2) - max(x1, gx1)) * max(0, min(y2, gy2) - max(y1, gy1))
    cover = inter / max(1.0, bwd * bht)
    return (contrast, inside, cover, 'ok')


for split in ('train', 'val', 'test'):
    meta = D / f'metadata/{split}.json'
    if not meta.exists():
        continue
    frames = json.load(open(meta))['frames']
    both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
    random.seed(2024)
    sel = random.sample(both, min(N_SAMPLE, len(both)))
    n = weak = hit = miss = 0
    covers = []
    contrasts = []
    for s in sel:
        lp = D / f'labels_ir/{split}' / (s + '.txt')
        ip = D / f'images_ir/{split}' / (s + '.jpg')
        if not (lp.exists() and ip.exists()):
            continue
        try:
            box = np.loadtxt(lp, ndmin=2).reshape(-1, 5)
        except Exception:
            continue
        if len(box) != 1:
            continue
        ir = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if ir is None:
            continue
        r = analyze(ir, box[0][1:5])
        if r is None:
            continue
        contrast, inside, cover, tag = r
        n += 1
        if tag == 'weak':
            weak += 1
            continue
        contrasts.append(contrast)
        covers.append(cover)
        if inside:
            hit += 1
        else:
            miss += 1
    valid = hit + miss
    print(f'[{split}] 抽样 {n} 帧 | 无显著热斑 {weak} | 可判定 {valid}')
    if valid:
        cs = np.array(covers)
        print(f'   热斑质心落在框内 : {hit}/{valid} = {hit/valid:.4f}')
        print(f'   热斑被框覆盖率   : p10={np.percentile(cs,10):.2f} '
              f'p50={np.percentile(cs,50):.2f} p90={np.percentile(cs,90):.2f} '
              f'| >=0.9 占比 {(cs>=0.9).mean():.3f} | <=0.1 占比 {(cs<=0.1).mean():.3f}')
        print(f'   框内对比度       : p10={np.percentile(contrasts,10):.0f} '
              f'p50={np.percentile(contrasts,50):.0f}')
