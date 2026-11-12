import cv2
import numpy as np

D = '/root/rivermind-data/uav_data_v2/'
OUT = '/root/rivermind-data/MutilModel_3485988898/YOLO26/logs/_diag_panel/'
S = ['20190925_101846_1_1_00000', '20190925_101846_1_1_00050',
     '20190926_195515_1_1_00220', '20190925_134750_1_1_00135']


def lb(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    rs = cv2.resize(img, (nw, nh))
    if rs.ndim == 2:
        rs = rs[..., None]
    c = rs.shape[2]
    canvas = np.full((size, size, c), 114, np.uint8)
    canvas[int(round(dh)):int(round(dh)) + nh, int(round(dw)):int(round(dw)) + nw] = rs
    return canvas, r, dw, dh


def draw(vis, line, color, r, dw, dh, tag):
    p = line.split()
    x, y, w, h = [float(v) for v in p[1:5]]
    cx, cy, bw, bh = x * 640 * r + dw, y * 512 * r + dh, w * 640 * r, h * 512 * r
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    cv2.putText(vis, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


rows = []
for st in S:
    rgb = cv2.imread(D + 'images/train/' + st + '.jpg')
    ir = cv2.imread(D + 'images_ir/train/' + st + '.jpg', 0)
    if rgb is None or ir is None:
        print('missing', st)
        continue
    v_rgb, r, dw, dh = lb(rgb)
    v_ir, r2, dw2, dh2 = lb(ir)
    v_ir3 = np.repeat(v_ir, 3, axis=2)
    lr = open(D + 'labels/train/' + st + '.txt').read().strip().splitlines()
    li = open(D + 'labels_ir/train/' + st + '.txt').read().strip().splitlines()
    for ln in lr:
        if ln.strip():
            draw(v_rgb, ln, (0, 200, 255), r, dw, dh, 'rgb')
            draw(v_ir3, ln, (0, 200, 255), r2, dw2, dh2, 'rgb+')
    for ln in li:
        if ln.strip():
            draw(v_ir3, ln, (0, 0, 255), r2, dw2, dh2, 'ir')
            draw(v_rgb, ln, (0, 0, 255), r, dw, dh, 'ir+')
    pair = np.hstack([v_rgb, v_ir3])
    cv2.imwrite(OUT + 'chk_' + st + '.jpg', pair)
    rows.append(cv2.resize(pair, (640, 320)))
    print('saved', st, 'n_rgb', len([l for l in lr if l.strip()]), 'n_ir', len([l for l in li if l.strip()]))
if rows:
    cv2.imwrite('/tmp/chk_grid.jpg', np.vstack(rows))
    print('grid /tmp/chk_grid.jpg')
