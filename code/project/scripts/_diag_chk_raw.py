import cv2
import numpy as np

D = '/root/rivermind-data/uav_data_v2/'
OUT = '/root/rivermind-data/MutilModel_3485988898/YOLO26/logs/_diag_panel/'
S = ['20190925_101846_1_1_00000', '20190925_101846_1_1_00050']

rows = []
for st in S:
    rgb = cv2.imread(D + 'images/train/' + st + '.jpg')
    ir = cv2.imread(D + 'images_ir/train/' + st + '.jpg', 0)
    ir3 = np.repeat(ir[..., None], 3, axis=2)
    v = np.hstack([rgb, ir3])
    for path, color, tag in ((D + 'labels/train/' + st + '.txt', (0, 200, 255), 'rgb'),
                             (D + 'labels_ir/train/' + st + '.txt', (0, 0, 255), 'ir')):
        for ln in open(path).read().strip().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            x, y, w, h = [float(t) for t in p[1:5]]
            for off in (0, 640):
                x1, y1 = int((x - w / 2) * 640) + off, int((y - h / 2) * 512)
                x2, y2 = int((x + w / 2) * 640) + off, int((y + h / 2) * 512)
                cv2.rectangle(v, (x1, y1), (x2, y2), color, 1)
                cv2.putText(v, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    rows.append(cv2.resize(v, (1280, 512)))
cv2.imwrite('/tmp/chk_raw.jpg', np.vstack(rows))
print('saved /tmp/chk_raw.jpg')
