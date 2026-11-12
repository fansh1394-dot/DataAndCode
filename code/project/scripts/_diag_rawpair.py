import cv2
import numpy as np

D = '/root/rivermind-data/uav_data_v2/'
S = ['20190925_101846_1_1_00000', '20190925_101846_1_1_00050']
out = []
for st in S:
    a = cv2.imread(D + 'images/train/' + st + '.jpg')
    b = cv2.imread(D + 'images_ir/train/' + st + '.jpg', 0)
    b3 = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    out.append(np.hstack([a, b3]))
    print(st, a.shape, b.shape, 'rgb mean', round(float(a.mean()), 1), 'ir mean', round(float(b.mean()), 1))
cv2.imwrite('/tmp/raw_pair.jpg', np.vstack(out))
print('saved /tmp/raw_pair.jpg')
