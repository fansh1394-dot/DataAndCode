import cv2
import numpy as np

D = '/root/rivermind-data/uav_data_v2/'
st = '20190925_101846_1_1_00000'
a = cv2.imread(D + 'images/train/' + st + '.jpg')
b = cv2.imread(D + 'images_ir/train/' + st + '.jpg', 0)
print('rgb', a.shape, 'mean', round(float(a.mean()), 2), 'p1/p50/p99', np.percentile(a, [1, 50, 99]))
print('ir ', b.shape, 'mean', round(float(b.mean()), 2), 'p1/p50/p99', np.percentile(b, [1, 50, 99]))
print('label rgb :', open(D + 'labels/train/' + st + '.txt').read().strip())
print('label ir  :', open(D + 'labels_ir/train/' + st + '.txt').read().strip())
