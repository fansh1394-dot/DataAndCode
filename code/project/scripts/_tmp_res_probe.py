# -*- coding: utf-8 -*-
"""打印一帧 RGB / IR 图像的真实分辨率。"""
from pathlib import Path

import cv2

root = Path('/root/rivermind-data/uav_data_v2')
for sub in ('images/test', 'images_ir/test'):
    files = sorted((root / sub).glob('*'))
    if not files:
        print(sub, 'empty'); continue
    for f in files[:2]:
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        print(sub, f.name, None if im is None else im.shape)
