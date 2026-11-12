# -*- coding: utf-8 -*-
# 本地双光推理演示脚本（在 PyCharm 中直接运行此文件）
# 方案 A：验证集图片成对预测；方案 B：真实视频（先抽帧再配对推理）
import os

import cv2

from ultralytics import YOLOMM
from ultralytics.tools import MultiModalSampler

WEIGHTS = r"C:\Users\不如喝茶去\Desktop\双光谱毕设\ResTest\yolomm-c3k2-pkimodule\weights\best_local.pt"
DATA_YAML = r"D:\uav_data\data.yaml"

model = YOLOMM(WEIGHTS)

# ========== 方案 A：验证集图片成对预测 ==========
sampler = MultiModalSampler(DATA_YAML, split='val')
rgb_list, x_list = sampler.sample_source_list(n=10)

print("=== 双模态预测（RGB + IR 图片）===")
model.predict(rgb_source=rgb_list,
              x_source=x_list,
              project='ResTest',
              name='predict_img_dual',
              save=True,
              exist_ok=True)

# ========== 方案 B：真实视频预测（抽帧 → 图片对推理） ==========
# 多模态推理只支持图片（inference_dataset 为 images only），视频需先抽帧。
# 视频已复制到无中文目录 D:/uav_data/demo_videos/
RGB_VIDEO = r"D:\uav_data\demo_videos\visible.mp4"
IR_VIDEO = r"D:\uav_data\demo_videos\infrared.mp4"
FRAME_RGB = r"D:\uav_data\demo_frames\rgb"
FRAME_IR = r"D:\uav_data\demo_frames\ir"

if not (os.path.exists(RGB_VIDEO) and os.path.exists(IR_VIDEO)):
    print("[跳过] 未找到视频对，请先确认已复制视频到: D:/uav_data/demo_videos/")
else:
    if not (os.path.exists(FRAME_RGB) and os.path.exists(FRAME_IR)):
        os.makedirs(FRAME_RGB, exist_ok=True)
        os.makedirs(FRAME_IR, exist_ok=True)
        cap_r = cv2.VideoCapture(RGB_VIDEO)
        cap_i = cv2.VideoCapture(IR_VIDEO)
        idx = 0
        saved = 0
        while True:
            ok_r, fr_r = cap_r.read()
            ok_i, fr_i = cap_i.read()
            if not ok_r or not ok_i:
                break
            if idx % 10 == 0:
                name = "frame_%06d.jpg" % saved
                cv2.imwrite(os.path.join(FRAME_RGB, name), fr_r)
                cv2.imwrite(os.path.join(FRAME_IR, name), fr_i)
                saved += 1
            idx += 1
        print("抽帧完成:", saved, "对")
    else:
        print("帧目录已存在，直接推理")

    names = sorted(os.listdir(FRAME_RGB))
    rgb_list = [os.path.join(FRAME_RGB, n) for n in names]
    ir_list = [os.path.join(FRAME_IR, n) for n in names]
    print("=== 双模态预测（视频抽帧 %d 对）===" % len(rgb_list))
    model.predict(rgb_source=rgb_list,
                  x_source=ir_list,
                  project='ResTest',
                  name='predict_video_dual',
                  save=True,
                  exist_ok=True)
