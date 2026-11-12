"""RTDETRMM OBB training entry script."""

from ultralytics import RTDETRMM


if __name__ == "__main__":
    model = RTDETRMM("rtdetr-r18-mm-mid-obb.yaml") #m模型选择
    model.train(
        data="/root/datasets/obb/obb_edge.yaml",  #数据集选择
        epochs=100,
        # imgsz=256,
        cache=True,
        workers=8,
        device=0,
        batch=16,
        optimizer='AdamW' ,# 推荐使用 AdamW 用于收敛稳定
        lr0= 1e-4, #初始学习率
        lrf= 1e-3, #最终学习率
        # project='ResTest', name='RTDETRMM-obb-Test')
        project='ResTest', name='RTDETRMM-obb-edge-Test')
