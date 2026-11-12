# YOLOMM Pose 训练脚本参考示例
# 请根据实际需求修改配置参数

from ultralytics import YOLOMM

if __name__ == '__main__':
    # 加载多模态姿态估计模型配置
    model = YOLOMM('ultralytics/cfg/models/mm/obb/yolo11/base/yolo11n-mm-obb.yaml')

    # 训练配置
    model.train(
        data='/root/datasets/obb/obb.yaml',
        epochs=20,
        batch=16,
        imgsz=640,
        scale='n',  # 模型规模: n/s/m/l/x
        cache=True,
        # exist_ok=True,
        project='ResTest',
        name='OBB-Test',
        # modality='edge',  # 模态消融参数，非必要不开启
        # loss_cls='bce',      # 分类损失类型: bce | focal (默认: bce)
        # loss_box='prob',     # OBB IoU 损失: prob | ciou | none (默认: prob，即基础 ProbIoU)
        # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
    )
