# 这是提供参考示例的 YOLOMM 分类（CLS）训练器使用方法
# 必须按照你的需求来更改，而不是盲目使用

from ultralytics import YOLOMM


if __name__ == "__main__":
    # 多模态分类模型配置（RGB+X）
    # 你也可以替换为自己的 mm-cls YAML 或 .pt 权重
    model = YOLOMM("yolo11n-mm-mid-cls.yaml")

    # 说明：
    # - data 需要是“YOLO 风格”的 YAML（与检测一致的 YAML 管理范式），并包含：
    #   - train/val 路径（指向 RGB images/<split> 目录）
    #   - modality_used: ['rgb', '<x_modality>'] 例如 ['rgb','depth'] / ['rgb','thermal']
    #   - modality 映射（可选）：{'rgb': 'images', '<x_modality>': 'images_<x_modality>'}
    #   - Xch: X 模态通道数（1/3/...）
    #   - names/nc 类别定义
    model.train(
        data="/home/zhizi/work/multimodel/ultralyticmm/datasets/cifar10/cifar10.yaml",
        epochs=2,
        batch=64,
        imgsz=224,
        # modality="rgb",  # 模态消融参数：非必要不要开启（单模态训练）
        # 注意: 分类任务使用交叉熵损失，不支持 loss_cls/loss_box/loss_dfl 参数
        cache=True,
        exist_ok=True,
        project="ResTest",
        name="Test/CLS_MM",
    )

