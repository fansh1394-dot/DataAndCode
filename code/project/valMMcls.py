# 这是提供参考示例的 YOLOMM 分类（CLS）验证器使用方法
# 必须按照你的需求来更改，而不是盲目使用

from ultralytics import YOLOMM


if __name__ == "__main__":
    # 使用训练得到的权重
    model = YOLOMM("/mnt/SSD/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/Test/CLS_MM/weights/best.pt")

    model.val(
        data="/home/zhizi/work/multimodel/ultralyticmm/datasets/cifar10/cifar10.yaml",
        split="val",
        device="0",
        imgsz=224,
        # modality="rgb",  # 模态消融参数：非必要不要开启
        project="ResTest",
        name="Val_CLS_MM",
    )

