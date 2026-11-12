# 这是提供参考示例的 YOLOMM 分类（CLS）推理器使用方法
# 必须按照你的需求来更改，而不是盲目使用

from pathlib import Path

from ultralytics import YOLOMM


if __name__ == "__main__":
    # 使用训练得到的权重
    model = YOLOMM("/mnt/SSD/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/Test/CLS_MM/weights/best.pt")

    # 示例路径：使用仓库内的两张图片占位，你需要替换为"同一场景配对"的 RGB / X 模态样本
    root = Path("/mnt/SSD/work/multimodel/ultralyticmm/datasets/cifar10")
    rgb_img = (root / "images/val/val_00000.png").as_posix()
    x_img = (root / "images_edge/val/val_00000.png").as_posix()

    common_kwargs = dict(project="ResTest", save=True, exist_ok=True)

    print("=== 测试双模态 CLS 预测 ===")
    # 重要：双模态输入使用 [rgb, x]；并将 batch=2，确保两张图进入同一 batch 触发双模态预处理
    model.predict(
        [rgb_img, x_img],
        name="CLS_MM_dual_modal",
        batch=2,
        **common_kwargs,
    )

    print("\n=== 测试单模态 RGB CLS 预测 ===")
    model.predict(
        rgb_img,
        name="CLS_MM_single_rgb",
        modality="rgb",
        **common_kwargs,
    )

    print("\n=== 测试单模态 X CLS 预测 ===")
    model.predict(
        x_img,
        name="CLS_MM_single_x",
        modality="x",
        **common_kwargs,
    )

