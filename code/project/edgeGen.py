"""
EdgeGen 离线 Edge（边缘）模态生成示例脚本。

用法与 depthGen.py / DEMGen.py 对齐：只需初始化 EdgeGen，然后 run(data.yaml)。

默认保存目录：
- 当 save_dir=None 且输入为 data.yaml 时，输出到 data.yaml 对应的 images_edge/<split>/ 下。

训练/推理配置建议（data.yaml）：
  modality:
    rgb: images
    edge: images_edge
  x_modality: edge
  Xch: 3
"""

from ultralytics import EdgeGen


if __name__ == "__main__":
    gen = EdgeGen(
        source_modality="rgb",  # 从哪个模态生成 Edge，默认 rgb
        device="cuda:0",
        save_dir=None,          # None: 默认保存到 images_edge/<split>
        split=None,             # 可选: "train" / "val" / "test" / "train,val"
        save_format="png",      # 'png' | 'npy' | 'tif'（可选: 'npz'），默认 png
        xch=1,                  # 输出通道数：1 或 3
        batch_size=8,
        num_workers=0,
        # 算法参数（Sobel/Scharr 边缘强度）
        gaussian_ksize=5,
        sobel_ksize=3,
        use_scharr=False,
        normalize=True,
        gamma=1.0,              # <1 更“亮”，>1 更“尖”
        binarize=False,         # True 可输出二值边缘
        threshold=0.3,          # binarize=True 时生效（0~1）
    )

    stats = gen.run("/root/datasets/obb/obb.yaml") # 数据集yaml 加载 支持 YOLO 或者 YOLOMM 格式
    print(f"生成完成: 成功 {stats.success}/{stats.total}, 失败 {stats.failed}")
    if stats.failures:
        print("失败列表:")
        for path, err in stats.failures:
            print(f"- {path}: {err}")
