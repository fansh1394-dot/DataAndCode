"""
NormalGen 离线表面法向模态生成示例脚本。

用法与 depthGen.py / DEMGen.py / edgeGen.py 对齐：初始化 NormalGen，然后 run(data.yaml)。

默认保存目录：
- 当 save_dir=None 且输入为 data.yaml 时，输出到 data.yaml 对应的 images_normal/<split>/ 下。

训练/推理配置建议（data.yaml）：
  modality:
    rgb: images
    normal: images_normal
  x_modality: normal
  Xch: 3

深度来源：
- 已有 images_depth/（npy 或单通道 png/tif，不要用 Spectral 伪彩）时自动读取
- 否则传入 DepthGen 权重，从可见光现算深度再做法向
"""

from ultralytics import NormalGen


if __name__ == "__main__":
    gen = NormalGen(
        source_modality="rgb",
        device="cuda:0",
        save_dir=None,          # None: 默认保存到 images_normal/<split>
        split=None,             # 可选: "train" / "val" / "test" / "train,val"
        save_format="png",      # 'png' | 'npy' | 'tif'（可选: 'npz'）
        depth_source="auto",    # 'auto' | 'file' | 'generate'
        weights=None,           # generate / auto 缺深度图时必填 DepthGen 权重
        # depth_model="da",     # 'da' Depth Anything V2 / 'zd' ZipDepth
        # encoder="vitb",
        invert=True,            # DA/ZipDepth 相对深度默认 closer-is-larger，反转为 Z
        metric=False,           # True: 把深度当米制 Z，不再做分位拉伸
        fov_deg=60.0,           # 未知内参时的默认视场（与 DSINE 一致）
        jump_rel=0.08,          # 遮挡跳变阈值（相对深度）
        batch_size=1,
        num_workers=0,
    )

    stats = gen.run("/root/datasets/obb/obb.yaml")  # 数据集 yaml，支持 YOLO 或 YOLOMM 格式
    print(f"生成完成: 成功 {stats.success}/{stats.total}, 失败 {stats.failed}")
    if stats.failures:
        print("失败列表:")
        for path, err in stats.failures:
            print(f"- {path}: {err}")
