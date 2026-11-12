from ultralytics import DepthGen

if __name__ == "__main__":

    # 两套后端二选一：解开要用的 model/weights，注释掉另一套。
    # 权重从网盘 MMProject/DepthWeight/ 下载后改成本机路径。
    gen = DepthGen(
        # --- Depth Anything V2（默认，model="da" 可省略）---
        model="da",
        weights="/path/to/depth_anything_v2_vitb.pth",  # 填权重地址；同目录还有 vits / vitl
        # --- ZipDepth ---
        # model="zd",
        # weights="/path/to/zipdepth_base.pth",  # GPU unfold；NPU 头改 zipdepth_base_npu.pth
        device="0",
        save_dir=None,  # None: 默认保存到 data.yaml 对应 images_depth/<split>
        save_comparison=False,  # 对比图存到 images_depth_compare/<split>
        save_npy=False,
        save_color=True,
        save_raw16=False,
        split=None,  # 可选: "train", "val", "test" 或 "train,val"
        batch_size=1,
        num_workers=0,
    )

    stats = gen.run("/path/to/data.yaml")
    print(f"生成完成: 成功 {stats.success}/{stats.total}, 失败 {stats.failed}")
    if stats.failures:
        print("失败列表:")
        for path, err in stats.failures:
            print(f"- {path}: {err}")
