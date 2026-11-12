# RTDETRMM 剪枝后微调训练示例脚本
#
# 注意事项:
# - 只接受带 prune_info 标记的剪枝权重 (由 model.prune(save_dir=...) 生成)
# - 不允许把原始 best.pt 当成 finetrain() 输入
# - 不要传 scale / model_scale 参数，剪枝后结构必须直接来自 checkpoint
# - 参数风格与 trainRT.py 一致，直接改参数、直接运行

from ultralytics import RTDETRMM


if __name__ == '__main__':
    # 加载剪枝后的权重 (必须包含 prune_info)
    model = RTDETRMM('./RTDETRMM-pruned/pruned.pt')

    model.finetrain(
        data='/path/to/data.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        device='0',
        cache=True,
        exist_ok=True,
        project='./',
        name='PruneExp/RTDETRMM-r18-mm-mid-finetrain',
        # modality='RGB',   # 可选: 单模态微调
        # amp=False,         # 可选: 关闭混合精度
    )
