# YOLOMM 剪枝后微调训练示例脚本
#
# 注意事项:
# - 只接受带 prune_info 标记的剪枝权重 (由 model.prune(save_dir=...) 生成)
# - 不允许把原始 best.pt 当成 finetrain() 输入
# - 不要传 scale / model_scale 参数，剪枝后结构必须直接来自 checkpoint
# - 参数风格与 trainMM.py 一致，直接改参数、直接运行

from ultralytics import YOLOMM


if __name__ == '__main__':
    # 加载剪枝后的权重 (必须包含 prune_info)
    model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/YOLO-LST-pruned/pruned.pt')

    model.finetrain(
        data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
        epochs=50,
        batch=32,
        device='1',
        cache=True,
        exist_ok=True,
        project='./',
        name='PruneExp/test/yolomm-pruned-l1-r30-finetrain',
        # modality='RGB',   # 可选: 单模态微调
        # amp=False,         # 可选: 关闭混合精度
    )
