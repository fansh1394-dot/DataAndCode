# 这是提供参考示例的 RTDETRMM 剪枝脚本（零注册引擎）
#
# 与 pruneMM.py 同一引擎：依赖发现全自动。RTDETR 的结构约束（AIFI embed 固化、
# decoder 宽度耦合、denoising embedding）由引擎的算子级追踪与围栏自动保护
# （保守跳过并在 prune_spec.json 中说明），不再依赖类型白名单保命。
# 实际可剪范围自动收敛到双模态骨干 + neck 卷积段。
from pathlib import Path

from ultralytics import RTDETRMM

if __name__ == '__main__':
    # 正式剪枝时，建议直接加载已经训练完成的权重，而不是空白 YAML。
    model = RTDETRMM('/path/to/RTDETRMM-r18-mm-mid/weights/best.pt')

    # 剪枝结果会在该目录下生成：
    # 1. pruned.pt         -> 运行时权威结构（后续 finetrain/val/predict 用它）
    # 2. pruned.yaml       -> 完整可加载配置（配合 state_dict 可重建剪后结构）
    # 3. prune_spec.json   -> 逐单元剪枝决策 + 保守跳过原因清单
    # 4. prune_report.json -> 引擎完整报告（save_report=True 时）
    prune_save_dir = Path('./RTDETRMM-pruned/')

    model.prune(
        method='l1',       # 可选: 'l1', 'l2', 'lamp', 'bn', 'random'（注册制，可扩展）
        ratio=0.30,        # 目标剪枝比例
        save_dir=str(prune_save_dir),
        round_to=8,        # 保留通道数按 8 对齐，便于部署
        min_ch=8,          # 每个剪枝单元至少保留的通道数
        save_report=True,
    )

    # ===== 非结构化 / 2:4 半结构化稀疏 =====
    # model.prune_sparse(sparsity=0.5, pattern='unstructured', save_dir='./RTDETRMM-sparse/')

    # 剪枝后可直接微调（无需落盘重载；DDP 多卡时请从 pruned.pt 重载）：
    # model.finetrain(data='data.yaml', epochs=50, device='0')
    #
    # 注意：剪枝模型调用 train() 会直接报错（防止按 YAML 重建丢权重），必须用 finetrain()。
