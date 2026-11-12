# 这是提供参考示例的 YOLOMM 剪枝脚本（零注册引擎）
#
# 引擎说明：依赖发现全自动（算子级追踪），任何注册进 parse_model 的模块
# 要么被正确剪枝，要么被保守跳过并在报告中说明原因，不需要维护任何类型表。
# 全任务家族（detect/segment/pose/obb/classify/semseg/depth + RTDETRMM）均支持。
from pathlib import Path

from ultralytics import YOLOMM

if __name__ == '__main__':
    # 正式剪枝时，建议直接加载已经训练完成的权重，而不是空白 YAML。
    model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM-LST/weights/best.pt')

    # 剪枝结果会在该目录下生成：
    # 1. pruned.pt          -> 运行时权威结构（后续 finetrain/val/predict 用它）
    # 2. pruned.yaml        -> 完整可加载配置；配合 state_dict 可重建剪后结构
    #                          （rebuild_pruned_model：yaml 构建骨架 + 形状自适应 + strict 加载）
    # 3. prune_spec.json    -> 逐单元剪枝决策 + 保守跳过原因清单（审计）
    # 4. prune_report.json  -> 引擎完整报告（save_report=True 时）

    # ===== 结构化剪枝（物理移除通道，直接得到小模型）=====
    model.prune(
        method='l1',      # 可选: 'l1', 'l2', 'lamp', 'bn', 'random'（注册制，可扩展）
        ratio=0.30,       # 目标剪枝比例
        save_dir='./YOLO-LST-pruned/',
        round_to=8,       # 保留通道数按 8 对齐，便于部署
        min_ch=8,         # 每个剪枝单元至少保留的通道数
        save_report=True,
        # scope=[0, 1, 2, ...],  # 可选：限制剪枝的层号范围
    )

    # ===== 非结构化 / 2:4 半结构化稀疏（结构不变，权重置零）=====
    # model.prune_sparse(sparsity=0.5, pattern='unstructured', save_dir='./YOLO-LST-sparse/')
    # model.prune_sparse(pattern='semi24', save_dir='./YOLO-LST-24/')  # Ampere+ 稀疏加速

    # 剪枝后可直接微调（无需落盘重载；DDP 多卡时请从 pruned.pt 重载）：
    # model.finetrain(data='data.yaml', epochs=50, device='0')
    #
    # 或从落盘产物开始：
    # model = YOLOMM('./YOLO-LST-pruned/pruned.pt')
    # model.finetrain(data='data.yaml', epochs=50, device='0')
    #
    # 注意：剪枝模型调用 train() 会直接报错（防止按 YAML 重建丢权重），必须用 finetrain()。
