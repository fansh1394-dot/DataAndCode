# -*- coding: utf-8 -*-
"""DualHeadFusionDetectV26 冒烟测试：构建 / 前向 / 损失 / 迭代一步。

用法（在 YOLO26/MutilModel_3485988898 下）：
    /opt/conda/envs/rivermind/bin/python scripts/smoke_dual_head_v26.py
"""
import sys
from pathlib import Path

import torch

# 服务器为多核 CPU（默认 torch 线程数可达 96+），小张量下线程争用会让前向慢一个数量级，
# 冒烟只需验证结构，故显式限流。
torch.set_num_threads(4)

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

CFG = 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
NC = 1
IMG_SZ = 320  # 冒烟用较小分辨率，结构检查不受影响


def hr(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


def main():
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.mm.dual_head_v26 import DualHeadFusionDetectV26

    hr('1. 构建模型')
    model = DetectionModel(cfg=CFG, ch=6, nc=NC, verbose=False)
    m = model.model[-1]
    assert isinstance(m, DualHeadFusionDetectV26), type(m)
    print(f'聚合头: {m}')
    print(f'train_branch_names: {m.train_branch_names}')
    print(f'has_fuse_ir: {m.has_fuse_ir}')
    print(f'end2end: {m.end2end}  模型 stride: {model.stride.tolist()}')
    for name in m.train_branch_names:
        h = m.branch(name)
        print(f'  {name:8s} type={type(h).__name__:12s} no={h.no} reg_max={h.reg_max} '
              f'stride={h.stride.tolist()} end2end={h.end2end}')

    n_par = sum(p.numel() for p in model.parameters())
    print(f'总参数: {n_par/1e6:.2f} M')

    hr('2. 训练态前向（结构校验）')
    model.train()
    x = torch.zeros(2, 6, IMG_SZ, IMG_SZ)
    with torch.no_grad():
        out = model(x)
    print(f'输出类型: {type(out).__name__}, keys={list(out) if isinstance(out, dict) else "n/a"}')
    for name in m.train_branch_names:
        raw = out[name]
        om = m.one2many(raw)
        oo = m.one2one(raw)
        print(f'  {name:8s} raw={type(raw).__name__} '
              f'one2many={[tuple(t.shape) for t in om]} '
              f'one2one={[tuple(t.shape) for t in oo]}')

    hr('3. 计算损失（构建 criterion 并前向）')
    batch = {
        'img': x,
        'batch_idx': torch.tensor([0.0, 0.0, 1.0]),
        'cls': torch.tensor([[0.0], [0.0], [0.0]]),
        'bboxes': torch.tensor([[0.50, 0.50, 0.08, 0.08],
                                [0.30, 0.40, 0.06, 0.06],
                                [0.60, 0.55, 0.10, 0.10]]),
        'bboxes_ir': torch.tensor([[0.52, 0.46, 0.08, 0.08],
                                   [0.32, 0.36, 0.06, 0.06],
                                   [0.62, 0.51, 0.10, 0.10]]),
        'cls_ir': torch.tensor([[0.0], [0.0], [0.0]]),
        'batch_idx_ir': torch.tensor([0.0, 0.0, 1.0]),
        'valid_rgb': torch.tensor([True, True]),
        'valid_ir': torch.tensor([True, True]),
    }
    model.args = type('A', (), {
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
        'mm_dual_w_rgb': 1.0, 'mm_dual_w_ir': 1.0, 'mm_dual_w_fuse': 1.0,
        'mm_dual_w_fuse_ir': 1.0,
        'mm_dfs_weight': 0.0, 'mm_dfs_mode': 'cls', 'mm_dfs_temperature': 1.0,
        'mm_dual_consist_weight': 0.0, 'mm_dual_w_fuse_ir_consist': 0.0,
        'mm_dual_w_fuse_ir_distill': 0.0, 'mm_gate_reg_weight': 0.0,
        'mm_gate_reg_target': 0.15, 'mm_fuse_ir_label_always': False,
        'loss_assign_metric': 'iou',
    })()

    criterion = model.init_criterion()
    print(f'criterion: {type(criterion).__name__}')
    print(f'  end2end={criterion.end2end} prog_loss={criterion.prog_loss} '
          f'o2m={criterion.o2m:.2f} o2o={criterion.o2o:.2f} '
          f'o2o_heads={list(criterion.branch_criteria_o2o) if criterion.branch_criteria_o2o else None}')
    preds = model(batch['img'])
    loss, items = criterion(preds, batch)
    lt = loss.detach()
    print(f'loss: shape={tuple(lt.shape)} sum={float(lt.sum()):.4f} '
          f'有限值={bool(torch.isfinite(lt).all())}')
    print(f'loss_items = {[round(float(v), 4) for v in items]}')

    hr('4. 反向传播')
    loss.sum().backward()
    params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    no_grad = [n for n, p in params if p.grad is None]
    nonfinite = [n for n, p in params if p.grad is not None and not torch.isfinite(p.grad).all().item()]
    print(f'聚合头参数 {len(params)} 个：有梯度 {len(params) - len(no_grad)}，'
          f'无梯度 {len(no_grad)}，非有限梯度 {len(nonfinite)}')
    if no_grad:
        print(f'  无梯度样例: {no_grad[:3]}')
    if nonfinite:
        print(f'  非有限样例: {nonfinite[:3]}')

    hr('5. 推理态前向（NMS-free 路径）')
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f'eval 输出类型: {type(out_eval).__name__}')
    if isinstance(out_eval, dict):
        for name in m.train_branch_names:
            print(f'  {name}: {type(out_eval[name]).__name__}')
    elif isinstance(out_eval, (list, tuple)) and len(out_eval) == 2:
        y, raw = out_eval
        print(f'  y.shape = {tuple(y.shape) if torch.is_tensor(y) else type(y)}')
        print(f'  raw keys = {list(raw) if isinstance(raw, dict) else type(raw)}')

    hr('结果：全部检查通过')


if __name__ == '__main__':
    main()
