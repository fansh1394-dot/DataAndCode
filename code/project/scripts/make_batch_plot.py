"""复现 YOLO26 训练用的增强链，单独绘制指定的 train_batch<N> 面板。

用法（服务器）：
  python scripts/make_batch_plot.py --run YOLO26/ResTest/v26_E1_rgbonly_s0 --batch 39

做法：用 args.yaml 的同名超参构建数据集 → 取该 batch 的原始样本索引 →
按训练时相同顺序应用 mm_transforms（随机项以 --seed 固定）→ 喂给真实
MultiModalPlotter（带 bboxes_ir），输出 train_batch<N>_labels_{rgb,ir,multimodal}.jpg。
"""
import argparse
import sys
from copy import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter  # noqa: E402
from ultralytics.models.utils.multimodal.vis import (  # noqa: E402
    ensure_batch_idx_long,
    extract_precision_contract_from_batch,
)
from ultralytics.utils import YAML  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='训练 run 目录（含 args.yaml）')
    ap.add_argument('--batch', type=int, required=True, help='要绘制的 batch 序号')
    ap.add_argument('--out', default=None, help='输出目录，默认写回 run 目录')
    ap.add_argument('--seed', type=int, default=None, help='覆盖随机种子，默认取 args.yaml')
    ap.add_argument('--mode', default='dual', choices=['dual', 'rgb_only', 'x_only'],
                    help='面板模式：dual 同时输出 rgb/ir/multimodal 三张（默认）')
    ap.add_argument('--dual-labels', action='store_true',
                    help='强制用 dual 标签口径（labels=RGB, labels_ir=IR）。'
                         '不改原 data.yaml，只在 _d2p1/eval_yamls 下写一份派生 yaml。')
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    a = YAML.load(run_dir / 'args.yaml')
    cfg = get_cfg(overrides={k: v for k, v in a.items()
                             if k in ('imgsz', 'batch', 'cache', 'rect', 'single_cls',
                                      'fraction', 'seed', 'workers', 'task')})
    for k, v in a.items():
        if k.startswith('mm_'):
            setattr(cfg, k, v)
    for k in ('mosaic', 'close_mosaic', 'mixup', 'copy_paste', 'erasing',
              'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
              'shear', 'perspective', 'flipud', 'fliplr', 'bgr',
              'use_multimodal_aug', 'auto_augment', 'mask_ratio', 'overlap_mask'):
        if k in a:
            setattr(cfg, k, a[k])
    if args.seed is not None:
        cfg.seed = args.seed
    torch.manual_seed(int(cfg.seed))

    data = check_det_dataset(a['data'])
    if args.dual_labels:
        # 当前 data.yaml 可能是 vis/ir 口径（单模态训练用，labels.ir == labels）。
        # 绘图要看 IR 标签，必须用 dual 口径；派生一份 yaml，不动原文件（训练在跑）。
        import yaml as _yaml
        src = Path(a['data'])
        if not src.is_absolute():
            src = (Path('/root/rivermind-data/uav_data_v2') / src.name)
        raw = _yaml.safe_load(src.read_text())
        raw['labels'] = {'rgb': 'labels', 'ir': 'labels_ir'}
        derive_dir = run_dir.parents[1] / '_d2p1' / 'eval_yamls'
        derive_dir.mkdir(parents=True, exist_ok=True)
        dst = derive_dir / f'plot_dual_{src.stem}.yaml'
        dst.write_text(_yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
        data = check_det_dataset(str(dst))
        print(f'[info] 使用派生 dual 口径 yaml: {dst}')
    else:
        print(f'[warn] 沿用原 data.yaml 的 labels 映射: {data.get("labels")}')
    from ultralytics.nn.mm.utils import resolve_modality_pair
    _rgb, x_mod = resolve_modality_pair(data, strict=True)
    from ultralytics.nn.mm.utils import get_modality_path_from_data
    x_dir = get_modality_path_from_data(data, x_mod)

    bs = int(a.get('batch', 32))
    ds = build_yolo_dataset(cfg, data['train'], bs, data, mode='train', rect=False,
                            stride=32, multi_modal_image=True,
                            x_modality=x_mod, x_modality_dir=x_dir)
    print(f'[info] 数据集 {ds.ni} 帧, batch={bs}, 目标 batch={args.batch}')

    start = args.batch * bs
    idxs = list(range(start, min(start + bs, ds.ni)))
    if not idxs:
        print('[错误] batch 超出数据集范围'); return 2
    print(f'[info] 采样索引 {idxs[0]}..{idxs[-1]} ({len(idxs)} 帧)')

    samples = [ds[i] for i in idxs]
    batch = ds.collate_fn(samples)

    out_dir = Path(args.out).resolve() if args.out else run_dir
    xch = int(data.get('Xch', 3))
    bboxes_x = batch.get('bboxes_ir')
    bboxes_x = bboxes_x if isinstance(bboxes_x, torch.Tensor) and bboxes_x.numel() > 0 else None
    cls_x = batch.get('cls_ir')
    cls_x = cls_x.squeeze(-1) if (bboxes_x is not None and isinstance(cls_x, torch.Tensor)) else None
    bidx_x = batch.get('batch_idx_ir')
    bidx_x = ensure_batch_idx_long(bidx_x) if (bboxes_x is not None and isinstance(bidx_x, torch.Tensor)) else None

    # 归一化到 [0,1]：plotter 的 legacy 精度路径直接 clamp(0,1)，
    # 而 uint8 数据（0~255）会被整体截成 1 → 整张图纯白。
    # 训练期 batch img 也是 uint8，正常路径靠 preprocess 归一化；
    # 这里在绘图前显式做 /255，与 legacy 语义一致。
    imgs = batch['img']
    if imgs.dtype == torch.uint8:
        imgs = imgs.float() / 255.0
    else:
        imgs = imgs.float()
        if float(imgs.max()) > 1.5:
            imgs = imgs / 255.0

    MultiModalPlotter(out_dir, on_plot=None).plot_batch(
        stage='train', source='labels', images=imgs,
        batch_idx=ensure_batch_idx_long(batch['batch_idx']),
        cls=batch['cls'].squeeze(-1), bboxes=batch['bboxes'],
        bboxes_x=bboxes_x, cls_x=cls_x, batch_idx_x=bidx_x,
        xch=xch,
        x_modality_name=MultiModalPlotter.resolve_x_modality_name(
            getattr(a, 'single_modal', None), data, None),
        precision_contract=extract_precision_contract_from_batch(batch),
        modality_mode=args.mode,
        paths=batch['im_file'], batch_ni=args.batch,
        threaded=False,
    )
    print('[done] 输出目录:', out_dir)
    for p in sorted(out_dir.glob(f'train_batch{args.batch}_labels_*.jpg')):
        print('  ', p.name, p.stat().st_size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
