# -*- coding: utf-8 -*-
"""部署预检 step1：ONNX 导出 + 算子清单（不依赖 ultralytics 的 exporter，避免 torch 版本冲突）。

用法：
    PYTHONPATH=/3250410011/root/rivermind-data/deploy_env \
    /opt/conda/envs/rivermind/bin/python scripts/deploy_precheck.py onnx [yolo26|yolo11]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

PROJ = Path('/3250410011/root/rivermind-data/MutilModel_3485988898')
sys.path.insert(0, str(PROJ))

# RKNN-Toolkit2 (RK3588) 支持度参考：这些算子风险较高或需回落 CPU
RISK_OPS = {
    'Softmax': 'NPU 支持但低效，DFL 头大量使用会造成延迟抖动',
    'TopK': 'RKNPU 支持有限，NMS 后处理典型瓶颈',
    'NonMaxSuppression': 'NPU 不支持，必须回落 CPU',
    'GridSample': 'RKNPU 对 GridSample 支持受限（可变形卷积相关）',
    'Erf': 'GELU 分解算子，通常回落',
    'Gelu': '通常回落 CPU',
    'Mod': '部分版本不支持',
    'Trilu': '部分版本不支持',
}


def export_onnx(kind: str, out: Path, imgsz: int = 640):
    import torch
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    if kind == 'yolo26':
        cfg_yaml = PROJ / 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-early.yaml'
    else:
        cfg_yaml = PROJ / 'ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-early.yaml'

    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(cfg=str(cfg_yaml), ch=6, nc=1, verbose=False)
    model.eval()
    model.fuse() if hasattr(model, 'fuse') else None

    dummy = torch.zeros(1, 6, imgsz, imgsz)
    for p in model.parameters():
        try:
            p.requires_grad_(False)
        except RuntimeError:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        opset_version=12,
        input_names=['images'],
        output_names=['output0'] if not getattr(model, 'end2end', False) else ['output0'],
        do_constant_folding=True,
        dynamic_axes=None,
    )
    return model, dummy


def op_inventory(onnx_path: Path):
    import onnx
    m = onnx.load(str(onnx_path))
    cnt = collections.Counter(n.op_type for n in m.graph.node)
    inp = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outp = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    return cnt, inp, outp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['onnx', 'rknn'])
    ap.add_argument('kind', nargs='?', default='yolo26', choices=['yolo26', 'yolo11'])
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    tag = f'{args.kind}_mm_early'
    out = PROJ / '_d2p1' / 'deploy' / f'{tag}.onnx'

    if args.stage == 'onnx':
        model, dummy = export_onnx(args.kind, out, args.imgsz)
        print(f'[EXPORT] {out}  ({out.stat().st_size/1e6:.2f} MB)')
        cnt, inp, outp = op_inventory(out)
        print(f'[INPUT ] {inp}')
        print(f'[OUTPUT] {outp}')
        print(f'[OPS   ] 共 {len(cnt)} 种 / {sum(cnt.values())} 个节点')
        for op, c in cnt.most_common():
            flag = f'  <== RISK: {RISK_OPS[op]}' if op in RISK_OPS else ''
            print(f'  {op:28} x {c:4d}{flag}')

        # 统计风险算子总数
        risky = {op: cnt[op] for op in RISK_OPS if op in cnt}
        print(f'[RISK  ] {risky if risky else "无风险算子"}')

        # s 版改名后再跑一次
        if args.kind == 'yolo26':
            cfg = out.with_suffix('.json')
        else:
            cfg = out.with_suffix('.json')
        cfg.write_text(json.dumps({
            'kind': args.kind, 'imgsz': args.imgsz, 'onnx': str(out),
            'n_nodes': sum(cnt.values()), 'n_op_types': len(cnt),
            'ops': dict(cnt), 'risky': risky,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[JSON  ] {cfg}')

    elif args.stage == 'rknn':
        from rknn.api import RKNN
        rknn_out = out.with_suffix('.rknn')
        r = RKNN(verbose=False)
        r.config(
            mean_values=[[0, 0, 0, 0, 0, 0]],
            std_values=[[255, 255, 255, 255, 255, 255]],
            target_platform='rk3588',
            quantized_dtype='w8a8',
            optimization_level=3,
        )
        if r.load_onnx(model=str(out)) != 0:
            print('[RKNN] load_onnx 失败'); sys.exit(2)
        print('[RKNN] load_onnx OK，开始 build（未量化）')
        if r.build(do_quantization=False) != 0:
            print('[RKNN] build 失败（未量化）'); sys.exit(3)
        if r.export_rknn(str(rknn_out)) != 0:
            print('[RKNN] export 失败'); sys.exit(4)
        print(f'[RKNN] OK -> {rknn_out} ({rknn_out.stat().st_size/1e6:.2f} MB)')
        r.release()


if __name__ == '__main__':
    main()
