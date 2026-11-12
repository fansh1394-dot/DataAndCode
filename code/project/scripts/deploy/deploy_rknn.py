# -*- coding: utf-8 -*-
"""RK3588 端侧部署工具链（清单 T6.1）。

四段流水线：ONNX 导出 → INT8/FP16 转换 → 精度对齐 → 基准测试。

设计要点
--------
1. **YOLO26 是 NMS-free**：one2one 分支直接输出 [N,6] 解码框，
   导出时不需要 NonMaxSuppression 算子，省掉 RKNPU 最不擅长的后处理。
2. **检测头敏感层保 FP16**：RKNN 的 `hybrid_quantization` 让指定层跳过 INT8，
   避免 reg/cls 分支量化误差放大（T6.4 的量化精度保护）。
3. **双模态输入是 6 通道**：RKNN 的 mean_values/std_values 必须给 6 个通道，
   顺序与训练一致（RGB 3ch + IR 3ch）。
4. **校准集分层抽样**：覆盖昼夜/背景类型，否则 INT8 量化在夜间会明显掉点。

用法（服务器，需 PYTHONPATH 指向 deploy_env）::

    export PYTHONPATH=/root/rivermind-data/deploy_env
    PY=/opt/conda/envs/rivermind/bin/python

    # 1) 导出 ONNX
    $PY scripts/deploy_rknn.py export \
        --weights YOLO26/ResTest/v26_E4_deform_s0/weights/best.pt \
        --head fuse --imgsz 640

    # 2) 转 RKNN（fp16 / int8 可选）
    $PY scripts/deploy_rknn.py convert \
        --onnx YOLO26/deploy/v26_E4_fuse.onnx --precision fp16

    # 3) 精度对齐（ONNX vs RKNN 在模拟器上的余弦相似度）
    $PY scripts/deploy_rknn.py verify \
        --onnx ...onnx --rknn ...rknn --n 50

    # 4) 基准（仅 RK3588 板端可用）
    $PY scripts/deploy_rknn.py bench --rknn ...rknn --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path('/3250410011/root/rivermind-data/MutilModel_3485988898')
sys.path.insert(0, str(PROJ))

DATA = Path('/root/rivermind-data/uav_data_v2')
OUT_DIR = PROJ / 'YOLO26' / 'deploy'

# 导出时用的 head：YOLO26 四头模型需指定取哪一路输出
HEAD_CHOICES = ('rgb', 'ir', 'fuse', 'fuse_ir')


# ======================================================================
# 1. ONNX 导出
# ======================================================================
def export_onnx(weights: str, head: str, imgsz: int = 640, opset: int = 12,
                out: Path | None = None) -> Path:
    """把训练好的 YOLO26 权重导出为单头 ONNX（6 通道输入，NMS-free 输出）。

    Args:
        weights: .pt 路径（相对 PROJ 或绝对）
        head: 取哪一路检测头，决定部署用途
              - ``rgb``  : 可见光单模态（只喂 RGB，IR 通道补零）
              - ``ir``   : 红外单模态（只喂 IR，RGB 通道补零）
              - ``fuse`` : 双模态融合（6 通道，最常用）
              - ``fuse_ir``: IR 坐标系的融合头（一般不部署）
        imgsz: 输入尺寸
        opset: ONNX opset（12 对 RKNN 2.3 兼容性最好）
        out: 输出路径，默认 ``YOLO26/deploy/<run>_<head>.onnx``

    返回: 导出的 ONNX 路径

    做法要点：
      - 双头/四头模型的 forward 返回 dict，导出前需要包一层只取指定头的
        Module，否则 torch.onnx.export 会因为输出是 dict 而失败。
      - 必须 ``model.eval()`` + 关闭梯度，否则导出图里带 training 分支。
    """
    raise NotImplementedError('TODO(T6.1): 实现单头包装 + torch.onnx.export')

    # 参考实现（取消 raise 后可用，需按实际 forward 签名调整）：
    #
    # import torch
    # from ultralytics.nn.tasks import attempt_load_one_weight
    #
    # wp = Path(weights) if Path(weights).is_absolute() else PROJ / weights
    # if not wp.exists():
    #     raise FileNotFoundError(wp)
    # model, _ = attempt_load_one_weight(str(wp))
    # model.eval()
    # for p in model.parameters():
    #     p.requires_grad_(False)
    # if hasattr(model, 'fuse'):
    #     model.fuse()
    #
    # class _HeadOnly(torch.nn.Module):
    #     """只输出指定头的推理结果，供 ONNX 导出。"""
    #     def __init__(self, m, head):
    #         super().__init__()
    #         self.m = m
    #         self.head = head
    #
    #     def forward(self, x):
    #         y = self.m(x)
    #         if isinstance(y, dict):
    #             return y[self.head]
    #         return y
    #
    # wrapper = _HeadOnly(model, head).eval()
    # dummy = torch.zeros(1, 6, imgsz, imgsz)
    # dst = out or (OUT_DIR / f'{wp.parent.parent.name}_{head}.onnx')
    # dst.parent.mkdir(parents=True, exist_ok=True)
    # torch.onnx.export(
    #     wrapper, dummy, str(dst), opset_version=opset,
    #     input_names=['images'], output_names=['pred'],
    #     do_constant_folding=True, dynamic_axes=None,
    # )
    # return dst


# ======================================================================
# 2. RKNN 转换
# ======================================================================
def build_calibration_set(n: int = 300, imgsz: int = 640,
                          out_dir: Path | None = None) -> Path:
    """从 train split 分层抽 n 帧，生成 INT8 校准用的 .npy 列表。

    分层维度：frame_kind（both / only_rgb / only_ir）+ IR 平均亮度分位（昼夜）。
    直接均匀随机抽会让夜间帧占比过低，量化后在夜间掉点。

    返回: npy 列表文件路径（每行一个 .npy 绝对路径，RKNN 的
    ``load_calibration_dataset`` 需要这个格式）。
    """
    raise NotImplementedError('TODO(T6.1): 实现分层抽样 + 6 通道 npy 落盘')


# ======================================================================
# 3. 转换
# ======================================================================
def convert_rknn(onnx_path: Path, precision: str = 'fp16',
                 calib_list: Path | None = None,
                 hybrid_layers: tuple[str, ...] = (),
                 out: Path | None = None) -> Path:
    """ONNX → RKNN。

    Args:
        precision: ``fp16``（默认，精度无损）或 ``int8``（需 calib_list）
        calib_list: INT8 校准列表（``build_calibration_set`` 的产物）
        hybrid_layers: 这些层名保持 FP16（RKNN 的混合量化）。
                       INT8 模式下建议放检测头的 Conv / DFL 投影层。

    返回: .rknn 路径
    """
    raise NotImplementedError('TODO(T6.1): 实现 rknn.api.RKNN 流程')


# ======================================================================
# 4. 精度对齐
# ======================================================================
def verify_alignment(onnx_path: Path, rknn_path: Path, n: int = 50,
                     imgsz: int = 640, tol: float = 0.99) -> dict:
    """ONNX 与 RKNN 在同一批输入上的输出余弦相似度。

    这是 T6.4「量化精度保护」的量化证据：逐层或逐输出比对，
    余弦相似度 < ``tol`` 的输出说明该路径量化损伤过大，需要回退 FP16。

    返回: {"cos": [...], "min": float, "pass": bool}
    """
    raise NotImplementedError('TODO(T6.1): onnxruntime + rknn 模拟器双跑比对')


# ======================================================================
# 5. 板端基准
# ======================================================================
def bench_rknn(rknn_path: Path, n: int = 200) -> dict:
    """RK3588 板端：推理延迟 / NPU 占用 / 内存。

    必须在**真机**上跑（``rknnlite`` 只在板端可用）。
    桌面 GPU 的数字不能代替（清单 T6.3 硬约束）。

    返回: {"fps": float, "latency_ms": {"mean","p50","p95"}, "mem_mb": float}
    """
    raise NotImplementedError('TODO(T6.1): rknnlite 三核并行 + 计时')


# ======================================================================
# CLI
# ======================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='RK3588 部署工具链（T6.1）')
    sub = ap.add_subparsers(dest='stage', required=True)

    p1 = sub.add_parser('export', help='导出 ONNX')
    p1.add_argument('--weights', required=True, help='.pt 路径（相对项目根或绝对）')
    p1.add_argument('--head', default='fuse', choices=HEAD_CHOICES)
    p1.add_argument('--imgsz', type=int, default=640)
    p1.add_argument('--opset', type=int, default=12)
    p1.add_argument('--out', default=None)

    p2 = sub.add_parser('convert', help='ONNX → RKNN')
    p2.add_argument('--onnx', required=True)
    p2.add_argument('--precision', default='fp16', choices=['fp16', 'int8'])
    p2.add_argument('--calib', default=None, help='INT8 校准列表（.txt）')
    p2.add_argument('--hybrid', nargs='*', default=[],
                    help='保持 FP16 的层名（INT8 模式）')
    p2.add_argument('--out', default=None)

    p3 = sub.add_parser('calib', help='生成 INT8 校准集')
    p3.add_argument('--n', type=int, default=300)
    p3.add_argument('--imgsz', type=int, default=640)

    p4 = sub.add_parser('verify', help='ONNX vs RKNN 精度对齐')
    p4.add_argument('--onnx', required=True)
    p4.add_argument('--rknn', required=True)
    p4.add_argument('--n', type=int, default=50)
    p4.add_argument('--tol', type=float, default=0.99)

    p5 = sub.add_parser('bench', help='板端基准（需真机）')
    p5.add_argument('--rknn', required=True)
    p5.add_argument('--n', type=int, default=200)

    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if args.stage == 'export':
            dst = export_onnx(args.weights, args.head, args.imgsz, args.opset,
                              Path(args.out) if args.out else None)
            print(f'[EXPORT] {dst} ({dst.stat().st_size / 1e6:.2f} MB)')
        elif args.stage == 'convert':
            dst = convert_rknn(Path(args.onnx), args.precision,
                               Path(args.calib) if args.calib else None,
                               tuple(args.hybrid),
                               Path(args.out) if args.out else None)
            print(f'[RKNN] {dst} ({dst.stat().st_size / 1e6:.2f} MB)')
        elif args.stage == 'calib':
            dst = build_calibration_set(args.n, args.imgsz)
            print(f'[CALIB] {dst}')
        elif args.stage == 'verify':
            r = verify_alignment(Path(args.onnx), Path(args.rknn), args.n, tol=args.tol)
            print(json.dumps(r, ensure_ascii=False, indent=2))
            if not r['pass']:
                print('[FAIL] 余弦相似度低于阈值，建议相关层回退 FP16', file=sys.stderr)
                return 1
        elif args.stage == 'bench':
            r = bench_rknn(Path(args.rknn), args.n)
            print(json.dumps(r, ensure_ascii=False, indent=2))
    except NotImplementedError as e:
        print(f'[未实现] {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
