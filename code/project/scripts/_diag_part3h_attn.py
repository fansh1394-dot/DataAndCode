# -*- coding: utf-8 -*-
"""验证 part3h 中 ChannelPartitionFusion 的 attn 分支是否空转。

背景
----
`max_offset=0` 时 K 个采样点完全重合，`grid_sample` 对每个点取到同一位置，
softmax(attn) 对聚合结果的影响退化为常数加权（各点特征相同，权重怎么变结果都一样），
于是 ∂L/∂attn = 0 —— attn 头梯度恒为 0，是纯空转参数。

若该结论成立，part3h 的负面结果就只能归因于「通道硬切分」本身，
而不能解读为「通道分割 + 注意力」这一整体设计无效。

做法
----
对 fuse 头的检测损失反传（不更新权重），按融合模块内部子模块统计
参数梯度的范数占比：rgb_proj / ir_proj / offset / attn / gate / out。

判读
----
attn 占比 ≈ 0 且 |grad| 恒为 0  -> 空转成立（结构性，非训练不足）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SUBMODULES = ("rgb_proj", "ir_proj", "offset", "attn", "gate", "out")
FUSION_LAYERS = (46, 48, 50)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--python-path", default="/3250410011/root/rivermind-data/MutilModel_3485988898")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="默认 CPU：避免与外部租户抢显存")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--out", default="logs/part3h_attn_probe.json")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, args.python_path)

    import yaml
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
    from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator
    from ultralytics.nn.mm.partition_fusion import ChannelPartitionFusion
    from ultralytics.utils.patches import torch_load

    data_path = Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_path.parent)
    val_sub = str(cfg.get("val", "images/val"))

    ckpt = torch_load(str(args.weights), map_location="cpu", weights_only=False)
    model = ckpt.get("model") or ckpt.get("ema")
    # --device 接受 "cpu" 或 GPU 序号（如 "0"），序号需补成 cuda:N
    dev = str(args.device).strip()
    device = torch.device(f"cuda:{dev}" if dev.isdigit() else dev)
    model = model.to(device).float()
    inner = model.model

    mods = {li: inner[li] for li in FUSION_LAYERS}
    for li, m in mods.items():
        if not isinstance(m, ChannelPartitionFusion):
            print(f"[错误] 层 {li} 不是 ChannelPartitionFusion，实为 {type(m).__name__}")
            return 1
        print(f"  L{li}: split=({m.n_rgb},{m.n_ir},{m.n_mix}) "
              f"K={m.num_points} max_offset={m.max_offset}")

    # ---- 构建与训练同口径的 val loader ----
    data = check_det_dataset(str(data_path))
    train_args = ckpt.get("train_args") or {}
    vargs = get_cfg(overrides={
        **{k: v for k, v in (train_args.items() if isinstance(train_args, dict) else [])
           if not isinstance(v, (dict, list)) or k == "data"},
        "task": "detect", "mode": "val", "data": str(data_path), "split": "val",
        "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
        "device": args.device, "conf": 0.001, "iou": 0.7, "max_det": 300,
        "half": False, "plots": False, "save_json": False, "save_txt": False,
        "verbose": False, "gt_label_source": "auto",
    })
    builder = MultiModalDetectionValidator(args=vars(vargs))
    builder.data = data
    # stride 挂在末层检测头（DualHeadFusionDetectV26）上，不在 nn.Sequential 外壳
    builder.stride = int(inner[-1].stride.max())
    dataset = builder.build_dataset(data.get("val") or val_sub, mode="val", batch=args.batch)
    loader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)
    print(f"[loader] n={len(dataset)}")

    v = DualHeadDetectionValidator(args=vars(vargs))
    v.data = data
    v.device = device

    # model.loss() 内部读 self.args.box / cls / dfl 等属性（dual_loss.py L279），
    # 必须挂 namespace；ckpt 序列化下来的 args 可能是 dict，会报 'dict' object has no attribute 'box'。
    model.args = vargs

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    stat = {li: {n: {"sum": 0.0, "n": 0} for n in SUBMODULES} for li in FUSION_LAYERS}
    done = 0
    for batch in loader:
        batch = v.preprocess(batch)
        model.zero_grad(set_to_none=True)
        preds = model(batch["img"])
        loss, loss_items = model.loss(batch, preds)
        # DualHeadFusionCriterion 返回的是向量（各分支损失），需先归约为标量
        scalar = loss.sum() if loss.dim() > 0 else loss
        if not torch.isfinite(scalar):
            continue
        scalar.backward()

        for li, mod in mods.items():
            for name in SUBMODULES:
                sub = getattr(mod, name, None)
                if sub is None:
                    continue
                tot = 0.0
                for p in sub.parameters():
                    if p.grad is not None:
                        tot += float(p.grad.detach().float().norm())
                stat[li][name]["sum"] += tot
                stat[li][name]["n"] += 1
        done += 1
        print(f"  批次 {done}/{args.batches}  loss={float(scalar):.4f}", flush=True)
        if done >= args.batches:
            break

    print("\n" + "=" * 78)
    print("ChannelPartitionFusion 各子模块梯度范数（fuse 检测损失反传）")
    print("=" * 78)
    print(f"{'层':<6}{'子模块':<12}{'|grad| 均值':>14}{'占模块比例':>14}")
    print("-" * 78)
    payload = {}
    for li in FUSION_LAYERS:
        total = sum(s["sum"] for s in stat[li].values() if s["n"])
        payload[f"L{li}"] = {}
        for name in SUBMODULES:
            s = stat[li][name]
            if not s["n"]:
                continue
            mean = s["sum"] / s["n"]
            share = mean / total if total > 0 else 0.0
            payload[f"L{li}"][name] = {"grad_norm": mean, "share": share}
            print(f"L{li:<5}{name:<12}{mean:>14.6e}{share * 100:>13.2f}%")
        print("-" * 78)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n写入 {args.out}")

    attn_all_zero = all(
        v for li in FUSION_LAYERS for k, val in payload[f"L{li}"].items()
        if k == "attn" for v in [val["grad_norm"] == 0.0]
    )
    print(f"\n判读：attn 分支梯度{'全为 0 -> 空转成立' if attn_all_zero else '非零 -> 未空转'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
