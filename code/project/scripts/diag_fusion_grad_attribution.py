# -*- coding: utf-8 -*-
"""融合头梯度归因诊断（C1「损失无关性」的第二种独立度量）。

问题
----
D0/C4/D2-P1b 已证明：**训练损失对模态依赖不可见**
    cls_fuse ≈ cls_fuse_ir（差 0.5%）
    但掩码探针 fuse=0.0007 vs fuse_ir=0.6485（差 926 倍）
第一层证据是「前向探针」（mAP），本脚本给出**反向证据**：梯度。

方法
----
对 fuse 头的检测损失单独反传（不更新权重），在**融合节点的两条上游支路**
上采集输出梯度：
    融合节点 46 <- 层 27 (RGB 支路) + 层 39 (IR 支路)
    融合节点 48 <- 层 30 (RGB 支路) + 层 42 (IR 支路)
    融合节点 50 <- 层 33 (RGB 支路) + 层 45 (IR 支路)
统计每个上游层的输出梯度范数，得到
    g_rgb = ||∂L_fuse / ∂(RGB 支路特征)||_2
    g_ir  = ||∂L_fuse / ∂(IR  支路特征)||_2
以及各自「有效梯度通道占比」。

为何 hook 上游支路而非融合节点的 list 输入
------------------------------------------
``register_full_backward_hook`` 对 ``list`` 输入的 grad_input 会把多输入展平/
打包，无法稳定区分 rgb_half 与 ir_half。改为 hook 上游两条支路的输出，天然
按模态分开，且语义明确：这就是「梯度有多少流回 RGB 编码器 / IR 编码器」。

判读
----
若 g_rgb / (g_rgb + g_ir) 接近 1 -> **优化信号把全部梯度压在 RGB 支路**：
即使前向特征里 IR 含独立信息（掩码探针 fuse_ir=0.648 证明），反向梯度也
根本不奖励 fuse 头去读它。这把「损失无关性」从「loss 数值相同」升级为
「梯度流向单一」的机制解释。

用法
----
    python scripts/diag_fusion_grad_attribution.py \
        --weights ResTest/v2_C4_deform_s0/weights/best.pt \
        --data /root/rivermind-data/uav_data_v2/data.yaml \
        --batches 20 --out _d2p1/grad_attr_C4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# 融合节点 -> (RGB 上游层, IR 上游层)
FUSION_PAIRS = {46: (27, 39), 48: (30, 42), 50: (33, 45)}


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "train_full.py").exists() and (p / "ultralytics").is_dir():
            return p
    for p in [start, *start.parents]:
        if not p.is_dir():
            continue
        for sub in p.iterdir():
            if sub.is_dir() and (sub / "train_full.py").exists() and (sub / "ultralytics").is_dir():
                return sub
    raise FileNotFoundError("未找到仓库根（需含 train_full.py 与 ultralytics/）。")


def main() -> int:
    here = Path(__file__).resolve()
    repo = find_repo_root(here.parent)
    sys.path.insert(0, str(repo))

    ap = argparse.ArgumentParser(description="融合头梯度归因诊断")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batches", type=int, default=20, help="采集批数")
    ap.add_argument("--branches", default="fuse,fuse_ir,rgb",
                    help="逗号分隔：分别对哪些分支的损失单独反传并统计梯度（用于对照）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLOMultiModalImageDataset
    from ultralytics.nn.mm.dual_loss import DualHeadDetectionLoss
    from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator
    from ultralytics.utils.patches import torch_load

    import pathlib

    import yaml

    data_path = Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_path.parent)
    val_sub = str(cfg.get("val", "images/val"))

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("融合头梯度归因诊断")
    print("=" * 78)
    print(f"  权重: {args.weights}")
    print(f"  设备: {device}  采集批数: {args.batches}")

    orig_winpath = getattr(pathlib, "WindowsPath", None)
    if hasattr(pathlib, "WindowsPath"):
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        try:
            ckpt = torch_load(str(args.weights), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch_load(str(args.weights), map_location="cpu")
    finally:
        if orig_winpath is not None:
            pathlib.WindowsPath = orig_winpath

    model = ckpt.get("model") or ckpt.get("ema")
    if model is None or not hasattr(model, "model"):
        print("[错误] 无法从 ckpt 取到检测模型")
        return 1
    print(f"  ckpt epoch: {ckpt.get('epoch')}  末层: {type(model.model[-1]).__name__}")
    model = model.to(device).float()
    inner = model.model

    up_layers = sorted({li for pair in FUSION_PAIRS.values() for li in pair})
    print("\n  上游支路校验:")
    for node, (lr, lx) in FUSION_PAIRS.items():
        print(f"    融合节点 L{node}  <- RGB L{lr} ({type(inner[lr]).__name__})"
              f" + IR L{lx} ({type(inner[lx]).__name__})")

    # ---- 数据 ----
    hyp = get_cfg()
    for k, v in (("mosaic", 0.0), ("mixup", 0.0), ("cutmix", 0.0), ("copy_paste", 0.0),
                 ("mm_rgb_albu_enable", False), ("mm_x_non_geo_enable", False)):
        setattr(hyp, k, v)
    for _k in ("cache", "cache_ir"):
        cfg.pop(_k, None)

    ds = YOLOMultiModalImageDataset(
        img_path=str(root / "images" / val_sub.split("/")[-1]),
        data=cfg, x_modality="ir", imgsz=args.imgsz,
        augment=False, hyp=hyp, rect=False, cache=False,
    )
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)
    print(f"  验证集样本数: {len(ds)}")

    # ---- 梯度钩子（hook 上游支路输出） ----
    # 每个待测分支单独一轮：每轮清空 stats，只对当前分支反传。
    stat = {li: {"g": [], "ch_frac": []} for li in up_layers}
    handles = []

    def reset_stat():
        for li in stat:
            stat[li]["g"].clear()
            stat[li]["ch_frac"].clear()

    def make_hook(layer_idx: int):
        def hook(module, grad_input, grad_output):
            go = grad_output
            if isinstance(go, (list, tuple)):
                go = next((t for t in go if torch.is_tensor(t) and t.dim() == 4), None)
            if not torch.is_tensor(go) or go.dim() != 4:
                return
            with torch.no_grad():
                g = go.detach().float()
                stat[layer_idx]["g"].append(float(g.pow(2).sum().sqrt()))
                per_ch = g.pow(2).sum(dim=(0, 2, 3)).sqrt()
                mx = float(per_ch.max())
                thr = mx * 0.01 if mx > 0 else 0.0
                stat[layer_idx]["ch_frac"].append(float((per_ch > thr).float().mean()))
        return hook

    for li in up_layers:
        handles.append(inner[li].register_full_backward_hook(make_hook(li)))

    vargs = {
        "conf": 0.001, "iou": 0.7, "max_det": 300, "single_cls": False,
        "agnostic_nms": False, "task": "detect", "split": "val", "half": False,
        "plots": False, "save_json": False, "val": True, "verbose": False,
        "data": str(data_path), "mode": "val", "imgsz": args.imgsz,
        "batch": args.batch, "workers": args.workers, "device": args.device,
    }
    v = DualHeadDetectionValidator(dataloader=loader, save_dir=None, args=vargs)
    v.data, v.device = cfg, device
    v.args.plots = False

    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    # ckpt 里 model.args 可能是 dict（pyTorch pickle 后丢失 SimpleNamespace）；
    # DualHeadDetectionLoss 需要 model.args.box 这类属性，故在此补齐。
    from ultralytics.cfg import get_cfg as _get_cfg
    if not hasattr(model, "args") or isinstance(getattr(model, "args", None), dict) or model.args is None:
        model.args = _get_cfg()
    criterion = DualHeadDetectionLoss(model)
    print(f"  criterion 分支: {criterion.branch_names}")

    # ckpt 加载后参数默认 requires_grad=False；本脚本需要反向梯度，故显式打开。
    for p in model.parameters():
        p.requires_grad_(True)

    branch_list = [b.strip() for b in args.branches.split(",") if b.strip()]

    print("\n" + "=" * 78)
    print(f"采集梯度：分支 {branch_list}，各 {args.batches} 批")
    print("=" * 78)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    results = {"weights": str(args.weights), "ckpt_epoch": ckpt.get("epoch"),
               "n_batches": args.batches, "pairs": FUSION_PAIRS, "by_branch": {}}

    for target in branch_list:
        if target not in criterion.branch_names:
            print(f"  [跳过] 分支 {target} 不在 {criterion.branch_names}")
            continue
        reset_stat()
        n = 0
        for bi, batch in enumerate(loader):
            batch = v.preprocess(batch)
            for k in ("bboxes", "cls", "batch_idx", "bboxes_ir", "cls_ir", "batch_idx_ir",
                      "valid_rgb", "valid_ir", "frame_kind"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            preds = model(batch["img"])
            if not isinstance(preds, dict):
                print("[错误] 模型输出非 dict，该权重不是双头架构")
                return 1

            model.zero_grad(set_to_none=True)
            old_w = dict(criterion.branch_weights)
            for name in criterion.branch_names:
                criterion.branch_weights[name] = 1.0 if name == target else 0.0
            loss, _ = criterion(preds, batch)
            criterion.branch_weights = old_w
            loss.sum().backward()

            n += 1
            if bi % 5 == 0:
                print(f"    [{target}] 已处理 {bi + 1} 批  loss={float(loss.sum()):.4f}", end="\r")
            if n >= args.batches:
                break
        print()

        # ---- 汇总该分支 ----
        print("\n" + "=" * 78)
        print(f"分支 {target}：梯度归因（损失对融合颈上游支路的梯度）")
        print("=" * 78)
        print(f"{'节点/支路':<26}{'g':>16}{'梯度占比':>12}{'通道激活':>12}")
        print("-" * 78)
        entry = {"pairs": {}}
        sum_rgb, sum_ir = 0.0, 0.0
        for node, (lr, lx) in FUSION_PAIRS.items():
            gr, gi = mean(stat[lr]["g"]), mean(stat[lx]["g"])
            fr, fi = mean(stat[lr]["ch_frac"]), mean(stat[lx]["ch_frac"])
            share = gr / (gr + gi) if (gr + gi) > 0 else float("nan")
            sum_rgb += gr
            sum_ir += gi
            print(f"L{node} RGB(L{lr}){'':<8}{gr:>16.6f}{share:>12.4f}{fr:>12.4f}")
            print(f"L{node} IR (L{lx}){'':<8}{gi:>16.6f}{1 - share:>12.4f}{fi:>12.4f}")
            entry["pairs"][f"L{node}"] = {
                "rgb_upstream_layer": lr, "ir_upstream_layer": lx,
                "g_rgb_mean": gr, "g_ir_mean": gi, "rgb_grad_share": share,
                "rgb_channel_active_frac": fr, "ir_channel_active_frac": fi,
            }
        if sum_rgb + sum_ir > 0:
            overall = sum_rgb / (sum_rgb + sum_ir)
            entry["rgb_grad_share_overall"] = overall
            entry["g_rgb_sum"] = sum_rgb
            entry["g_ir_sum"] = sum_ir
            print("-" * 78)
            print(f"{'合计':<26}{sum_rgb:>16.6f}{overall:>12.4f}")
            print(f"{'(IR)':<26}{sum_ir:>16.6f}{1 - overall:>12.4f}")
        results["by_branch"][target] = entry

    # ---- 判读（只看 fuse 头） ----
    fuse_entry = results["by_branch"].get("fuse")
    if fuse_entry and "rgb_grad_share_overall" in fuse_entry:
        overall = fuse_entry["rgb_grad_share_overall"]
        print("\n" + "=" * 78)
        print("判读（fuse 头）")
        print("=" * 78)
        print(f"  fuse 头损失的梯度中，RGB 支路占比 = {100 * overall:.1f}%")
        if overall > 0.9:
            print("  -> 优化信号**几乎全部**压在 RGB 支路：即使前向特征里 IR 含独立信息")
            print("     （掩码探针 fuse_ir≈0.648 证明），反向梯度也不驱动 fuse 头去读它。")
            print("     这是「训练损失对模态依赖不可见」的机制层解释（第二种独立度量）。")
        elif overall > 0.7:
            print("  -> RGB 支路明显主导。对照同权重下 rgb 头 / fuse_ir 头的占比可判断")
            print("     这是否为「融合头特有」的偏置，而非全局结构效应。")
        else:
            print("  -> 两路梯度相对均衡，未复现「梯度单边」假设。")

    out = Path(args.out) if args.out else data_path.parent / "grad_attr.json"
    for h in handles:
        h.remove()
    try:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  结果已保存: {out}")
    except (PermissionError, OSError) as e:
        print(f"\n  [提示] 无法写入 {out}（{type(e).__name__}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
