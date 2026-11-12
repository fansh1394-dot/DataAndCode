# -*- coding: utf-8 -*-
"""融合颈逐通道消融（C1 的第三种独立度量）。

问题
----
掩码探针（``diag_fusion_attribution.py``）把 RGB/IR **整半边**置零，只能回答
「有没有依赖」。逐通道消融回答更细的问题：

  * fuse 头对 RGB 半边的依赖是**少数通道承载全部信息**（可定位、可裁剪），
    还是**全通道均摊**（结构性纠缠、不可裁剪）？
  * 对称地，IR 半边有多少通道是 fuse 头真正在读的？

与梯度归因互补：梯度说明优化信号流向，消融说明前向信息通路的冗余度。

方法
----
按「该通道输出激活的 L2 幅值」降序排列融合节点上游支路的 C 个通道，
分档置零 top-k%（k = 0/10/25/50/100），每次只改该支路、另一支路保持原值，
记录 fuse 头 mAP50-95。置零通过 forward hook 改**该上游层输出**实现，
融合算子（含 offset/reliability 子网）拿到的就是被置零后的张量，
故这也是「真实消融」口径（与训练期 ``mm_fusion_rgb_mask`` 同位置）。

用法
----
    python scripts/diag_channel_ablation.py \
        --weights ResTest/v2_C4_deform_s0/weights/best.pt \
        --data /root/rivermind-data/uav_data_v2/data.yaml \
        --batches 40 --out _d2p1/chan_ablation_C4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

FUSION_PAIRS = {46: (27, 39), 48: (30, 42), 50: (33, 45)}
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)


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

    ap = argparse.ArgumentParser(description="融合颈逐通道消融")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batches", type=int, default=40, help="每档评测批数")
    ap.add_argument("--only-node", type=int, default=None,
                    help="只消融指定融合节点（默认全部；UAV 小目标场景建议只测 P3 节点 46，"
                         "因 P4/P5 对 mAP50-95 贡献近乎为零、会稀释信号）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLOMultiModalImageDataset
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
    print("融合颈逐通道消融")
    print("=" * 78)
    print(f"  权重: {args.weights}")
    print(f"  设备: {device}  每档批数: {args.batches}")

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
    model = model.to(device).float().eval()
    inner = model.model

    up_layers = sorted({l for pair in FUSION_PAIRS.values() for l in pair})
    for node, (lr, lx) in FUSION_PAIRS.items():
        print(f"    融合节点 L{node}  <- RGB L{lr} ({type(inner[lr]).__name__})"
              f" + IR L{lx} ({type(inner[lx]).__name__})")

    hyp = get_cfg()
    for k, v_ in (("mosaic", 0.0), ("mixup", 0.0), ("cutmix", 0.0), ("copy_paste", 0.0),
                  ("mm_rgb_albu_enable", False), ("mm_x_non_geo_enable", False)):
        setattr(hyp, k, v_)
    for _k in ("cache", "cache_ir"):
        cfg.pop(_k, None)

    ds = YOLOMultiModalImageDataset(
        img_path=str(root / "images" / val_sub.split("/")[-1]),
        data=cfg, x_modality="ir", imgsz=args.imgsz,
        augment=False, hyp=hyp, rect=False, cache=False,
    )
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)
    print(f"  验证集样本数: {len(ds)}")

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

    # 消融状态：层 -> 要置零的通道索引（None 表示不消融）
    ABL = {li: None for li in up_layers}

    def make_ablate_hook(layer_idx: int):
        def hook(module, inputs, output):
            ch = ABL[layer_idx]
            if ch is None or not torch.is_tensor(output) or output.dim() != 4:
                return output
            y = output.clone()
            y[:, ch] = 0.0
            return y
        return hook

    handles = [inner[li].register_forward_hook(make_ablate_hook(li)) for li in up_layers]

    def run_eval(cache_batches):
        """在**固定的缓存批次**上评测，返回各分支总体 mAP50-95。

        cache_batches 是预先取好的 batch 列表（已 preprocess），故所有消融档位
        评的是**完全相同的样本** —— 这是 Δ 可比的前提。
        （build_dataloader 返回 InfiniteDataLoader，重新 enumerate 会落在
        随机相位，直接导致档间不可比；早期版本即因此噪声过大。）

        注意：cache 常驻 CPU（图像一个 batch 就 ~9 MB，150 批留在显存会直接 OOM），
        每个 batch 评测前再搬到设备。
        """
        v.init_metrics(model)
        v.jdict = []
        for batch_cpu in cache_batches:
            batch = {}
            for k, val in batch_cpu.items():
                batch[k] = val.to(device, non_blocking=True) if torch.is_tensor(val) else val
            with torch.no_grad():
                preds_raw = model(batch["img"])
            if not isinstance(preds_raw, dict):
                return None
            for name in v.branch_names:
                if name not in preds_raw:
                    continue
                preds = v.postprocess(v._unwrap_branch_preds(preds_raw[name]))
                v._update_branch_metrics(name, preds, batch)
        overall = {}
        for k, val in v._branch_stats().items():
            if k.endswith("mAP50-95(B)"):
                overall[k.split("/")[1]] = float(val)
        return overall

    def clear():
        for li in ABL:
            ABL[li] = None

    # ---- 预先缓存固定批次（CPU 侧；所有档位复用同一份数据） ----
    print("\n" + "=" * 78)
    print(f"缓存 {args.batches} 批固定样本到 CPU（所有档位复用，保证 Δ 可比）")
    print("=" * 78)
    clear()
    cache = []
    for bi, batch in enumerate(loader):
        batch = v.preprocess(batch)
        batch_cpu = {k: (val.detach().cpu() if torch.is_tensor(val) else val) for k, val in batch.items()}
        cache.append(batch_cpu)
        if (bi + 1) % 20 == 0:
            print(f"    已缓存 {bi + 1} 批 ...", end="\r")
        if len(cache) >= args.batches:
            break
    torch.cuda.empty_cache()
    print(f"\n  已缓存 {len(cache)} 批（{len(cache) * args.batch} 张图，驻留 CPU）")

    print("\n" + "=" * 78)
    print(f"基准（无消融，同一份 {len(cache)} 批）")
    print("=" * 78)
    clear()
    base = run_eval(cache)
    if base is None:
        print("[错误] 模型输出非 dict，该权重不是双头架构")
        return 1
    print("  " + "  ".join(f"{k}={v_:.4f}" for k, v_ in base.items()))

    results = {"weights": str(args.weights), "ckpt_epoch": ckpt.get("epoch"),
               "n_batches": args.batches, "baseline": base, "ablation": {}}

    for node, (lr, lx) in FUSION_PAIRS.items():
        if args.only_node is not None and node != args.only_node:
            continue
        for side_name, layer_idx in (("rgb_side", lr), ("ir_side", lx)):
            # 用一次干净前向采集该层输出激活的逐通道 L2 幅值，据此降序排序
            holder = {}

            def cap_hook(module, inputs, output, _h=holder):
                _h["a"] = output.detach().float().pow(2).sum(dim=(0, 2, 3)).sqrt().cpu().clone()

            hh = inner[layer_idx].register_forward_hook(cap_hook)
            clear()
            with torch.no_grad():
                for batch_cpu in cache:
                    img = batch_cpu["img"].to(device)
                    model(img)
            hh.remove()
            amp = holder["a"]
            order = torch.argsort(amp, descending=True)
            c = int(order.numel())

            print(f"\n{'=' * 78}")
            print(f"消融 {side_name}（上游层 L{layer_idx}，{c} 通道，服务节点 L{node}）")
            print(f"{'=' * 78}")
            print(f"{'置零':>8}{'通道数':>8}{'fuse':>10}{'rgb':>10}{'ir':>10}{'fuse_ir':>10}{'Δfuse':>10}")
            print("-" * 78)
            rows = {}
            # 0% 档 = 基准（复用，不重跑）
            print(f"{'0%':>8}{0:>8}{base.get('fuse', 0):>10.4f}{base.get('rgb', 0):>10.4f}"
                  f"{base.get('ir', 0):>10.4f}{base.get('fuse_ir', 0):>10.4f}{0.0:>+10.4f}")
            rows["0%"] = {"k": 0, "fuse": base.get("fuse"), "rgb": base.get("rgb"),
                          "ir": base.get("ir"), "fuse_ir": base.get("fuse_ir"), "delta_fuse": 0.0}
            for frac in FRACTIONS:
                k = max(1, int(round(c * frac)))
                clear()
                ABL[layer_idx] = order[:k]
                res = run_eval(cache)
                clear()
                if res is None:
                    continue
                dfuse = res.get("fuse", 0.0) - base.get("fuse", 0.0)
                print(f"{100 * frac:>7.0f}%{k:>8}{res.get('fuse', 0):>10.4f}{res.get('rgb', 0):>10.4f}"
                      f"{res.get('ir', 0):>10.4f}{res.get('fuse_ir', 0):>10.4f}{dfuse:>+10.4f}")
                rows[f"{int(frac * 100)}%"] = {
                    "k": k, "fuse": res.get("fuse"), "rgb": res.get("rgb"),
                    "ir": res.get("ir"), "fuse_ir": res.get("fuse_ir"), "delta_fuse": dfuse,
                }
            results["ablation"][f"L{node}/{side_name}"] = rows

    clear()
    for h in handles:
        h.remove()

    print("\n" + "=" * 78)
    print("判读")
    print("=" * 78)
    for key, rows in results["ablation"].items():
        if "10%" in rows and "100%" in rows:
            d5 = rows.get("5%", {}).get("delta_fuse", 0.0)
            d10 = rows["10%"]["delta_fuse"]
            d30 = rows.get("30%", {}).get("delta_fuse", 0.0)
            d50 = rows.get("50%", rows["100%"])["delta_fuse"]
            d100 = rows["100%"]["delta_fuse"]
            if d100 > -0.02:
                tag = "IR/该支路对 fuse 头无贡献（全通道置零亦不变）"
            elif d10 < -0.15:
                tag = "依赖高度集中（top-10% 通道即捷径）"
            elif d50 > -0.05 and d100 < -0.15:
                tag = "依赖全通道均摊（置零过半才显著下降）-> 结构性纠缠"
            else:
                tag = "渐进式、介于两者之间"
            print(f"  {key:<18} Δ5%={d5:+.4f} Δ10%={d10:+.4f} Δ30%={d30:+.4f} "
                  f"Δ50%={d50:+.4f} Δ100%={d100:+.4f}  -> {tag}")

    out = Path(args.out) if args.out else data_path.parent / "chan_ablation.json"
    try:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  结果已保存: {out}")
    except (PermissionError, OSError) as e:
        print(f"\n  [提示] 无法写入 {out}（{type(e).__name__}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
