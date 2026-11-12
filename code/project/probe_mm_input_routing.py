# -*- coding: utf-8 -*-
"""判别实验：双头模型的 IR 支路**实际**吃到的是 IR 图还是 RGB 图。

背景
----
异常事实：
  单模态 IR 基线（真 IR 图 + 同结构骨干）= 0.207/0.211
  双头 C4 的 ir 头（标称 IR 特征 + IR 标签）  = 0.6415
同一份 IR 内容，差 0.43。而 RGB 侧完全自洽（rgb 头 0.7052 ≈ RGB-only 0.7075）。
唯一能让 ir 头达到 0.64 的可能：它的输入**根本不是 IR 图**。

做法（零训练、只前向一个 batch 抓输入）
--------------------------------------
在 C4 的 RGB 支路首层 Conv（inner[0]）与 IR 支路首层 Conv（inner[11]）上挂
forward_pre_hook，直接取「喂进卷积的原始张量」，然后：
  1. 打印各张量形状
  2. 判断 inner[11] 的输入是否 == inner[0] 的输入（若是 -> IR 支路吃了 RGB）
  3. 与 dataloader 里 batch["img"] 的 RGB 段[0:3] / X 段[3:6] 做逐元素相关
     - 若 inner[11] 输入 ≈ batch X 段 -> 路由正确（IR 支路吃 IR）
     - 若 inner[11] 输入 ≈ batch RGB 段 -> 路由错误（IR 支路吃 RGB）

并对单模态 IR 基线做同样抓取，作为「真 IR」的参照。

用法：
  python probe_mm_input_routing.py --weights ResTest/v2_C4_deform_s0/weights/best.pt \
      --mode dual --data /root/rivermind-data/uav_data_v2/data.yaml
  python probe_mm_input_routing.py --weights ResTest/v2_E2_ironly_s0/weights/best.pt \
      --mode single_ir --data /root/rivermind-data/uav_data_v2/data.yaml
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path

import torch
import yaml


def find_root() -> Path:
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (p / "train_full.py").exists() and (p / "ultralytics").is_dir():
            return p
    return Path("/3250410011/root/rivermind-data/MutilModel_3485988898")


def load_model(weights: str):
    from ultralytics.utils.patches import torch_load
    ow = getattr(pathlib, "WindowsPath", None)
    if hasattr(pathlib, "WindowsPath"):
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        ck = torch_load(weights, map_location="cpu", weights_only=False)
    finally:
        if ow is not None:
            pathlib.WindowsPath = ow
    m = ck.get("model") or ck.get("ema")
    return m, ck


def stats(t: torch.Tensor, tag: str) -> None:
    t = t.detach().float()
    print(f"    {tag:<34} shape={tuple(t.shape)}  mean={t.mean():+.4f}  std={t.std():+.4f}  "
          f"min={t.min():+.2f}  max={t.max():+.2f}")


def rel(a: torch.Tensor, b: torch.Tensor) -> str:
    """返回 a 与 b 的相对差（同形状时）。"""
    if a.shape != b.shape:
        return f"形状不同 {tuple(a.shape)} vs {tuple(b.shape)}"
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    d = (a - b).abs()
    denom = b.abs().mean().clamp_min(1e-8)
    return (f"max|Δ|={d.max():.4e}  mean|Δ|={d.mean():.4e}  "
            f"相对={float(d.mean() / denom):.4e}  cos={float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-8)):.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", required=True, choices=["dual", "single_ir", "single_rgb"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--x-layer", type=int, default=11, help="IR 支路首层索引（dual 为 11，single 为 0）")
    args = ap.parse_args()

    root = find_root()
    sys.path.insert(0, str(root))
    os.chdir(root)

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLOMultiModalImageDataset

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.weights)
    meta = ck.get("single_modal_build") or (ck.get("model") or {})
    print(f"[ckpt] {args.weights}")
    print(f"[ckpt] single_modal_build = {ck.get('single_modal_build')}")
    model = model.to(device).float().eval()
    inner = model.model
    print(f"[model] 层数={len(inner)}  末层={type(inner[-1]).__name__}")
    print(f"[model] inner[0]={type(inner[0]).__name__}  inner[{args.x_layer}]={type(inner[args.x_layer]).__name__}")

    # ---- 数据 ----
    data_path = Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    droot = Path(cfg.get("path") or data_path.parent)
    val_sub = str(cfg.get("val", "images/val")).split("/")[-1]
    hyp = get_cfg()
    for k, v in (("mosaic", 0.0), ("mixup", 0.0), ("cutmix", 0.0), ("copy_paste", 0.0),
                 ("mm_rgb_albu_enable", False), ("mm_x_non_geo_enable", False)):
        setattr(hyp, k, v)
    for k in ("cache", "cache_ir"):
        cfg.pop(k, None)
    ds = YOLOMultiModalImageDataset(
        img_path=str(droot / "images" / val_sub), data=cfg, x_modality="ir",
        imgsz=args.imgsz, augment=False, hyp=hyp, rect=False, cache=False,
    )
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)
    batch = next(iter(loader))
    raw = batch["img"]
    print(f"\n[dataloader] batch['img'] shape = {tuple(raw.shape)}  dtype={raw.dtype}")
    if raw.dim() == 4 and raw.shape[1] >= 6:
        rgb_seg = raw[:, 0:3].float()
        x_seg = raw[:, 3:6].float()
        stats(rgb_seg, "batch img[0:3] (RGB 段)")
        stats(x_seg, "batch img[3:6] (X/IR 段)")
        # RGB 段与 X 段是否相同（若相同 -> images_ir 是 RGB 的副本）
        print(f"\n    [关键] RGB 段 vs X 段: {rel(rgb_seg, x_seg)}")
    else:
        rgb_seg, x_seg = None, None
        stats(raw.float(), "batch img")

    # ---- 抓取支路首层输入 ----
    cap: dict = {}

    def mk(name, idx):
        def pre(module, inputs):
            cap[name] = inputs[0].detach() if inputs else None
        return inner[idx].register_forward_pre_hook(pre)

    h0 = mk("primary", 0)
    h1 = mk("secondary", args.x_layer)

    xin = raw.to(device).float()
    with torch.no_grad():
        try:
            model(xin)
            ok = True
        except Exception as e:
            ok = False
            print(f"\n[警告] 6 通道前向失败: {type(e).__name__}: {e}")
            print("        改为只喂 X 段（3 通道）重试…")
            model(xin[:, 3:6])
    h0.remove()
    h1.remove()

    print(f"\n[抓取] 前向{'成功（6 通道）' if ok else '（3 通道 X 段）'}")
    if not cap:
        print("[错误] 未抓到任何输入")
        return 1
    for name in ("primary", "secondary"):
        if name in cap and cap[name] is not None:
            stats(cap[name], f"inner[{0 if name=='primary' else args.x_layer}] 输入")

    if args.mode == "dual" and "primary" in cap and "secondary" in cap \
            and cap["primary"] is not None and cap["secondary"] is not None:
        print(f"\n[关键] IR 支路首层输入 vs RGB 支路首层输入: {rel(cap['secondary'], cap['primary'])}")
        print("        （若 max|Δ|≈0 -> IR 支路直接吃了 RGB 支路同一张图）")

    if "secondary" in cap and cap["secondary"] is not None and x_seg is not None:
        s = cap["secondary"]
        b_x = x_seg.to(s.device)
        b_r = rgb_seg.to(s.device)
        if s.shape[1] == 3 and s.shape[0] == b_x.shape[0]:
            print(f"\n[关键] IR 支路输入 vs dataloader X 段[3:6] : {rel(s, b_x)}")
            print(f"[关键] IR 支路输入 vs dataloader RGB 段[0:3]: {rel(s, b_r)}")
            print("        解读：与哪一段更接近，就说明 IR 支路实际吃的是哪张图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
