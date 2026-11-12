# -*- coding: utf-8 -*-
"""S0 频域诊断（零训练成本）。

回答三个问题（对应《基于FreDFT频域思想的极小目标融合改进方案》第 7 节）
---------------------------------------------------------------
Q1  P3/P4 的高频能量占比是多少？若 P3 高频占比过低，则"融合可用的高频信息"不足，
    方案前提不成立。
Q2  IR 支路 / RGB 支路的高频能量比是多少？若接近 0 或远小于 1，
    说明"fuse 头对 IR 零响应"的根因可能在高频段，LFCF 对症。
Q3  每帧高频占比与"该帧是否含 lt16 目标"的相关性（point-biserial）。
    若 |r| < 0.1，说明高频与极小目标无明显关联，方案应终止。

用固定高斯核做带分解（与 DFT 域滤波等价），不引入任何 FFT。

用法
----
    python scripts/diag_spectral_s0.py \
        --weights YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt \
        --data /3250410011/root/rivermind-data/mendeley_split/data.yaml \
        --split test --imgsz 640 --batch 4 \
        --out /3250410011/root/rivermind-data/spec_diag_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from pathlib import Path

import torch
import torch.nn.functional as F


# 融合颈的两个尺度：RGB/IR 的 P3、P4、P5 输出层（见 dual-fusion.yaml 层 27/39/30/42/33/45）
LAYER_PAIRS = {
    "P3": (27, 39),
    "P4": (30, 42),
    "P5": (33, 45),
}
SIDE_LT16 = 16.0


def make_gauss_kernel(c: int, device, dtype):
    b = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
    k = (b[:, None] * b[None, :]) / 16.0
    return k.view(1, 1, 3, 3).repeat(c, 1, 1, 1)


def band_ratio(x: torch.Tensor) -> torch.Tensor:
    """返回每张图的高频 RMS 占比 sqrt(E_hi / E_all)，形状 (B,)。"""
    c = x.shape[1]
    k = make_gauss_kernel(c, x.device, x.dtype)
    lo = F.conv2d(x, k, padding=1, groups=c)
    hi = x - lo
    e_all = x.float().pow(2).mean(dim=(1, 2, 3))
    e_hi = hi.float().pow(2).mean(dim=(1, 2, 3))
    return (e_hi / e_all.clamp_min(1e-12)).sqrt()


def hi_rms(x: torch.Tensor) -> torch.Tensor:
    """返回每张图的高频 RMS（绝对量级），形状 (B,)。"""
    c = x.shape[1]
    k = make_gauss_kernel(c, x.device, x.dtype)
    lo = F.conv2d(x, k, padding=1, groups=c)
    hi = x - lo
    return hi.float().pow(2).mean(dim=(1, 2, 3)).sqrt()


def point_biserial(values, labels) -> float:
    """values: 连续量；labels: 0/1。返回经典 point-biserial 相关系数。"""
    n = len(values)
    n1 = sum(labels)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    m1 = sum(v for v, l in zip(values, labels) if l) / n1
    m0 = sum(v for v, l in zip(values, labels) if not l) / n0
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    if sd < 1e-12:
        return float("nan")
    return (m1 - m0) / sd * math.sqrt(n0 * n1) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-batches", type=int, default=0, help="0 表示全跑")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLOMultiModalImageDataset
    from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator
    from ultralytics.utils.patches import torch_load

    data_path = Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_path.parent)
    sub = str(cfg.get(args.split, f"images/{args.split}"))

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("S0 频域诊断（固定高斯核带分解，无 FFT）")
    print("=" * 78)
    print(f"  权重: {args.weights}\n  数据: {data_path}  split={args.split}\n  设备: {device}")

    # Linux 侧 pickle 的 PosixPath -> WindowsPath 兼容
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
    print(f"  ckpt epoch: {ckpt.get('epoch')}   末层: {type(model.model[-1]).__name__}")
    model = model.to(device).float().eval()
    inner = model.model
    for scale, (lr, li) in LAYER_PAIRS.items():
        print(f"  {scale}: RGB=L{lr}({inner[lr].type:<8} f={inner[lr].f})  "
              f"IR=L{li}({inner[li].type:<8} f={inner[li].f})")

    # ---- 数据 ----
    hyp = get_cfg()
    for k, v in (("mosaic", 0.0), ("mixup", 0.0), ("cutmix", 0.0), ("copy_paste", 0.0),
                 ("mm_rgb_albu_enable", False), ("mm_x_non_geo_enable", False)):
        setattr(hyp, k, v)
    for _k in ("cache", "cache_ir"):
        cfg.pop(_k, None)

    ds = YOLOMultiModalImageDataset(
        img_path=str(root / "images" / sub.split("/")[-1]),
        data=cfg, x_modality="ir", imgsz=args.imgsz,
        augment=False, hyp=hyp, rect=False, cache=False,
    )
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)
    print(f"  样本数: {len(ds)}")

    vargs = {
        "conf": 0.001, "iou": 0.7, "max_det": 300, "single_cls": False,
        "agnostic_nms": False, "task": "detect", "split": args.split, "half": False,
        "plots": False, "save_json": False, "val": True, "verbose": False,
        "data": str(data_path), "mode": "val", "imgsz": args.imgsz,
        "batch": args.batch, "workers": args.workers, "device": args.device,
    }
    v = DualHeadDetectionValidator(dataloader=loader, save_dir=None, args=vargs)
    v.data, v.device, v.training = cfg, device, False
    v.args.plots = False

    # ---- 逐层统计缓冲：hook 每批写入，主循环按批取出，顺序严格对齐 ----
    buf_ratio = {name: [] for name in ("P3_RGB", "P3_IR", "P4_RGB", "P4_IR", "P5_RGB", "P5_IR")}
    buf_hirms = {k: [] for k in buf_ratio}

    def make_hook(tag):
        def hook(module, inputs, output):
            if not torch.is_tensor(output):
                return
            with torch.no_grad():
                buf_ratio[tag].append(band_ratio(output).detach().cpu())
                buf_hirms[tag].append(hi_rms(output).detach().cpu())
        return hook

    handles = []
    for scale, (lr, li) in LAYER_PAIRS.items():
        handles.append(inner[lr].register_forward_hook(make_hook(f"{scale}_RGB")))
        handles.append(inner[li].register_forward_hook(make_hook(f"{scale}_IR")))

    # ---- 逐帧记录 ----
    frames = []  # 每项: {file, ori_h, ori_w, has_lt16, min_side, ratios..., hirms...}
    n_batch = 0
    with torch.inference_mode():
        for batch in loader:
            batch = v.preprocess(batch)
            model(batch["img"])

            files = batch.get("im_file") or batch.get("img_path")
            oris = batch.get("ori_shape")
            boxes = batch.get("bboxes")
            bidx = batch.get("batch_idx")
            bs = len(files)
            for j in range(bs):
                h, w = int(oris[j][0]), int(oris[j][1])
                has_lt16, min_side = False, float("inf")
                if boxes is not None and bidx is not None:
                    sel = boxes[bidx == j]
                    for b in sel:
                        bw, bh = float(b[2]) * w, float(b[3]) * h
                        side = math.sqrt(max(bw, 1e-6) * max(bh, 1e-6))
                        min_side = min(min_side, side)
                        if side < SIDE_LT16:
                            has_lt16 = True
                frames.append({
                    "file": str(files[j]),
                    "ori_h": h, "ori_w": w,
                    "has_lt16": has_lt16,
                    "min_side": None if min_side == float("inf") else round(min_side, 2),
                })
            for tag in buf_ratio:
                arr = buf_ratio[tag].pop()
                hr = buf_hirms[tag].pop()
                assert len(arr) == bs, f"{tag} 数量不一致 {len(arr)} vs {bs}"
                for j in range(bs):
                    frames[len(frames) - bs + j][tag] = float(arr[j])
                    frames[len(frames) - bs + j][tag + "_hirms"] = float(hr[j])

            n_batch += 1
            if n_batch % 20 == 0:
                print(f"    已处理 {n_batch} 批 / {len(frames)} 帧", end="\r")
            if args.max_batches and n_batch >= args.max_batches:
                break
    for h in handles:
        h.remove()
    print(f"\n  完成，共 {len(frames)} 帧")

    # ---- 汇总 ----
    def mean_of(key, subset=None):
        vals = [f[key] for f in frames if key in f and (subset is None or subset(f))]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {}
    print("\n" + "=" * 78)
    print("Q1 各层高频 RMS 占比（sqrt(E_hi/E_all)，全帧均值）")
    print("=" * 78)
    for tag in ("P3_RGB", "P3_IR", "P4_RGB", "P4_IR", "P5_RGB", "P5_IR"):
        m = mean_of(tag)
        summary.setdefault("hi_ratio_mean", {})[tag] = m
        print(f"  {tag}: {m:.4f}  ({100 * m:.1f}%)")

    print("\n" + "=" * 78)
    print("Q2 IR / RGB 高频能量比（高频 RMS 之比）")
    print("=" * 78)
    for scale in ("P3", "P4", "P5"):
        num = mean_of(f"{scale}_IR_hirms")
        den = mean_of(f"{scale}_RGB_hirms")
        r = num / den if den > 0 else float("nan")
        summary.setdefault("ir_rgb_hi_rms_ratio", {})[scale] = r
        print(f"  {scale}: IR={num:.4f}  RGB={den:.4f}  -> IR/RGB = {r:.3f}"
              f"  (IR 能量占比 {100 * num / (num + den):.1f}%)")

    print("\n" + "=" * 78)
    print("Q3 高频占比 与 lt16 存在的相关性（point-biserial）")
    print("=" * 78)
    summary["point_biserial"] = {}
    for tag in ("P3_RGB", "P3_IR", "P4_RGB", "P4_IR"):
        vals = [f[tag] for f in frames if tag in f]
        labs = [1 if f["has_lt16"] else 0 for f in frames if tag in f]
        r = point_biserial(vals, labs)
        m1 = mean_of(tag, lambda f: f["has_lt16"])
        m0 = mean_of(tag, lambda f: not f["has_lt16"])
        summary["point_biserial"][tag] = {"r": r, "mean_with_lt16": m1, "mean_without": m0}
        print(f"  {tag}: r={r:+.3f}   含lt16帧均值={m1:.4f}   不含lt16帧均值={m0:.4f}")

    n_pos = sum(1 for f in frames if f["has_lt16"])
    summary["n_frames"] = len(frames)
    summary["n_frames_with_lt16"] = n_pos
    print(f"\n  含 lt16 目标的帧: {n_pos}/{len(frames)} ({100 * n_pos / max(len(frames), 1):.1f}%)")

    # ---- 判读 ----
    print("\n" + "=" * 78)
    print("判读（按方案文档第 7 节的终止规则）")
    print("=" * 78)
    p3_rgb = summary["hi_ratio_mean"]["P3_RGB"]
    verdicts = {
        "P3_hi_ratio_ge_10pct": p3_rgb >= 0.10,
        "ir_hi_dominant_or_comparable": summary["ir_rgb_hi_rms_ratio"]["P3"] >= 0.5,
        "correlation_ge_0.1": abs(summary["point_biserial"]["P3_RGB"]["r"]) >= 0.1,
    }
    print(f"  [1] P3(RGB) 高频占比 {100 * p3_rgb:.1f}% >= 10%  -> "
          f"{'通过' if verdicts['P3_hi_ratio_ge_10pct'] else '不通过（高频信息不足，方案前提存疑）'}")
    print(f"  [2] P3 IR/RGB 高频比 {summary['ir_rgb_hi_rms_ratio']['P3']:.3f} >= 0.5  -> "
          f"{'通过（IR 高频未被压制）' if verdicts['ir_hi_dominant_or_comparable'] else '不通过（IR 高频被压制，LFCF 可能对症）'}")
    print(f"     注：[2] 的方向含义与 [1] 相反 —— IR/RGB 越低，越说明 IR 高频被压制，"
          f"越支持「高频门控」这一干预。")
    r = summary["point_biserial"]["P3_RGB"]["r"]
    print(f"  [3] |r| = {abs(r):.3f} >= 0.1  -> "
          f"{'通过' if verdicts['correlation_ge_0.1'] else '不通过（高频与 lt16 无明显关联，应终止）'}")

    summary["verdicts"] = verdicts
    summary["note"] = ("判读规则见《基于FreDFT频域思想的极小目标融合改进方案》第 7 节；"
                       "[2] 为反向指标，IR/RGB 越低越支持干预，不是否决条件。")

    out = Path(args.out) if args.out else Path("spec_diag.json")
    out.write_text(json.dumps({"summary": summary, "frames": frames}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
