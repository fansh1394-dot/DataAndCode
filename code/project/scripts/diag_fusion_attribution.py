# -*- coding: utf-8 -*-
"""融合头归因诊断（零训练成本）。

回答三个问题
------------
Q1  融合颈两侧（RGB 半边 / IR 半边）激活幅值谁主导？
Q2  基准分模态表现：各分支在全验证集以及 both / only_rgb / only_ir
    子集上分别多少？（论文表格口径，由验证器原生 subset 累加器给出）
Q3  把融合分支的 **RGB 加性项** 置零（不动 offset/reliability 的输入，
    故不引入分布外伪影），融合头能否转向 IR 特征？

为什么 Q3 要用「加性项置零」而不是「屏蔽输入半边」
--------------------------------------------------
早期做法是在融合颈处把 RGB 半边**特征置零**（即训练期
``mm_fusion_rgb_mask`` 的同一位置）。但 ``ReliabilityAwareFusion`` 的
``offset`` / ``reliability`` 两个子网都以 ``cat(rgb_proj(x_rgb), ir_proj(x_ir))``
为输入，一旦把 RGB 半边置零，这两个子网立刻收到**分布外**输入：测到的是
「模块被损坏后能否兜底」，而非「融合是否有能力使用 IR」。实测该做法下
fuse 头塌到 0.044，属伪影所致，不能作为结论。

本脚本因此提供三种掩码口径，全部只在**推理**中生效（不改权重、不训练）：

    orig            不掩码（基准，走原始 forward）
    repl_off        与 orig 等价，但走脚本内复刻的 forward（自检复刻是否逐位一致）
    drop_rgb_term   仅去掉 fused 里的 RGB 加性项：fused = w_ir * ir_warp
                    （offset / reliability 仍读干净的完整特征，无伪影）
    ir_only         门控塌缩到纯 IR：fused = ir_warp（幅值与正常同量级，
                    用于排除 drop_rgb_term「幅值减半」这一混淆）

用法
----
    python scripts/diag_fusion_attribution.py \
        --weights ResTest/v2_A4_s0/weights/best.pt \
        --data /root/rivermind-data/uav_data_v2/data.yaml \
        --mask-mode drop_rgb_term \
        --out /root/rivermind-data/diag_A4_best_maskv2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 融合颈的三个节点层索引（见 yolo11n-mm-dual-fusion*.yaml 层 46/48/50）
FUSION_CONCAT_LAYERS = (46, 48, 50)

# 推理期掩码状态；被复刻的 ReliabilityAwareFusion.forward 读取
MASK_STATE = {"mode": "orig"}

# 幅值诊断（drop_rgb_term_scaled 会写入，用于验证「是否只是幅值问题」）
_MAG = {}


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


def load_frame_kinds(data_path: Path, split: str) -> dict:
    """返回 {frame_id: kind}，kind ∈ {both, only_ir, only_vis}。

    优先读 frame_kinds.json（外层取第一个值即为 {frame_id: kind}）；
    缺失时回退读 v2 元数据 metadata/<split>.json 的 frames 字段，
    并把 v2 的 only_rgb 归一为 only_vis。
    """
    fp = data_path.parent / "frame_kinds.json"
    if fp.exists():
        raw = json.loads(fp.read_text(encoding="utf-8"))
        inner = next(iter(raw.values())) if raw else {}
        return {str(k): ("only_vis" if v == "only_rgb" else v) for k, v in inner.items()}
    mp = data_path.parent / "metadata" / f"{split}.json"
    if mp.exists():
        raw = json.loads(mp.read_text(encoding="utf-8"))
        frames = raw.get("frames", raw) if isinstance(raw, dict) else {}
        out = {}
        for k, v in frames.items():
            kind = v.get("kind") if isinstance(v, dict) else v
            out[str(k)] = "only_vis" if kind == "only_rgb" else kind
        return out
    return {}


def extract_maps(stats: dict) -> tuple:
    """从扁平 stats 抽出「分支总体」与「分支×子集」两套 mAP50-95。

    键格式（见 dual_val.py）：
        metrics/<branch>/metrics/mAP50-95(B)                       -> 总体
        metrics/subset/<branch>/<subset>/metrics/mAP50-95(B)       -> 子集
    """
    overall, subset = {}, {}
    for k, val in stats.items():
        if not k.endswith("mAP50-95(B)"):
            continue
        parts = k.split("/")
        if len(parts) == 4 and parts[0] == "metrics" and parts[2] == "metrics":
            overall[parts[1]] = float(val)
        elif len(parts) == 6 and parts[0] == "metrics" and parts[1] == "subset":
            subset[(parts[2], parts[3])] = float(val)
    return overall, subset


#: 报表分支首选顺序：已知分支按此排序，未知分支随后按名称排序。
_STD_BRANCHES = ("rgb", "ir", "fuse", "fuse_ir")


def kv_dynamic_order(*maps: dict | None) -> list:
    """汇总多次评测产出的分支名，按固定优先级排序后返回。

    三头权重只产出 ``rgb/ir/fuse``，四头旧权重才会多出 ``fuse_ir``；
    报表一律以实际键为准，避免写死四元组后 KeyError 或漏行。
    """
    names = set()
    for m in maps:
        if m:
            names.update(m.keys())
    known = [n for n in _STD_BRANCHES if n in names]
    return known + sorted(names - set(_STD_BRANCHES))


def main() -> int:
    here = Path(__file__).resolve()
    repo = find_repo_root(here.parent)
    sys.path.insert(0, str(repo))

    ap = argparse.ArgumentParser(description="融合头归因诊断（含无伪影掩码评测）")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--mask-mode", default="drop_rgb_term",
                    choices=("drop_rgb_term", "drop_rgb_term_scaled", "ir_only", "repl_off"))
    ap.add_argument("--also-ir-only", action="store_true",
                    help="mask-mode=drop_rgb_term 时同时跑 ir_only 以排除幅值混淆")
    ap.add_argument("--also-scaled", action="store_true",
                    help="mask-mode=drop_rgb_term 时同时跑 drop_rgb_term_scaled（幅值匹配版）")
    ap.add_argument("--sanity-batches", type=int, default=20,
                    help="复刻 forward 自检所用的批数（orig vs repl_off 逐位比较）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLOMultiModalImageDataset
    from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator
    from ultralytics.nn.mm.reliable_fusion import ReliabilityAwareFusion
    try:
        from ultralytics.nn.mm.gated_fusion import GatedReliabilityFusion
    except Exception:  # 老环境无该类时保持向后兼容
        GatedReliabilityFusion = None
    try:
        from ultralytics.nn.mm.deform_fusion import DeformableCrossModalFusion
    except Exception:  # C4 新增类，老环境无则跳过
        DeformableCrossModalFusion = None
    try:
        from ultralytics.nn.mm.partition_fusion import ChannelPartitionFusion
    except Exception:  # D3 新增类，老环境无则跳过
        ChannelPartitionFusion = None
    from ultralytics.utils.patches import torch_load

    import pathlib

    import yaml

    data_path = Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_path.parent)
    val_sub = str(cfg.get("val", "images/val"))

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("融合头归因诊断（零训练成本）")
    print("=" * 78)
    print(f"  权重: {args.weights}")
    print(f"  数据: {data_path}")
    print(f"  设备: {device}   掩码口径: {args.mask_mode}")

    # ckpt 在 Linux 侧 pickle 为 PosixPath。正确方向是 WindowsPath -> PosixPath。
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

    print("\n  融合颈层结构校验:")
    for li in FUSION_CONCAT_LAYERS:
        m = inner[li]
        print(f"    L{li:<3} type={type(m).__name__:<22} f={getattr(m, 'f', None)}")
    has_raf = all(isinstance(inner[li], ReliabilityAwareFusion) for li in FUSION_CONCAT_LAYERS)
    has_gated = (
        GatedReliabilityFusion is not None
        and all(isinstance(inner[li], GatedReliabilityFusion) for li in FUSION_CONCAT_LAYERS)
    )
    has_deform = (
        DeformableCrossModalFusion is not None
        and all(isinstance(inner[li], DeformableCrossModalFusion) for li in FUSION_CONCAT_LAYERS)
    )
    has_partition = (
        ChannelPartitionFusion is not None
        and all(isinstance(inner[li], ChannelPartitionFusion) for li in FUSION_CONCAT_LAYERS)
    )
    has_fuse = has_raf or has_gated or has_deform or has_partition
    if not has_fuse:
        print("  [提示] 融合颈并非 ReliabilityAwareFusion / GatedReliabilityFusion / "
              "DeformableCrossModalFusion / ChannelPartitionFusion（可能为普通 Concat），"
              "drop_rgb_term / ir_only 不可用")

    # ---- 复刻 forward + 掩码注入（仅推理，不改权重）----
    _ORIG_FORWARD = ReliabilityAwareFusion.forward

    def _repl_forward(self, xs):
        mode = MASK_STATE["mode"]
        if mode == "orig":
            return _ORIG_FORWARD(self, xs)
        # 与 reliable_fusion.ReliabilityAwareFusion.forward 逐行等价，只在最后融合步改写。
        # offset / reliability 始终读「干净完整」的 rgb/ir 特征，故不产生分布外伪影。
        rgb, ir = xs
        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        pair = torch.cat((rgb, ir), dim=1)
        raw_offset = self.offset(pair).tanh() * self.max_offset
        b, _, h, w = raw_offset.shape
        base = self._base_grid(b, h, w, pair.device, pair.dtype)
        rgb_grid = (base + raw_offset[:, 0:2].permute(0, 2, 3, 1)).clamp(-1, 1)
        ir_grid = (base + raw_offset[:, 2:4].permute(0, 2, 3, 1)).clamp(-1, 1)
        rgb_warp = F.grid_sample(rgb, rgb_grid, mode="bilinear", align_corners=True, padding_mode="border")
        ir_warp = F.grid_sample(ir, ir_grid, mode="bilinear", align_corners=True, padding_mode="border")
        weights = self.reliability(torch.cat((rgb_warp, ir_warp), dim=1)).softmax(dim=1)
        self.last_offset_abs_mean = raw_offset.detach().abs().mean()
        self.last_reliability_mean = weights.detach().mean(dim=(0, 2, 3))
        if mode == "drop_rgb_term":
            fused = weights[:, 1:2] * ir_warp           # 仅去掉 RGB 加性项（幅值约减半）
        elif mode == "ir_only":
            fused = ir_warp                             # 门控塌缩到纯 IR（幅值同量级）
        else:  # repl_off：与 orig 的融合式一致，仅用于自检复刻
            fused = weights[:, 0:1] * rgb_warp + weights[:, 1:2] * ir_warp
        return self.out(fused)

    # 门控融合（改动 A）的复刻 forward：与 gated_fusion.GatedReliabilityFusion.forward 逐行等价，
    # 仅在最后融合步改写。融合式为 fused = base_feat + g*(ir_warp - rgb_warp)，
    # 展开即 fused = (1-g)*rgb_warp + g*ir_warp。
    _ORIG_FORWARD_GATED = (
        GatedReliabilityFusion.forward if GatedReliabilityFusion is not None else None
    )

    def _repl_forward_gated(self, xs):
        mode = MASK_STATE["mode"]
        if mode == "orig":
            return _ORIG_FORWARD_GATED(self, xs)
        rgb, ir = xs
        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        pair = torch.cat((rgb, ir), dim=1)
        raw_offset = self.offset(pair).tanh() * self.max_offset
        b, _, h, w = raw_offset.shape
        base = self._base_grid(b, h, w, pair.device, pair.dtype)
        rgb_grid = (base + raw_offset[:, 0:2].permute(0, 2, 3, 1)).clamp(-1, 1)
        ir_grid = (base + raw_offset[:, 2:4].permute(0, 2, 3, 1)).clamp(-1, 1)
        rgb_warp = F.grid_sample(rgb, rgb_grid, mode="bilinear", align_corners=True, padding_mode="border")
        ir_warp = F.grid_sample(ir, ir_grid, mode="bilinear", align_corners=True, padding_mode="border")
        g = torch.sigmoid(self.gate(torch.cat((rgb_warp, ir_warp), dim=1)))
        base_feat = 0.5 * (rgb_warp + ir_warp) if getattr(self, "base_mode", "rgb") == "mean" else rgb_warp
        self.last_offset_abs_mean = raw_offset.detach().abs().mean()
        self.last_gate_mean = g.detach().mean()
        normal = base_feat + g * (ir_warp - rgb_warp)   # 基准融合式（= (1-g)rgb + g*ir）
        if mode == "drop_rgb_term":
            # 去掉 RGB 加性项：(1-g)*rgb 置零 -> g*ir。注意 g≈0.5 时幅值约为正常的一半，
            # 故该口径本身含「幅值减半」混淆；幅值匹配版见 drop_rgb_term_scaled。
            fused = g * ir_warp
        elif mode == "drop_rgb_term_scaled":
            # 幅值匹配版：去掉 RGB 项后，把 fused 整体 RMS 归一到基准 fused 的 RMS，
            # 使 out 卷积看到的幅值与训练分布一致，从而**只**改变「有无 RGB 信息」这一个变量。
            raw = g * ir_warp
            n_rms = normal.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            r_rms = raw.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            fused = raw * (n_rms / r_rms)
            _MAG["gated_scaled_ratio"] = float((n_rms / r_rms).item())
            _MAG["gated_raw_rms"] = float(r_rms.item())
            _MAG["gated_normal_rms"] = float(n_rms.item())
        elif mode == "ir_only":
            fused = ir_warp                             # 门控塌缩到纯 IR（幅值同量级）
        else:  # repl_off：与 orig 的融合式一致，仅用于自检复刻
            fused = normal
        return self.out(fused)

    if has_raf:
        ReliabilityAwareFusion.forward = _repl_forward
    if has_gated:
        GatedReliabilityFusion.forward = _repl_forward_gated

    # 可变形交叉注意力融合（改动 C4）的复刻 forward：与 deform_fusion.DeformableCrossModalFusion.forward
    # 逐行等价，仅在最后融合步改写。融合式 fused = base_feat + g*(ir_attended - rgb)。
    # 注意：C4 的 IR 证据来自「K 点可变形采样 + 内容注意力」，故「IR 半边」不是单一 ir_warp，
    # 而是 ir_attended。drop_rgb_term / ir_only 需按此语义改写，否则探针口径不成立。
    _ORIG_FORWARD_DEFORM = (
        DeformableCrossModalFusion.forward if DeformableCrossModalFusion is not None else None
    )

    def _repl_forward_deform(self, xs):
        mode = MASK_STATE["mode"]
        if mode == "orig":
            return _ORIG_FORWARD_DEFORM(self, xs)
        import math as _math
        rgb, ir = xs
        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        b, c, h, w = rgb.shape
        pair = torch.cat((rgb, ir), dim=1)
        offsets = torch.tanh(self.offset(pair)) * self.max_offset
        offsets = offsets.view(b, self.num_points, 2, h, w)
        base_grid = self._base_grid(b, h, w, pair.device, pair.dtype)
        sample_grid = (base_grid.unsqueeze(1) + offsets.permute(0, 1, 3, 4, 2)).clamp(-1, 1)
        ir_expand = ir.unsqueeze(1).expand(b, self.num_points, c, h, w).reshape(b * self.num_points, c, h, w)
        grid_flat = sample_grid.reshape(b * self.num_points, h, w, 2)
        ir_sampled = F.grid_sample(ir_expand, grid_flat, mode="bilinear", align_corners=True,
                                   padding_mode="border").view(b, self.num_points, c, h, w)
        attn_in = torch.cat((pair, offsets.reshape(b, 2 * self.num_points, h, w)), dim=1)
        attn = self.attn(attn_in).softmax(dim=1)
        ir_attended = (ir_sampled * attn.unsqueeze(2)).sum(dim=1)
        g = torch.sigmoid(self.gate(pair))
        base_mode = getattr(self, "base_mode", "rgb")
        base_feat = 0.5 * (rgb + ir_attended) if base_mode == "mean" else (
            ir_attended if base_mode == "ir" else rgb)
        self.last_offset_abs_mean = offsets.detach().abs().mean()
        self.last_gate_mean = g.detach().mean()
        normal = base_feat + g * (ir_attended - rgb)
        if mode == "drop_rgb_term":
            fused = g * ir_attended
        elif mode == "drop_rgb_term_scaled":
            raw = g * ir_attended
            n_rms = normal.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            r_rms = raw.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            fused = raw * (n_rms / r_rms)
            _MAG["gated_scaled_ratio"] = float((n_rms / r_rms).item())
            _MAG["gated_raw_rms"] = float(r_rms.item())
            _MAG["gated_normal_rms"] = float(n_rms.item())
        elif mode == "ir_only":
            fused = ir_attended
        else:  # repl_off
            fused = normal
        return self.out(fused)

    if has_deform:
        DeformableCrossModalFusion.forward = _repl_forward_deform

    # 通道分工融合（改动 D3）的复刻 forward。
    #
    # D3 的 fused 由三段硬拼而成：
    #     [ RGB 专属(n_rgb) | IR 专属(n_ir) | 交互(n_mix) ]
    # 其中 RGB 专属段是**硬连接**（rgb_warp 直接切片），结构上无法去除；
    # IR 专属段同样硬连接（ir_attended 切片），无 gate。
    # 故 D3 下的掩码口径需重新定义，否则测的不是同一个东西：
    #
    #   drop_rgb_term / drop_rgb_term_scaled:
    #       把所有 RGB 来源全部去掉 —— RGB 专属段置零、交互段取 g*ir_mix
    #       （即交互段也走 IR）。这是 D3 语义下最接近「无 RGB 项」的操作。
    #       若 fuse 在此口径下仍保持高水平，说明**IR 通路确实承载了信息**，
    #       这正是 D3 相对 gated/deform 族的结构性优势所在。
    #
    #   ir_only:
    #       RGB 专属段与交互段的 RGB 项全部去除，仅保留 IR 专属段 + 交互段 IR 项。
    #
    #   repl_off:
    #       与 orig 等价的复刻，用于自检。
    _ORIG_FORWARD_PART = (
        ChannelPartitionFusion.forward if ChannelPartitionFusion is not None else None
    )

    def _repl_forward_partition(self, xs):
        mode = MASK_STATE["mode"]
        if mode == "orig":
            return _ORIG_FORWARD_PART(self, xs)
        import math as _math
        rgb, ir = xs
        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        b, c, h, w = rgb.shape
        pair = torch.cat((rgb, ir), dim=1)

        offsets = torch.tanh(self.offset(pair)) * self.max_offset
        offsets = offsets.view(b, self.num_points, 2, h, w)
        base_grid = self._base_grid(b, h, w, pair.device, pair.dtype)
        sample_grid = (base_grid.unsqueeze(1) + offsets.permute(0, 1, 3, 4, 2)).clamp(-1, 1)

        rgb_warp = rgb
        ir_expand = ir.unsqueeze(1).expand(b, self.num_points, c, h, w).reshape(
            b * self.num_points, c, h, w)
        grid_flat = sample_grid.reshape(b * self.num_points, h, w, 2)
        ir_sampled = F.grid_sample(
            ir_expand, grid_flat, mode="bilinear", align_corners=True, padding_mode="border"
        ).view(b, self.num_points, c, h, w)

        attn_in = torch.cat((pair, offsets.reshape(b, 2 * self.num_points, h, w)), dim=1)
        attn = self.attn(attn_in).softmax(dim=1)
        ir_attended = (ir_sampled * attn.unsqueeze(2)).sum(dim=1)

        n_rgb, n_ir = self.n_rgb, self.n_ir
        self.last_offset_abs_mean = offsets.detach().abs().mean()
        with torch.no_grad():
            p = attn.clamp_min(1e-9)
            self.last_attn_entropy = (
                -(p * p.log()).sum(dim=1).mean() / _math.log(self.num_points)
                if self.num_points > 1 else torch.zeros((), device=rgb.device))

        # ---- 正常拼接（repl_off 用）----
        def build(rgb_seg, ir_seg, mix):
            parts = []
            if n_rgb > 0:
                parts.append(rgb_seg)
            parts.append(ir_seg)
            if self.n_mix > 0:
                parts.append(mix)
            return torch.cat(parts, dim=1)

        rgb_seg = rgb_warp[:, :n_rgb]
        ir_seg = ir_attended[:, n_rgb:n_rgb + n_ir]
        if self.n_mix > 0:
            rgb_mix = rgb_warp[:, n_rgb + n_ir:]
            ir_mix = ir_attended[:, n_rgb + n_ir:]
            if self.gate is not None:
                g = torch.sigmoid(self.gate(pair))
                base_mix = 0.5 * (rgb_mix + ir_mix) if self.base_mode == "mean" else rgb_mix
                mix_normal = base_mix + g * (ir_mix - rgb_mix)
                self.last_gate_mean = g.detach().mean()
                self.last_gate_std = g.detach().std()
                # drop_rgb_term：交互段去掉 RGB 加性项 -> g*ir_mix
                mix_norgb = g * ir_mix
            else:
                mix_normal = rgb_mix
                mix_norgb = rgb_mix
        else:
            mix_normal = None
            mix_norgb = None

        if mode == "drop_rgb_term":
            # RGB 专属段归零 + 交互段去 RGB 项（幅值会下降，故另有 scaled 版）
            zeros = torch.zeros_like(rgb_seg)
            fused = build(zeros, ir_seg, mix_norgb)
        elif mode == "drop_rgb_term_scaled":
            # 幅值匹配版：把去 RGB 后的整体 RMS 归一到基准 fused 的 RMS
            zeros = torch.zeros_like(rgb_seg)
            raw = build(zeros, ir_seg, mix_norgb)
            normal = build(rgb_seg, ir_seg, mix_normal)
            n_rms = normal.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            r_rms = raw.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            fused = raw * (n_rms / r_rms)
            _MAG["gated_scaled_ratio"] = float((n_rms / r_rms).item())
            _MAG["gated_raw_rms"] = float(r_rms.item())
            _MAG["gated_normal_rms"] = float(n_rms.item())
        elif mode == "ir_only":
            # 仅保留 IR 来源：RGB 专属段归零，交互段取纯 IR
            zeros = torch.zeros_like(rgb_seg)
            fused = build(zeros, ir_seg, ir_attended[:, n_rgb + n_ir:] if self.n_mix > 0 else None)
        else:  # repl_off
            fused = build(rgb_seg, ir_seg, mix_normal)
        return self.out(fused)

    if has_partition:
        ChannelPartitionFusion.forward = _repl_forward_partition

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
    print(f"\n  验证集样本数: {len(ds)}")

    vargs = {
        "conf": 0.001, "iou": 0.7, "max_det": 300, "single_cls": False,
        "agnostic_nms": False, "task": "detect", "split": "val", "half": False,
        "plots": False, "save_json": False, "val": True, "verbose": False,
        "data": str(data_path), "mode": "val", "imgsz": args.imgsz,
        "batch": args.batch, "workers": args.workers, "device": args.device,
    }
    v = DualHeadDetectionValidator(dataloader=loader, save_dir=None, args=vargs)
    v.data, v.device, v.training = cfg, device, False
    v.args.plots = False

    frame_kind = load_frame_kinds(data_path, val_sub.split("/")[-1])
    n_ir = sum(1 for k in frame_kind.values() if k == "only_ir")
    n_vis = sum(1 for k in frame_kind.values() if k in ("only_vis", "only_rgb"))
    n_both = sum(1 for k in frame_kind.values() if k == "both")
    print(f"  逐帧清单: both={n_both}  仅IR={n_ir}  仅RGB={n_vis}")

    # ---- Q1：融合颈两侧激活幅值 ----
    amp_stat = {li: {"rgb": 0.0, "ir": 0.0, "n": 0} for li in FUSION_CONCAT_LAYERS}

    def make_amp_hook(layer_idx: int):
        def hook(module, inputs, output):
            # Concat/ReliabilityAwareFusion 的 forward 收「一个 list」[rgb_half, ir_half]
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
                return output
            x = inputs[0]
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                return output
            a, b = x
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                return output
            with torch.no_grad():
                ra = float(a.detach().float().pow(2).mean().sqrt())
                rb = float(b.detach().float().pow(2).mean().sqrt())
            st = amp_stat[layer_idx]
            st["rgb"] += ra
            st["ir"] += rb
            st["n"] += 1
            return output
        return hook

    handles = [inner[li].register_forward_hook(make_amp_hook(li)) for li in FUSION_CONCAT_LAYERS]

    def run_pass(mask_mode: str, max_batches: int = None):
        """跑一遍验证集，返回 {分支总体 + 分支×子集} 的 mAP50-95。"""
        MASK_STATE["mode"] = mask_mode
        v.init_metrics(model)
        v.jdict = []
        n = 0
        for bi, batch in enumerate(loader):
            batch = v.preprocess(batch)
            with torch.no_grad():
                preds_raw = model(batch["img"])
            if not isinstance(preds_raw, dict):
                return None
            for name in v.branch_names:
                if name not in preds_raw:
                    continue
                preds = v.postprocess(v._unwrap_branch_preds(preds_raw[name]))
                v._update_branch_metrics(name, preds, batch)
                v._update_subset_metrics(name, preds, batch)
            n += 1
            if max_batches and n >= max_batches:
                break
            if (bi + 1) % 20 == 0:
                print(f"    已处理 {bi + 1} 批 ...", end="\r")
        print()
        stats = {}
        stats.update(v._branch_stats())
        stats.update(v._subset_stats())
        return stats

    print("\n" + "=" * 78)
    print("Q1 融合颈两侧激活幅值占比（基准口径）")
    print("=" * 78)
    stats_base = run_pass("orig")
    if stats_base is None:
        print("[错误] 模型输出非 dict，该权重不是双头架构")
        return 1
    base_overall, base_subset = extract_maps(stats_base)

    for li in FUSION_CONCAT_LAYERS:
        st = amp_stat[li]
        if st["n"] == 0:
            continue
        ra, rb = st["rgb"] / st["n"], st["ir"] / st["n"]
        tot = ra + rb
        print(f"  L{li}  RGB半边RMS={ra:.4f}  IR半边RMS={rb:.4f}"
              f"   -> RGB占比 {100 * ra / tot:.1f}%  /  IR占比 {100 * rb / tot:.1f}%")

    # 冻结基准幅值统计（后续自检/掩码批次可能再触发 hook，不再计入）
    amp_final = {li: dict(amp_stat[li]) for li in FUSION_CONCAT_LAYERS}

    # ---- 复刻 forward 自检：同一批输入下 orig 与 repl_off 的原始输出必须逐位一致 ----
    # 注意：不能拿「部分批次跑出的 mAP」比 —— build_dataloader 返回的是
    # InfiniteDataLoader（底层 _RepeatSampler 循环取样），两次部分迭代会落在不同
    # 相位、拿到不同数据，比较无效。这里改为固定同一批输入、直接比原始张量。
    sanity = {"n_batches": 0, "max_abs_diff": None, "ok": None}
    if has_fuse and args.sanity_batches > 0:
        print("\n" + "=" * 78)
        print(f"自检：复刻 forward（repl_off）vs 原始 forward（orig），逐位比较前 {args.sanity_batches} 批原始输出")
        print("=" * 78)

        def _flat(o):
            items = o.values() if isinstance(o, dict) else [o]
            res = []
            for it in items:
                t = v._unwrap_branch_preds(it)
                if torch.is_tensor(t):
                    res.append(t)
            return res

        maxdiff, n_done = 0.0, 0
        for bi, batch in enumerate(loader):
            batch = v.preprocess(batch)
            outs = {}
            for mode in ("orig", "repl_off"):
                MASK_STATE["mode"] = mode
                with torch.no_grad():
                    o = model(batch["img"])
                outs[mode] = [t.clone() for t in _flat(o)]
            for a, b in zip(outs["orig"], outs["repl_off"]):
                maxdiff = max(maxdiff, float((a - b).abs().max()))
            n_done += 1
            if n_done >= args.sanity_batches:
                break
        MASK_STATE["mode"] = "orig"
        sanity = {"n_batches": n_done, "max_abs_diff": maxdiff, "ok": maxdiff == 0.0}
        print(f"  比较 {n_done} 批，最大绝对差 = {maxdiff:.3e}  -> "
              f"{'逐位一致 ✔' if maxdiff == 0.0 else '不一致 ✘（复刻有误，掩码结论不可用）'}")

    # ---- Q3：无伪影掩码 ----
    print("\n" + "=" * 78)
    print(f"Q3 无伪影掩码：{args.mask_mode}（offset/reliability 读干净输入）")
    print("=" * 78)
    MASK_STATE["mode"] = "orig"
    stats_mask = run_pass(args.mask_mode)
    mask_overall, mask_subset = extract_maps(stats_mask)

    also = {}
    if args.also_ir_only and args.mask_mode != "ir_only" and has_fuse:
        print("\n" + "=" * 78)
        print("Q3b 幅值对照：ir_only（fused = ir_warp，幅值同量级）")
        print("=" * 78)
        MASK_STATE["mode"] = "orig"
        s_ironly = run_pass("ir_only")
        also["ir_only"] = extract_maps(s_ironly)
    if args.also_scaled and args.mask_mode != "drop_rgb_term_scaled" and has_fuse:
        print("\n" + "=" * 78)
        print("Q3c 幅值匹配对照：drop_rgb_term_scaled（去掉 RGB 项但 RMS 归一到基准）")
        print("=" * 78)
        MASK_STATE["mode"] = "orig"
        s_scaled = run_pass("drop_rgb_term_scaled")
        also["scaled"] = extract_maps(s_scaled)
    MASK_STATE["mode"] = "orig"
    handles and [h.remove() for h in handles]

    # ---- 报表 ----
    # 分支名不得硬编码：三头改造后权重里已无 fuse_ir，四头旧权重的 fuse_ir 也只是
    # 可选第 4 头。一律按「基准/掩码两次评测实际产出的分支键」动态取并集。
    branch_names = list(kv_dynamic_order(base_overall, mask_overall, extra_overall))

    def print_table(title, kv_base, kv_mask, extra=None):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        head = f"{'分支':<12}{'基准':>12}{'掩码':>12}{'变化':>12}"
        if extra is not None:
            head += f"{'ir_only':>12}"
        print(head)
        print("-" * 78)
        for name in branch_names:
            if name not in kv_base and name not in kv_mask:
                continue
            b = kv_base.get(name, 0.0)
            m = kv_mask.get(name, 0.0)
            row = f"{name:<12}{b:>12.5f}{m:>12.5f}{m - b:>+12.5f}"
            if extra is not None:
                row += f"{extra.get(name, 0.0):>12.5f}"
            print(row)

    def print_subset(title, kv_base, kv_mask, extra=None):
        keys = sorted(set(kv_base) | set(kv_mask))
        if not keys:
            return
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        head = f"{'分支/子集':<20}{'基准':>12}{'掩码':>12}{'变化':>12}"
        if extra is not None:
            head += f"{'ir_only':>12}"
        print(head)
        print("-" * 78)
        for key in keys:
            label = f"{key[0]}/{key[1]}"
            b = kv_base.get(key, 0.0)
            m = kv_mask.get(key, 0.0)
            row = f"{label:<20}{b:>12.5f}{m:>12.5f}{m - b:>+12.5f}"
            if extra is not None:
                row += f"{extra.get(key, 0.0):>12.5f}"
            print(row)

    extra_overall = also.get("ir_only", ({}, {}))[0] if also else None
    extra_subset = also.get("ir_only", ({}, {}))[1] if also else None
    print_table("分支总体 mAP50-95(B)", base_overall, mask_overall, extra_overall)
    print_subset("分支 × 子集 mAP50-95(B)（both / only_rgb / only_ir）",
                 base_subset, mask_subset, extra_subset)

    dfuse = mask_overall.get("fuse", 0.0) - base_overall.get("fuse", 0.0)
    fuse_base = base_overall.get("fuse", 0.0)
    fuse_mask = mask_overall.get("fuse", 0.0)
    fuse_ironly = (extra_overall or {}).get("fuse", None)
    fuse_scaled = also.get("scaled", ({}, {}))[0].get("fuse", None) if also else None
    print("\n判读（fuse 头）：")
    if fuse_base > 0.3 and fuse_mask < 0.10:
        print(f"  掩码后 fuse 塌到 {fuse_mask:.4f}（基准 {fuse_base:.4f}，Δ={dfuse:+.4f}）")
        print("  -> 融合头**几乎没有可用的 IR 通路**：一旦失去 RGB 加性项即整体失效，")
        print("     本质上是一个「只读 RGB 半边」的检测头，IR 半边对其形同虚设。")
        if fuse_ironly is not None and fuse_ironly < 0.10:
            print(f"  且 ir_only（fused=ir_warp，幅值同量级）同样塌到 {fuse_ironly:.4f}，")
            print("     排除了「幅值减半」这一混淆：塌陷源于缺 IR 能力，而非尺度问题。")
        if fuse_scaled is not None:
            print(f"  幅值匹配版 drop_rgb_term_scaled 的 fuse={fuse_scaled:.4f}"
                  f"（{_MAG.get('gated_scaled_ratio', float('nan')):.2f}× 放大后）"
                  " —— 若它仍塌，则**确证**与幅值无关。")
    elif dfuse > 0.05:
        print("  去除 RGB 加性项后 fuse **显著变好** -> 融合确有利用 IR 的能力，")
        print("  常规情况下只是被 RGB 特征压制。")
    elif abs(dfuse) <= 0.05:
        print("  去除 RGB 加性项后 fuse **基本不变** -> 说明该融合分支的 RGB 加性项")
        print("  几乎不承载信息，融合贡献主要来自其余路径，需结合幅值占比复核。")
    else:
        print(f"  去除 RGB 加性项后 fuse 下降 {dfuse:.4f} -> 融合确在使用 RGB，")
        print("  但下降幅度即「RGB 通路的信息量」；对比 ir_only 可判断 IR 通路是否存活。")
    if _MAG:
        print(f"\n  幅值诊断：基准 fused RMS={_MAG.get('gated_normal_rms'):.3f}  "
              f"去RGB项后 RMS={_MAG.get('gated_raw_rms'):.3f}  "
              f"归一化系数={_MAG.get('gated_scaled_ratio'):.2f}×")

    amp_out = {}
    for li in FUSION_CONCAT_LAYERS:
        st = amp_final[li]
        if st["n"]:
            ra, rb = st["rgb"] / st["n"], st["ir"] / st["n"]
            amp_out[f"L{li}"] = {"rgb_rms": ra, "ir_rms": rb,
                                 "rgb_share": ra / (ra + rb), "ir_share": rb / (ra + rb)}
    payload = {
        "weights": str(args.weights),
        "ckpt_epoch": ckpt.get("epoch"),
        "mask_mode": args.mask_mode,
        "sanity_repl_vs_orig": sanity,
        "amp_rms_by_layer": amp_out,
        "map50_95_overall_base": base_overall,
        "map50_95_overall_masked": mask_overall,
        "map50_95_subset_base": {f"{k[0]}/{k[1]}": v for k, v in base_subset.items()},
        "map50_95_subset_masked": {f"{k[0]}/{k[1]}": v for k, v in mask_subset.items()},
        "delta_fuse_overall": dfuse,
    }
    if also:
        payload["map50_95_overall_ir_only"] = also["ir_only"][0]
        payload["map50_95_subset_ir_only"] = {f"{k[0]}/{k[1]}": v for k, v in also["ir_only"][1].items()}
    if "scaled" in also:
        payload["map50_95_overall_scaled"] = also["scaled"][0]
        payload["map50_95_subset_scaled"] = {f"{k[0]}/{k[1]}": v for k, v in also["scaled"][1].items()}
    if _MAG:
        payload["magnitude_diag"] = dict(_MAG)

    out = Path(args.out) if args.out else data_path.parent / "diag_fusion.json"
    try:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  结果已保存: {out}")
    except (PermissionError, OSError) as e:
        print(f"\n  [提示] 无法写入 {out}（{type(e).__name__}），结果已打印在终端。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())