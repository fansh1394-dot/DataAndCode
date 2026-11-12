"""Gated reliability fusion for non-aligned RGB/IR features.

Motivation
----------
``ReliabilityAwareFusion`` fuses as ``fused = w0*rgb + w1*ir`` with ``w = softmax(...)``,
i.e. a *convex* combination (``w0 + w1 = 1``).  A convex combination is provably bounded
by its two inputs, so the fused branch can never exceed the stronger modality, and the
softmax weight tends to collapse toward the RGB side -- the "pseudo-fusion" symptom
measured by ``diag_fusion_attribution.py --mask-mode drop_rgb_term`` (fuse drops from
~0.71 to ~0.15 when the RGB additive term is removed).

This module keeps the alignment machinery but replaces the convex mix with an explicit
gated *residual*:

    fused = base + g * (ir_warp - rgb_warp),   g = sigmoid(gate([rgb, ir]))

With ``base = rgb_warp`` (mode="rgb") the fusion can leave the convex hull: when the IR
branch carries complementary information the gate opens (``g -> 1``) and the output moves
toward IR; when IR is uninformative the gate closes (``g -> 0``) and the output reduces to
RGB.  The gate is zero-initialised at the logit level so training starts from the neutral
``g == 0.5`` state (``fused = (rgb + ir) / 2``), which is a safe, unbiased baseline.

A ``mean(gate)`` diagnostic is exposed via ``last_gate_mean`` so callers can detect gate
degeneracy (``g -> 0`` would mean the module silently stopped using IR).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["GatedReliabilityFusion"]


class GatedReliabilityFusion(nn.Module):
    """Dynamic local alignment + explicitly gated RGB/IR residual fusion.

    Args:
        inc: two input channel counts ``[c_rgb, c_ir]``.
        ouc: output channel count.
        max_offset: maximum normalized grid displacement in feature pixels.
        base: which modality forms the residual base: ``"rgb"`` (default, A1) or ``"mean"`` (A2).
        gate: gate granularity, ``"channel"`` (default) or ``"pixel"``.
    """

    def __init__(self, inc, ouc, max_offset: float = 0.08, base: str = "rgb", gate: str = "channel"):
        super().__init__()
        if not isinstance(inc, (list, tuple)) or len(inc) != 2:
            raise ValueError(f"GatedReliabilityFusion expects inc=[c_rgb,c_ir], got {inc}")
        self.c_rgb, self.c_ir, self.ouc = int(inc[0]), int(inc[1]), int(ouc)
        self.max_offset = float(max_offset)
        base = str(base).lower()
        gate = str(gate).lower()
        if base not in ("rgb", "mean"):
            raise ValueError(f"base must be 'rgb' or 'mean', got {base!r}")
        if gate not in ("channel", "pixel"):
            raise ValueError(f"gate must be 'channel' or 'pixel', got {gate!r}")
        self.base_mode = base
        self.gate_mode = gate
        hidden = max(min(self.ouc, 128), 16)
        self.rgb_proj = Conv(self.c_rgb, self.ouc, 1)
        self.ir_proj = Conv(self.c_ir, self.ouc, 1)
        self.offset = nn.Sequential(
            nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, 3, padding=1),
        )
        gate_out = self.ouc if gate == "pixel" else self.ouc
        self.gate = nn.Sequential(
            nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, gate_out, 1),
        )
        self.out = Conv(self.ouc, self.ouc, 3)
        # Zero-init the offset head -> identity alignment at start (same rationale as
        # ReliabilityAwareFusion: random early offsets would displace tiny UAV features).
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        # Gate 末层：**小随机**初始化（而非全零）。
        # 原因（P0 实测）：全零 -> gate 张量处处相等 -> std(g) 在常量点的梯度为 0，
        # 任何依赖 std/方差的激活正则（L_gate）在该点都是**零梯度鞍点**，门控永远
        # 停在 g≡0.5（diag_gate_reg_gradient.py 已复现：grad=0）。
        # 用小随机初始化打破对称，使起始 g 略微偏离 0.5（|Δ|<0.02，仍近似中性），
        # L_gate 从第 0 步就有非零梯度，门控能被激活。
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.gate[-1].bias, mean=0.0, std=1e-3)
        self.last_offset_abs_mean = None
        self.last_gate_mean = None
        # P0 (L_gate)：门控标准差诊断量。detach 版仅供日志/统计；``_gate_raw`` 保留
        # 非 detach 引用，供 DualHeadFusionCriterion 的激活正则真正取到梯度。
        self.last_gate_std = None
        self._gate_raw = None

    @staticmethod
    def _base_grid(b, h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != 2:
            raise ValueError(f"GatedReliabilityFusion expects two feature maps, got {type(xs).__name__}")
        rgb, ir = xs
        if rgb.shape[-2:] != ir.shape[-2:]:
            raise ValueError(f"feature spatial sizes differ: RGB={rgb.shape[-2:]} IR={ir.shape[-2:]}")
        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        pair = torch.cat((rgb, ir), dim=1)
        raw_offset = self.offset(pair).tanh() * self.max_offset
        b, _, h, w = raw_offset.shape
        base = self._base_grid(b, h, w, pair.device, pair.dtype)
        rgb_grid = (base + raw_offset[:, 0:2].permute(0, 2, 3, 1)).clamp(-1, 1)
        ir_grid = (base + raw_offset[:, 2:4].permute(0, 2, 3, 1)).clamp(-1, 1)
        rgb_warp = F.grid_sample(rgb, rgb_grid, mode="bilinear", align_corners=True, padding_mode="border")
        ir_warp = F.grid_sample(ir, ir_grid, mode="bilinear", align_corners=True, padding_mode="border")
        # gate: channel-wise [B, C, 1, 1] or pixel-wise [B, C, H, W]; broadcast over the other dims.
        gate = torch.sigmoid(self.gate(torch.cat((rgb_warp, ir_warp), dim=1)))
        if self.base_mode == "mean":
            base_feat = 0.5 * (rgb_warp + ir_warp)
        else:
            base_feat = rgb_warp
        fused = base_feat + gate * (ir_warp - rgb_warp)
        # Scalar diagnostics only; detached so inspection cannot touch autograd.
        self.last_offset_abs_mean = raw_offset.detach().abs().mean()
        self.last_gate_mean = gate.detach().mean()
        # L_gate 需要真正回传梯度到 gate 分支，故额外保留非 detach 引用（不进 state_dict）。
        self.last_gate_std = gate.detach().std()
        self._gate_raw = gate
        return self.out(fused)
