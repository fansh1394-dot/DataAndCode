"""Deformable cross-modal fusion for non-aligned RGB/IR features (改动 C4).

Motivation
----------
Both ``ReliabilityAwareFusion`` (convex mix) and ``GatedReliabilityFusion`` (gated
residual) align the two modalities through a **single dense offset field** per
modality, i.e. one ``(dx, dy)`` displacement per output pixel, applied via
``F.grid_sample``.  Two structural limits follow:

1. A single sample per location cannot represent *multi-hypothesis* misalignment.
   Under strong parallax / independent sensor motion the correct correspondence is
   ambiguous (a UAV can be matched to several background patches), so a unimodal
   offset estimate averages the hypotheses and blurs both tiny targets and edges.
2. ``grid_sample(mode="bilinear")`` resamples with a 2x2 kernel.  For UAV targets
   that occupy only a few feature pixels, this interpolation washes out exactly the
   high-frequency evidence the detector needs.

The measured "pseudo-fusion" symptom (``diag_fusion_attribution.py``: removing the
RGB term collapses ``fuse`` from ~0.71 to ~0.006) survived both a convex and a
gated residual operator **and** the B4-b supervision change, which indicates the
bottleneck is the *alignment operator itself*, not the mixing or the supervision.

This module attacks that directly: it replaces the single dense offset with a
**learnable deformable cross-attention** -- for every output location it predicts
``K`` sampling points and ``K`` attention weights on the *other* modality, then
aggregates them with a softmax:

    fused = base + g * (attend(ir -> rgb) - base)   with base = rgb_warped / mean

Concretely the IR feature is sampled at ``K`` learned offsets around each location
(multi-hypothesis alignment, so an ambiguous correspondence is kept as several
weighted candidates rather than averaged away) and the resulting ``K`` samples are
combined by content-dependent attention weights (so a spurious sample gets a low
weight instead of polluting the mix).

Interface compatibility
------------------------
The signature mirrors ``GatedReliabilityFusion`` exactly for the first four args
(``inc, ouc, max_offset`` plus the ``base`` selector) so ``parse_model`` and the
diagnostic scripts (``diag_fusion_attribution.py`` / ``diag_gate_stats.py``) keep
working unchanged.  ``num_points`` (``K``) is the single new hyper-parameter and
defaults to ``4`` (the deformable-conv-v2 default).

Safety
------
* Offsets are ``tanh``-bounded to ``+/- max_offset`` so the module stays a *local*
  correction and never degenerates into a hidden global registration network.
* The attention-logit head is zero-initialised; the offsets are zero-initialised.
  Hence at ``init`` the ``K`` samples coincide and the attention is uniform, so the
  module starts from the *neutral mean* state -- identical in spirit to the safe
  init of the two existing operators.
* ``pyramid_level``-style multi-scale sampling is intentionally omitted: the three
  fusion nodes already sit at P3/P4/P5, so scale is handled by the neck.

Diagnostics
-----------
``last_offset_abs_mean``, ``last_attn_entropy`` and ``last_deform_spread`` are
detached scalars for logging.  ``_attn_raw`` keeps the non-detached attention map
so an activation regulariser (cf. ``L_gate``) can reach it if needed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["DeformableCrossModalFusion"]


class DeformableCrossModalFusion(nn.Module):
    """K-point learnable deformable cross-attention fusion of RGB/IR features.

    Args:
        inc: two input channel counts ``[c_rgb, c_ir]``.
        ouc: output channel count.
        max_offset: maximum normalized grid displacement in feature pixels.
        num_points: number of deformable sampling points ``K`` per location.
        base: residual base: ``"rgb"`` (default), ``"ir"`` or ``"mean"``.
        attn: attention granularity, ``"channel"`` (default) or ``"pixel"``.
    """

    def __init__(
        self,
        inc,
        ouc,
        max_offset: float = 0.08,
        num_points: int = 4,
        base: str = "rgb",
        attn: str = "channel",
    ):
        super().__init__()
        if not isinstance(inc, (list, tuple)) or len(inc) != 2:
            raise ValueError(f"DeformableCrossModalFusion expects inc=[c_rgb,c_ir], got {inc}")
        self.c_rgb, self.c_ir, self.ouc = int(inc[0]), int(inc[1]), int(ouc)
        self.max_offset = float(max_offset)
        self.num_points = int(num_points)
        self.base_mode = str(base).lower()
        self.attn_mode = str(attn).lower()
        if self.num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {self.num_points}")
        if self.base_mode not in ("rgb", "ir", "mean"):
            raise ValueError(f"base must be 'rgb'/'ir'/'mean', got {self.base_mode!r}")
        if self.attn_mode not in ("channel", "pixel"):
            raise ValueError(f"attn must be 'channel'/'pixel', got {self.attn_mode!r}")

        hidden = max(min(self.ouc, 128), 16)
        self.rgb_proj = Conv(self.c_rgb, self.ouc, 1)
        self.ir_proj = Conv(self.c_ir, self.ouc, 1)

        # --- Deformable offset head: predicts K*(dx,dy) displacements for IR sampling ---
        # Output channels = 2*K; the offsets are predicted from the CONCATENATED pair so
        # the displacement can depend on *both* modalities (cross-modal, not IR-only).
        self.offset = nn.Sequential(
            nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * self.num_points, 3, padding=1),
        )

        # --- Attention weight head: K content-dependent weights per location ---
        # Input = pair (2*ouc) + 2*K offsets; output = K logits.
        attn_out = self.num_points
        self.attn = nn.Sequential(
            nn.Conv2d(self.ouc * 2 + 2 * self.num_points, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, attn_out, 1),
        )

        # --- Optional gate for the residual magnitude (keeps the "may ignore IR" escape) ---
        gate_out = self.ouc
        self.gate = nn.Sequential(
            nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, gate_out, 1),
        )
        self.out = Conv(self.ouc, self.ouc, 3)

        # Safe init: identity alignment + neutral gate.  Zero offsets mean the K samples
        # coincide (fully-overlapping deformable points).  The attention-logit head uses a
        # SMALL RANDOM init (not zeros): with all logits equal the softmax is uniform AND
        # the gradient w.r.t. the logit head is exactly zero (a saddle point -- the same
        # lesson learned from the P0 L_gate fix), so the attention could never learn to
        # re-weight the K samples.  A tiny perturbation (|delta| ~ 1e-3) keeps the initial
        # attention nearly uniform while giving it a non-zero gradient from step 0.
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        nn.init.normal_(self.attn[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.attn[-1].bias, mean=0.0, std=1e-3)
        # Gate uses a small random init (NOT zeros) for the same anti-saddle reason.
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.gate[-1].bias, mean=0.0, std=1e-3)

        self.last_offset_abs_mean = None
        self.last_attn_entropy = None
        self.last_deform_spread = None
        self.last_gate_mean = None
        self.last_gate_std = None
        self._gate_raw = None
        self._attn_raw = None

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
            raise ValueError(f"DeformableCrossModalFusion expects two feature maps, got {type(xs).__name__}")
        rgb, ir = xs
        if rgb.shape[-2:] != ir.shape[-2:]:
            raise ValueError(f"feature spatial sizes differ: RGB={rgb.shape[-2:]} IR={ir.shape[-2:]}")

        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        b, c, h, w = rgb.shape
        pair = torch.cat((rgb, ir), dim=1)

        if self.max_offset <= 0.0:
            # ---- 对齐模式（max_offset=0）：跳过位移预测与 grid_sample 重采样 ----
            # 前提：两模态几何已配准。此时非零位移只会引入双线性插值的低通损失
            # （本数据实测：offset 均值 3/9/12 px、注意力熵 ≈0.99、K 点退化为均匀均值，
            #  fuse 头相对朴素 Concat 为净负收益 −0.0035）。
            # 这里令 K 个采样点重合于原坐标，ir_attended 恒等于 ir_proj(ir)，
            # 模块退化为「投影 + 门控残差」。offset/attn 子模块结构保留，
            # 换到非对齐数据时把 max_offset 改回 >0 即可恢复完整可变形行为。
            offsets = torch.zeros((b, self.num_points, 2, h, w), device=pair.device, dtype=pair.dtype)
            attn = torch.full((b, self.num_points, h, w), 1.0 / self.num_points,
                              device=pair.device, dtype=pair.dtype)
            ir_sampled = None
            ir_attended = ir
        else:
            # ---- 1) predict K offsets for sampling IR around each location ----
            offsets = torch.tanh(self.offset(pair)) * self.max_offset  # [B, 2K, H, W]
            # reshape to [B, K, 2, H, W] -> per-point (dx, dy)
            offsets = offsets.view(b, self.num_points, 2, h, w)

            base_grid = self._base_grid(b, h, w, pair.device, pair.dtype)  # [B, H, W, 2]
            # sample_grid: [B, K, H, W, 2] = base + per-point offset (x/y ordering)
            sample_grid = base_grid.unsqueeze(1) + offsets.permute(0, 1, 3, 4, 2)  # [B,K,H,W,2]
            sample_grid = sample_grid.clamp(-1, 1)

            # ---- 2) deformable sampling of IR at K points (multi-hypothesis alignment) ----
            ir_expand = ir.unsqueeze(1).expand(b, self.num_points, c, h, w)  # [B,K,C,H,W]
            ir_expand = ir_expand.reshape(b * self.num_points, c, h, w)
            grid_flat = sample_grid.reshape(b * self.num_points, h, w, 2)
            ir_sampled = F.grid_sample(
                ir_expand, grid_flat, mode="bilinear", align_corners=True, padding_mode="border"
            )  # [B*K, C, H, W]
            ir_sampled = ir_sampled.view(b, self.num_points, c, h, w)  # [B,K,C,H,W]

            # ---- 3) content-dependent attention over the K samples ----
            attn_in = torch.cat(
                (pair, offsets.reshape(b, 2 * self.num_points, h, w)), dim=1
            )  # [B, 2C+2K, H, W]
            attn_logits = self.attn(attn_in)  # [B, K, H, W]
            attn = attn_logits.softmax(dim=1)  # [B, K, H, W], sums to 1 over K

            # weighted sum of the K IR samples -> [B, C, H, W]
            ir_attended = (ir_sampled * attn.unsqueeze(2)).sum(dim=1)

        # ---- 4) gated residual fusion (same form as GatedReliabilityFusion) ----
        gate = torch.sigmoid(self.gate(pair))
        # 通路级 dropout（D2-P1）：逐样本把 gate 强置为 1，使 fused = base + 1*(ir_attended-rgb)。
        # 当 base_mode="rgb"（默认）时等价于 fused = ir_attended，即完全消去 RGB 加性项。
        #
        # 为什么作用在这里而不是融合节点输入端（既有 mm_fusion_rgb_mask）：
        # fused = (1-gate)*rgb + gate*ir_attended，其中 rgb 是经 rgb_proj 投影后的**内部**
        # 变量。在输入端置零时 rgb_proj(0)=bias≠0，RGB 分量依然存在、gate 仍可依赖它，
        # 屏蔽被绕过（A4/S3 在 p=0.1/0.2 下均未破局）。
        # 直接消去 fused 表达式中的 rgb 项，物理上无法绕过；且与诊断探针
        # diag_fusion_attribution 的 drop_rgb_term 在同一数学位置、同一语义，
        # 训练扰动与评测探针构成同构因果链。
        #
        # 逐样本粒度：同一 batch 内混有「含 RGB / 纯 IR」两种 fused，fuse 头必须在同一次
        # forward 内同时支持两条通路，无法把 RGB 当捷径专精。
        # 只在 train 模式且显式置位时激活，故既有权重推理行为逐位不变。
        _gd_mask = getattr(self, "_mm_gate_drop_mask", None)
        if self.training and torch.is_tensor(_gd_mask) and _gd_mask.numel() == b and bool(_gd_mask.any()):
            _ones = torch.ones_like(gate)
            gate = torch.where(_gd_mask.view(b, 1, 1, 1), _ones, gate)
        if self.base_mode == "mean":
            base_feat = 0.5 * (rgb + ir_attended)
        elif self.base_mode == "ir":
            base_feat = ir_attended
        else:
            base_feat = rgb
        fused = base_feat + gate * (ir_attended - rgb)

        # ---- diagnostics (detached) ----
        self.last_offset_abs_mean = offsets.detach().abs().mean()
        self.last_gate_mean = gate.detach().mean()
        self.last_gate_std = gate.detach().std()
        with torch.no_grad():
            p = attn.clamp_min(1e-9)
            # K==1 => log(K)=0; entropy is trivially 0 (single sample) -> guard the divide.
            if self.num_points > 1:
                ent = -(p * p.log()).sum(dim=1).mean() / math.log(self.num_points)
            else:
                ent = torch.zeros((), device=rgb.device)
        self.last_attn_entropy = ent  # normalized to [0,1]: 1 = uniform, 0 = single point
        # spread = std of the K sampled points across the point axis (alignment diversity).
        # For K==1 std over a single sample is undefined, so report 0 explicitly.
        if ir_sampled is not None and self.num_points > 1:
            self.last_deform_spread = ir_sampled.detach().std(dim=1).mean()
        else:
            self.last_deform_spread = torch.zeros((), device=rgb.device)

        # non-detached refs for optional regularisation (kept out of state_dict)
        self._gate_raw = gate
        self._attn_raw = attn
        return self.out(fused)
