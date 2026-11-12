"""Reliability-aware fusion for non-aligned RGB/IR features.

The module deliberately avoids pixel-wise feature addition.  It first predicts a
small sampling offset for each modality, then predicts two reliability logits and
mixes the warped features with a spatial softmax.  The offset is bounded so the
module remains a local correction, not a hidden global registration network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["ReliabilityAwareFusion"]


class ReliabilityAwareFusion(nn.Module):
    """Dynamic local alignment + reliability-gated RGB/IR fusion.

    Args:
        inc: two input channel counts ``[c_rgb, c_ir]``.
        ouc: output channel count.
        max_offset: maximum normalized grid displacement in feature pixels.
    """

    def __init__(self, inc, ouc, max_offset: float = 0.08):
        super().__init__()
        if not isinstance(inc, (list, tuple)) or len(inc) != 2:
            raise ValueError(f"ReliabilityAwareFusion expects inc=[c_rgb,c_ir], got {inc}")
        self.c_rgb, self.c_ir, self.ouc = int(inc[0]), int(inc[1]), int(ouc)
        self.max_offset = float(max_offset)
        hidden = max(min(self.ouc, 128), 16)
        self.rgb_proj = Conv(self.c_rgb, self.ouc, 1)
        self.ir_proj = Conv(self.c_ir, self.ouc, 1)
        self.offset = nn.Sequential(
            nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, 3, padding=1),
        )
        self.reliability = nn.Sequential(
            nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1),
        )
        self.out = Conv(self.ouc, self.ouc, 3)
        # Start from an identity-like, unbiased fusion state.  Random offsets at
        # initialization can move tiny UAV features away before the detector has
        # learned any semantics; zero offsets + 0.5/0.5 reliability are a much
        # safer baseline and make the module's contribution easier to ablate.
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.last_offset_abs_mean = None
        self.last_reliability_mean = None

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
            raise ValueError(f"ReliabilityAwareFusion expects two feature maps, got {type(xs).__name__}")
        rgb, ir = xs
        if rgb.shape[-2:] != ir.shape[-2:]:
            raise ValueError(f"feature spatial sizes differ: RGB={rgb.shape[-2:]} IR={ir.shape[-2:]}")
        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        pair = torch.cat((rgb, ir), dim=1)
        raw_offset = self.offset(pair).tanh() * self.max_offset
        b, _, h, w = raw_offset.shape
        base = self._base_grid(b, h, w, pair.device, pair.dtype)
        # grid_sample uses x/y ordering; RGB and IR receive independent offsets.
        rgb_grid = (base + raw_offset[:, 0:2].permute(0, 2, 3, 1)).clamp(-1, 1)
        ir_grid = (base + raw_offset[:, 2:4].permute(0, 2, 3, 1)).clamp(-1, 1)
        rgb_warp = F.grid_sample(rgb, rgb_grid, mode="bilinear", align_corners=True, padding_mode="border")
        ir_warp = F.grid_sample(ir, ir_grid, mode="bilinear", align_corners=True, padding_mode="border")
        weights = self.reliability(torch.cat((rgb_warp, ir_warp), dim=1)).softmax(dim=1)
        # Scalar diagnostics only (no large activation retention).  These are
        # detached so logging/inspection cannot alter the autograd graph.
        self.last_offset_abs_mean = raw_offset.detach().abs().mean()
        self.last_reliability_mean = weights.detach().mean(dim=(0, 2, 3))
        fused = weights[:, 0:1] * rgb_warp + weights[:, 1:2] * ir_warp
        return self.out(fused)
