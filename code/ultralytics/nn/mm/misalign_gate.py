# -*- coding: utf-8 -*-
"""Misalignment-aware gated concatenation for RGB/IR fusion necks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MisalignGatedConcat"]


class MisalignGatedConcat(nn.Module):
    """Per-sample gated concat: ``out = cat([rgb, alpha * ir], dim=1)``.

    ``alpha`` is produced from a global cross-modal descriptor so a single forward
    pass suffices (no dependency on the detection heads).  The block is
    structure-preserving: channel layout and total channels match a plain
    ``Concat``, so it can replace a fusion ``Concat`` node without any other
    change to the neck.

    Args:
        inc: ``[c_rgb, c_ir]`` input channel counts (from ``ch[f]``).
        alpha_min: lower bound of ``alpha``; keeps the IR branch from being
            switched off entirely (matches the scan in the experiment log).
        hidden: width of the gating MLP.
    """

    def __init__(self, inc, alpha_min: float = 0.2, hidden: int = 32):
        super().__init__()
        if not isinstance(inc, (list, tuple)) or len(inc) != 2:
            raise ValueError(f"MisalignGatedConcat expects inc=[c_rgb, c_ir], got {inc!r}")
        self.c_rgb = int(inc[0])
        self.c_ir = int(inc[1])
        if self.c_rgb != self.c_ir:
            # The descriptor uses channel-wise cosine similarity, which needs equal width.
            raise ValueError(
                f"MisalignGatedConcat requires c_rgb == c_ir, got {self.c_rgb} vs {self.c_ir}"
            )
        self.alpha_min = float(alpha_min)
        if not 0.0 <= self.alpha_min < 1.0:
            raise ValueError(f"alpha_min must be in [0, 1), got {alpha_min}")

        h = max(8, int(hidden))
        self.fc = nn.Sequential(
            nn.Linear(self.c_rgb * 2 + 2, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        # Weight: small random (not zeros) so the gate is not perfectly constant across
        # samples and the anchor term has a non-zero gradient from step 0 (P0/L_gate lesson).
        nn.init.normal_(self.fc[-1].weight, mean=0.0, std=1e-2)
        # Bias: alpha ~= 0.9 at init.  Starting near "use most of IR" means training begins
        # close to the plain-concat baseline instead of the arbitrary 0.6 midpoint
        # (alpha_min + (1-alpha_min)/2), and the sigmoid is still far enough from
        # saturation to keep a usable gradient.
        with torch.no_grad():
            self.fc[-1].bias.fill_(2.2)

        # Diagnostics / loss hooks (not part of state_dict).
        self._alpha_raw = None
        self.last_alpha_mean = None
        self.last_alpha_std = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("MisalignGatedConcat expects [rgb, ir] feature maps")
        rgb, ir = x
        if rgb.shape[1] != self.c_rgb or ir.shape[1] != self.c_ir:
            raise ValueError(
                f"channel mismatch: expected ({self.c_rgb},{self.c_ir}), got ({rgb.shape[1]},{ir.shape[1]})"
            )
        if rgb.shape[-2:] != ir.shape[-2:]:
            raise ValueError(f"spatial mismatch: {rgb.shape[-2:]} vs {ir.shape[-2:]}")

        g_rgb = F.adaptive_avg_pool2d(rgb, 1).flatten(1)
        g_ir = F.adaptive_avg_pool2d(ir, 1).flatten(1)
        cos = F.cosine_similarity(g_rgb, g_ir, dim=1, eps=1e-6).unsqueeze(1)
        diff = (g_rgb - g_ir).abs().mean(dim=1, keepdim=True)
        desc = torch.cat([g_rgb, g_ir, cos, diff], dim=1)

        logit = self.fc(desc)
        alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(logit)  # (B, 1)
        a = alpha.view(-1, 1, 1, 1)

        self._alpha_raw = alpha
        with torch.no_grad():
            self.last_alpha_mean = float(alpha.mean())
            self.last_alpha_std = float(alpha.std()) if alpha.numel() > 1 else 0.0

        return torch.cat([rgb, ir * a], dim=1)
