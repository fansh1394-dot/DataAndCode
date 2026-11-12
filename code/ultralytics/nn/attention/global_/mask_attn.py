"""MaskAttention - 二进制掩码注意力
论文: MaskAttn-UNet: A Mask Attention-Driven Framework for Universal Low-Resolution Image Segmentation
会议: CVPR 2025
链接: https://arxiv.org/pdf/2503.10686
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MaskAttention']


class MaskAttention(nn.Module):
    """二进制掩码注意力模块 (Binary Mask Attention Module, MAM).

    通过固定种子的确定性二进制掩码抑制特征图中无信息区域,
    在保留 CNN 局部特征提取能力的同时捕捉长距离依赖,
    平衡局部细节与全局上下文信息。
    """

    def __init__(self, channels, size=None):
        super().__init__()
        self.channels = channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        # 确定性二进制掩码: 固定种子生成, 注册为 buffer
        # 空间尺寸变化时由 _init_mask 重新生成并自动跟随设备迁移
        self._init_mask(1, 1)

    def _init_mask(self, h, w, device=None):
        """生成固定种子的确定性二进制掩码并注册为 buffer."""
        hw = h * w
        gen = torch.Generator().manual_seed(42)
        binary_mask = torch.randint(0, 2, (1, hw), generator=gen)
        processed_mask = torch.where(
            binary_mask > 0.5,
            torch.tensor(0.0),
            torch.tensor(-float('inf')),
        )
        mask = processed_mask.unsqueeze(1).expand(-1, hw, -1)
        if device is not None:
            mask = mask.to(device)
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, C, H, W = x.size()
        if C != self.channels:
            raise ValueError(
                f"Input channel size ({C}) does not match "
                f"initialized channel size ({self.channels})."
            )

        hw = H * W
        # 空间尺寸变化时重新生成掩码 (固定种子保证确定性)
        if self.mask.size(-1) != hw:
            self._init_mask(H, W, device=x.device)

        # x: (B, C, H, W) -> (B, HW, C)
        x = x.view(B, C, hw).permute(0, 2, 1)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 缩放点积注意力 + 掩码 (mask 转 dtype 适配 AMP)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.channels ** 0.5)
        scores = scores + self.mask.to(scores.dtype)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # 残差连接 + 层归一化
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)

        # (B, HW, C) -> (B, C, H, W)
        return attention_output.view(B, C, H, W)


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
