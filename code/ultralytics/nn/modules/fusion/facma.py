import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Frequency-Aware Cross-Modality Attention (FACMA)

论文：FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection (Neurocomputing)
链接：https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 尺寸自适应改造：SFCA 内部 DCT 权重依赖固定尺寸，forward 中先 adaptive_avg_pool 到参考尺寸再 interpolate 回原尺寸
- 双输出→单输出：两路增强特征相加
"""

__all__ = ['FACMA']


def _get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    return result * math.sqrt(2)


def _get_dct_weights(width, height, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i * c_part:(i + 1) * c_part, t_x, t_y] = \
                    _get_1d_dct(t_x, u_x, width) * _get_1d_dct(t_y, v_y, height)
    return dct_weights


class _FCABlock(nn.Module):
    """FcaNet: Frequency Channel Attention Networks."""

    def __init__(self, channel, width, height, fidx_u, fidx_v, reduction=16):
        super().__init__()
        mid_channel = channel // reduction
        self.register_buffer('pre_computed_dct_weights',
                             _get_dct_weights(width, height, channel, fidx_u, fidx_v))
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2, 3])
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class _SFCA(nn.Module):
    def __init__(self, in_channel, width, height, fidx_u, fidx_v):
        super().__init__()
        fidx_u = [temp_u * (width // 8) for temp_u in fidx_u]
        fidx_v = [temp_v * (width // 8) for temp_v in fidx_v]
        self.FCA = _FCABlock(in_channel, width, height, fidx_u, fidx_v)
        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        F_fca = self.FCA(x)
        con = self.norm(self.conv1(x))
        F_con = x * con
        return F_fca + F_con


class FACMA(nn.Module):
    """Frequency-Aware Cross-Modality Attention.

    Args:
        channels (int): 输入通道数（parse_model 注入）。
        ref_size (int): DCT 参考空间尺寸（默认 64）。
    """

    def __init__(self, channels, ref_size=64):
        super().__init__()
        self.ref_size = ref_size
        fidx_u = [0, 1]
        fidx_v = [0, 1]
        self.sfca_a = _SFCA(channels, ref_size, ref_size, fidx_u, fidx_v)
        self.sfca_b = _SFCA(channels, ref_size, ref_size, fidx_u, fidx_v)

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        B, C, H, W = x1.shape
        ref = self.ref_size

        # 尺寸自适应：池化到参考尺寸 → SFCA 处理 → 插值回原尺寸
        x1_r = F.adaptive_avg_pool2d(x1, (ref, ref))
        x2_r = F.adaptive_avg_pool2d(x2, (ref, ref))

        out_b = self.sfca_a(x2_r)
        out_b = F.interpolate(out_b, size=(H, W), mode='bilinear', align_corners=False)
        out_a = self.sfca_b(x1_r)
        out_a = F.interpolate(out_a, size=(H, W), mode='bilinear', align_corners=False)

        # 跨模态增强
        out_a = x1 * out_a
        out_b = x2 * out_b

        # 单输出
        return out_a + out_b
