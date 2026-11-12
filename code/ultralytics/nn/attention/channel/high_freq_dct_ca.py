import torch
import torch.nn as nn
import torch.nn.functional as F

"""
High Frequency DCT Channel Attention (HighFreqDCTChannelAttention)

论文：HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection (AAAI 2025)
链接：https://arxiv.org/abs/2501.03775

说明：
- 接口：forward(x) 单输入单输出（通道注意力，特征 × 注意力权重）
- 通过 DCT 频域变换提取高频信息，双池化统计后生成通道注意力权重
- 懒加载：torch_dct 仅在 forward() 时导入，不使用时不需要安装
"""

__all__ = ['HighFreqDCTChannelAttention']


class HighFreqDCTChannelAttention(nn.Module):
    """High Frequency DCT Channel Attention.

    Args:
        channels (int): 输入通道数（parse_model 注入）。
        pool_size (int | tuple): 池化后的空间尺寸（默认 8）。
        freq_ratio (float | tuple): 高频保留比例（默认 0.25）。
    """

    def __init__(self, channels, pool_size=8, freq_ratio=0.25):
        super().__init__()

        self.channels = channels

        # 统一处理 pool_size 为 tuple
        if isinstance(pool_size, int):
            self.ph, self.pw = pool_size, pool_size
        else:
            self.ph, self.pw = pool_size

        # 统一处理 freq_ratio 为 tuple
        if isinstance(freq_ratio, (int, float)):
            self.freq_ratio = (freq_ratio, freq_ratio)
        else:
            self.freq_ratio = freq_ratio

        # 分组 1×1 卷积
        self.reduce_conv = nn.Conv2d(channels, channels, 1, groups=min(32, channels), bias=False)
        self.excite_conv = nn.Conv2d(channels, channels, 1, groups=min(32, channels), bias=False)

        self.relu = nn.ReLU()

    def _build_freq_mask(self, h, w, ratio):
        """构造频域高频掩码：抑制低频，保留高频。"""
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        mask = torch.ones((h, w), requires_grad=False)
        mask[:h0, :w0] = 0
        return mask

    def forward(self, x):
        # 懒加载：仅在首次 forward 时导入 torch_dct
        try:
            import torch_dct as DCT
        except ImportError as e:
            raise ImportError(
                "HighFreqDCTChannelAttention 需要 torch_dct，请运行: pip install torch_dct") from e

        n, c, h, w = x.size()

        # DCT 得到频域图
        freq_map = DCT.dct_2d(x, norm='ortho')

        # 构造高频掩码
        freq_mask = self._build_freq_mask(h, w, self.freq_ratio).to(x.device)
        freq_mask = freq_mask.view(1, 1, h, w).expand_as(freq_map)

        # 高频增强
        masked_freq = freq_map * freq_mask
        spatial_highfreq = DCT.idct_2d(masked_freq, norm='ortho')

        # 双池化统计
        max_pool = F.adaptive_max_pool2d(spatial_highfreq, (self.ph, self.pw))
        avg_pool = F.adaptive_avg_pool2d(spatial_highfreq, (self.ph, self.pw))

        max_pool = torch.sum(self.relu(max_pool), dim=[2, 3]).view(n, c, 1, 1)
        avg_pool = torch.sum(self.relu(avg_pool), dim=[2, 3]).view(n, c, 1, 1)

        # 通道注意力
        channel_desc = self.reduce_conv(max_pool) + self.reduce_conv(avg_pool)
        channel_weight = torch.sigmoid(self.excite_conv(channel_desc))

        return x * channel_weight
