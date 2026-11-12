import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState

"""
Fourier Convolution Block (FCB)

论文：Fourier Convolution Block with global receptive field for MRI reconstruction (2025 一区TOP)
链接：https://www.sciencedirect.com/science/article/pii/S1361841524002743

说明：
- 接口：forward(x) 单输入单输出（输出通道 = 输入通道）
- 尺寸自适应改造：权重参数在固定参考尺寸下初始化，forward 中通过插值适配当前 FFT 尺寸
- 可学习频域滤波器，具有全局感受野
"""

__all__ = ['FCB']


def _complexinit(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols = weights_real.shape
    fan_in = input_chs
    fan_out = output_chs

    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState()
    kernel_shape = weights_real.shape
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    weights_real.data = torch.Tensor(weight_real)
    weights_imag.data = torch.Tensor(weight_imag)


class FCB(nn.Module):
    """Fourier Convolution Block.

    Args:
        channels (int): 输入/输出通道数（parse_model 注入）。
        ref_size (int): 频域权重参考空间尺寸（默认 160）。
        stride (int): 下采样步长（默认 1，=2 时下采样）。
        init (str): 权重初始化方式（'he' 或 'glorot'）。
    """

    def __init__(self, channels, ref_size=32, stride=1, init='he'):
        super().__init__()
        self.ref_size = ref_size
        self.stride = stride

        # 频域权重（实部和虚部），参考尺寸下初始化
        self.weights_real = nn.Parameter(
            torch.Tensor(1, channels, ref_size, ref_size // 2 + 1))
        self.weights_imag = nn.Parameter(
            torch.Tensor(1, channels, ref_size, ref_size // 2 + 1))
        _complexinit(self.weights_real, self.weights_imag, init)

    def forward(self, x):
        B, C, H, W = x.shape

        # FFT
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))

        # 插值权重到当前 FFT 尺寸
        fft_h, fft_w = x_fft.shape[-2], x_fft.shape[-1]
        w_r = F.interpolate(self.weights_real, size=(fft_h, fft_w), mode='bilinear', align_corners=False)
        w_i = F.interpolate(self.weights_imag, size=(fft_h, fft_w), mode='bilinear', align_corners=False)

        # 复数乘法
        y_r = x_fft.real * w_r - x_fft.imag * w_i
        y_i = x_fft.real * w_i + x_fft.imag * w_r

        # IFFT
        x_out = torch.fft.irfftn(torch.complex(y_r, y_i), s=(H, W), dim=(-2, -1))

        if self.stride == 2:
            x_out = x_out[..., ::2, ::2]

        return x_out
