import torch
import torch.nn as nn

"""
Weighted Cross-Modality Fusion (WCMF)

论文：FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection (Neurocomputing)
链接：https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 通过自适应权重实现跨模态加权融合
- 输出单张量
"""

__all__ = ['WCMF']


class WCMF(nn.Module):
    """Weighted Cross-Modality Fusion.

    Args:
        channels (int): 输入通道数（parse_model 注入）。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv_r1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channels, 2, 3, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _fusion(self, f1, f2, f_vec):
        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        Fr = self.conv_r1(x1)
        Fd = self.conv_d1(x2)
        f = torch.cat([Fr, Fd], dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        return self._fusion(Fr, Fd, f)
