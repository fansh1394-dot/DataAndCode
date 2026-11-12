import torch
import torch.nn as nn

"""
Cross-View Feature Interaction (CVIM)

论文：Multi-Level Feature Fusion Network for Lightweight Stereo Image Super-Resolution (CVPR 2024)
链接：https://arxiv.org/pdf/2405.05497

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 跨视角特征交互，通过注意力矩阵实现双模态信息交换
- 输出单张量（x_l + F_r2l + x_r + F_l2r）
"""

__all__ = ['CVIM']


class CVIM(nn.Module):
    """Cross-View Feature Interaction.

    Args:
        channels (int): 输入通道数（parse_model 注入）。
    """

    def __init__(self, channels):
        super().__init__()
        c = channels
        self.scale = c ** -0.5

        self.l_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.l_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        Q_l = self.l_proj1(x1).permute(0, 2, 3, 1)
        Q_r_T = self.r_proj1(x2).permute(0, 2, 1, 3)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        V_l = self.l_proj2(x1).permute(0, 2, 3, 1)
        V_r = self.r_proj2(x2).permute(0, 2, 3, 1)

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)

        F_r2l = self.l_proj3(F_r2l.permute(0, 3, 1, 2))
        F_l2r = self.r_proj3(F_l2r.permute(0, 3, 1, 2))

        return x1 + F_r2l + x2 + F_l2r
