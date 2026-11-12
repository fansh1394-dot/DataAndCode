import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
Bilateral Information Exchange (BIE)

论文：Bilateral Event Mining and Complementary for Event Stream Super-Resolution (CVPR 2024)
链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Bilateral_Event_Mining_and_Complementary_for_Event_Stream_Super-Resolution_CVPR_2024_paper.pdf

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 三输入→双输入改造：共享特征 x_s = (x_1 + x_2) * 0.5
- 输出单张量（两路输出之和）
"""

__all__ = ['BIE']


def _initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class _LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return _LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class _ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        _initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class BIE(nn.Module):
    """Bilateral Information Exchange.

    Args:
        channels (int): 输入通道数（parse_model 注入）。
    """

    def __init__(self, channels):
        super().__init__()
        nf = channels

        self.conv1 = _ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
        self.convf2 = self.convf1

        self.scale = nf ** -0.5
        self.norm_s = _LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)

        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)

        _initialize_weights(
            [self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2=None, x_s=None):
        if x_2 is None and isinstance(x_1, (list, tuple)):
            x_1, x_2 = x_1

        # 三输入→双输入：共享特征取两模态平均
        if x_s is None:
            x_s = (x_1 + x_2) * 0.5

        b, c, h, w = x_1.shape

        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)

        shared_class_center1 = self.clustering(
            self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1)
        shared_class_center2 = self.clustering(
            self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1)

        v_1 = self.v1(x_1).view(b, c, -1).permute(0, 2, 1)
        v_2 = self.v2(x_2).view(b, c, -1).permute(0, 2, 1)

        att1 = torch.bmm(shared_class_center1, v_1) * self.scale
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale

        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)

        # 单输出：两路输出之和
        return out_1 + x_2_ + out_2 + x_1_
