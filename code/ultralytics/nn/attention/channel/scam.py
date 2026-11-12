import torch
import torch.nn as nn

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10423050
    论文题目：FFCA-YOLO for Small Object Detection in Remote Sensing Images（2024 一区TOP）
    中文题目：遥感图像小目标检测的FFCA-YOLO （2024 一区TOP）
"""

def autopad(k, p=None, d=1):  # kernel, padding, dilation  # 卷积核大小，填充，空洞卷积率
    # Pad to 'same' shape outputs # 填充到输出形状相同
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size # 计算实际卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad # 自动计算填充大小
    return p

class _Conv(nn.Module):
    # 标准卷积，参数：输入通道数, 输出通道数, 卷积核大小, 步长, 填充, 分组, 空洞卷积率, 激活函数
    default_act = nn.SiLU()  # 默认激活函数 SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 卷积层
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # 前向传播

    def forward_fuse(self, x):
        return self.act(self.conv(x)) #  融合时不使用BN

class Conv_withoutBN(nn.Module):
    # 不带BN的标准卷积，参数：输入通道数, 输出通道数, 卷积核大小, 步长, 填充, 分组, 空洞卷积率, 激活函数
    default_act = nn.SiLU()  # 默认激活函数 SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 卷积层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 激活函数

    def forward(self, x):
        return self.act(self.conv(x))  # 前向传播

class SCAM(nn.Module):
    # 空间通道注意力模块 (Spatial Channel Attention Module)
    def __init__(self, in_channels, reduction=1):
        super(SCAM, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.inter_channels = in_channels # 中间通道数

        self.k = _Conv(in_channels, 1, 1, 1)  #  用于生成k的1x1卷积
        self.v = _Conv(in_channels, self.inter_channels, 1, 1) # 用于生成v的1x1卷积
        self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1) # 用于生成m的1x1卷积, 不带BN
        self.m2 = _Conv(2, 1, 1, 1) # 用于融合avg和max的1x1卷积

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 (Global Average Pooling)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化 (Global Max Pooling)

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)  # 获取输入维度

        # avg max: [N, C, 1, 1]  # 平均和最大池化后的特征 【左半部分】
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c) # 全局平均池化后进行softmax并reshape
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c) # 全局最大池化后进行softmax并reshape

        # k: [N, 1, HW, 1] # k值  【右半部分】
        k = self.k(x).view(n, 1, -1, 1).softmax(2) # 计算k并reshape, 在空间维度上进行softmax
        # v: [N, 1, C, HW] # v值
        v = self.v(x).view(n, 1, c, -1) # 计算v并reshape
        # y: [N, C, 1, 1] # 通道注意力
        y = torch.matmul(v, k).view(n, c, 1, 1) # 计算通道注意力

        # y2:[N, 1, H, W] # 空间注意力  【左下半部分】
        y_avg = torch.matmul(avg, v).view(n, 1, h, w) # 计算基于平均池化的空间注意力
        y_max = torch.matmul(max, v).view(n, 1, h, w) # 计算基于最大池化的空间注意力
        # y_cat:[N, 2, H, W] #  拼接空间注意力
        y_cat = torch.cat((y_avg, y_max), 1) # 拼接平均和最大池化的空间注意力

        y = self.m(y) * self.m2(y_cat).sigmoid() # 融合通道注意力和空间注意力

        return x + y  # 残差连接
