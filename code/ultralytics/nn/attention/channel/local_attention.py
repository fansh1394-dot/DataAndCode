import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2409.13435
    论文题目：PlainUSR: Chasing Faster ConvNet for Eﬃcient Super-Resolution（ACCV 2024）
    中文题目：PlainUSR：追求更快的卷积神经网络以实现高效超分辨率
"""
class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        # 调用父类构造函数初始化网络层
        super(SoftPooling2D, self).__init__()
        # 定义平均池化操作，不计入填充部分的计算
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        # 对输入张量应用指数运算
        x_exp = torch.exp(x)
        # 对指数处理后的张量进行平均池化
        x_exp_pool = self.avgpool(x_exp)
        # 对输入与指数结果相乘后进行平均池化
        x = self.avgpool(x_exp * x)
        # 返回最终的加权平均结果
        return x / x_exp_pool

class LocalAttention(nn.Module):
    def __init__(self, channels, f=16):
        # 初始化模块，设置通道数和中间层维度
        super().__init__()
        # 定义注意力机制的主要处理流程
        self.body = nn.Sequential(
            # 使用1x1卷积调整通道数
            nn.Conv2d(channels, f, 1),
            # 应用Soft Pooling来捕捉重要性信息
            SoftPooling2D(7, stride=3),
            # 通过3x3卷积进一步处理，步长为2
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            # 恢复到原始通道数
            nn.Conv2d(f, channels, 3, padding=1),
            # 使用Sigmoid激活函数生成权重
            nn.Sigmoid(),
        )

        """
            定义门控机制：
            为避免步长卷积和双线性插值带来的伪影，重新校准局部重要性，采用门控机制对局部重要性进行特征优化。
        """
        self.gate = nn.Sequential(
            nn.Sigmoid(),  # 使用Sigmoid作为门控激活函数
        )

    def forward(self, x):

        # 对输入的第一通道应用门控机制
        g = self.gate(x[:, :1].clone())

        # 将body处理的结果调整大小以匹配输入尺寸
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # 返回输入与生成的权重和门控值的乘积
        return x * w * g
