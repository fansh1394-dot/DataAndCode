import torch
import torch.nn as nn
"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10820553/
    论文题目：Multiscale Sparse Cross-Attention Network for Remote Sensing Scene Classiﬁcation（2025 一区TOP）
    中文题目：用于遥感场景分类的多尺度稀疏交叉注意力网络（2025 一区TOP）
"""

class _ChannelAttention(nn.Module):
    # 通道注意力机制：关注图像中哪些通道（特征）最重要
    def __init__(self, in_channels, reduction_ratio=16):
        super(_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，将每个通道压缩为1个值
        self.fc = nn.Sequential(
            # 第一个全连接层，减少通道数（降维）
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),  # ReLU激活函数
            # 第二个全连接层，恢复通道数（升维）
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()  # Sigmoid激活函数，将输出值压缩到0-1之间
        )

    def forward(self, x):
        # 计算平均池化结果并展平为二维张量（batch_size, channels）
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        # 通过全连接层生成通道注意力权重
        channel_attention = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
        # 将注意力权重应用到输入特征图（逐元素相乘）
        return x * channel_attention

class _SpatialAttention(nn.Module):
    # 空间注意力机制：关注图像中哪些位置（区域）最重要
    def __init__(self, in_channels):
        super(_SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 1x1卷积，将多通道特征图压缩为单通道
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，将输出值压缩到0-1之间

    def forward(self, x):
        # 通过卷积生成空间注意力图
        attention = self.conv(x)
        # 应用Sigmoid激活函数
        attention = self.sigmoid(attention)
        # 将注意力图应用到输入特征图（逐元素相乘）
        x = x * attention
        return x

class _CBAM(nn.Module):
    # 卷积注意力模块：结合通道注意力和空间注意力
    def __init__(self, in_channels, reduction_ratio=4):
        super(_CBAM, self).__init__()
        # 初始化通道注意力模块
        self.channel_attention = _ChannelAttention(in_channels, reduction_ratio)
        # 初始化空间注意力模块
        self.spatial_attention = _SpatialAttention(in_channels)

    def forward(self, x):
        # 先应用通道注意力
        x = self.channel_attention(x)
        # 再应用空间注意力
        x = self.spatial_attention(x)
        return x

class GCBAM(nn.Module):
    # 分组卷积注意力模块：将特征图分组，每组应用独立的CBAM注意力
    def __init__(self, channel, group=8):
        super().__init__()

        self.cov1 = nn.Conv2d(channel, channel, kernel_size=1)  # 1x1卷积，用于特征变换
        self.cov2 = nn.Conv2d(channel, channel, kernel_size=1)  # 1x1卷积，用于特征整合
        self.group = group  # 分组数
        cbam = []
        # 为每个组创建独立的CBAM注意力模块
        for i in range(self.group):
            cbam_ = _CBAM(channel // group)  # 每个CBAM处理一组特征
            cbam.append(cbam_)

        self.cbam = nn.ModuleList(cbam)  # 将CBAM模块列表转换为PyTorch模块列表
        self.sigomid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        x0 = x  # 保存输入，用于残差连接
        x = self.cov1(x)  # 特征变换
        # 将特征图按通道维度分组
        y = torch.split(x, x.size(1) // self.group, dim=1)

        mask = []
        # 对每个组应用CBAM注意力并生成掩码
        for y_, cbam in zip(y, self.cbam):
            y_ = cbam(y_)  # 应用CBAM注意力
            y_ = self.sigomid(y_)  # 应用Sigmoid激活函数

            # 计算注意力图的均值，作为阈值
            mean = torch.mean(y_, [1, 2, 3])
            mean = mean.view(-1, 1, 1, 1)

            # 创建与注意力图形状相同的阈值张量
            gate = torch.ones_like(y_) * mean
            # 生成二值掩码：大于阈值的位置为1，否则保留原注意力值
            mk = torch.where(y_ > gate, 1, y_)
            mask.append(mk)
        mask = torch.cat(mask, dim=1)

        # 将掩码应用到特征图
        x = x * mask
        x = self.cov2(x)  # 特征整合
        x = x + x0  # 残差连接，将输入与处理后的特征相加
        return x
