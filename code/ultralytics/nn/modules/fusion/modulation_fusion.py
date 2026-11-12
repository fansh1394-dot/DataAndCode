import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2403.01105
    论文题目：Depth Information Assisted Collaborative Mutual Promotion Network for Single Image Dehazing（CVPR 2024）
    中文题目：用于单图像去雾的深度信息辅助协同互促网络（CVPR 2024）
    调制融合模块模块（Modulation Fusion Module, MFM）：
"""
class _ModulationFusionCore(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(_ModulationFusionCore, self).__init__()

        # 保存 height 参数，height 表示特征图的分组数
        self.height = height
        # 计算中间层的维度 d，取 dim 除以 reduction 的结果和 4 中的最大值
        d = max(int(dim / reduction), 4)

        # 定义自适应平均池化层，将输入特征图池化到 1x1 大小
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义一个顺序容器，包含一系列的卷积层和激活函数
        self.mlp = nn.Sequential(
            # 第一个卷积层，输入通道数为 dim，输出通道数为 d，卷积核大小为 1
            nn.Conv2d(dim, d, 1, bias=False),
            # ReLU 激活函数，增加模型的非线性
            nn.ReLU(),
            # 第二个卷积层，输入通道数为 d，输出通道数为 dim * height，卷积核大小为 1
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        # 定义 Softmax 激活函数，用于在维度 1 上进行归一化
        self.softmax = nn.Softmax(dim=1)

    # 前向传播方法，定义了模块的前向计算逻辑
    def forward(self, in_feats1, in_feats2):
        # 将输入的两个特征图存储在一个列表中
        in_feats = [in_feats1, in_feats2]
        # 获取输入特征图的批次大小 B、通道数 C、高度 H 和宽度 W
        B, C, H, W = in_feats[0].shape

        # 沿着通道维度将两个输入特征图拼接在一起
        in_feats = torch.cat(in_feats, dim=1)
        # 调整输入特征图的形状，将其按照 height 进行分组
        in_feats = in_feats.view(B, self.height, C, H, W)

        # 沿着 height 维度对输入特征图进行求和
        feats_sum = torch.sum(in_feats, dim=1)
        # 对求和后的特征图进行自适应平均池化，然后通过 MLP 网络
        attn = self.mlp(self.avg_pool(feats_sum))
        # 调整注意力图的形状，并通过 Softmax 函数进行归一化
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        # 将输入特征图与注意力图逐元素相乘，然后沿着 height 维度求和
        out = torch.sum(in_feats * attn, dim=1)
        # 返回最终的输出特征图
        return out

class Modulation_Fusion_Module(nn.Module):
    """MFM (CVPR 2024) 融合包装。
    原始模块接受 (in_feats1, in_feats2) 双输入，本包装接受 Concat 后的拼接张量。
    拼接张量 (B, c1, H, W) → 拆分为两路 (c1//2) → 调制融合 → 投影到 c2。
    在 parse_model 中作为 base_modules 注册。
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.half = c1 // 2
        self.core = _ModulationFusionCore(dim=self.half, height=2, reduction=8)
        self.proj = nn.Conv2d(self.half, c2, 1) if self.half != c2 else nn.Identity()

    def forward(self, x):
        c = x.shape[1] // 2
        x1, x2 = x[:, :c], x[:, c:]
        return self.proj(self.core(x1, x2))

