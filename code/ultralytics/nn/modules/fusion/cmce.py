import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Contrastive Multimodal Contrastive Enhancement (CMCE)

论文：Locate and Verify: A Two-Stream Network for Improved Deepfake Detection (CCF-A 2023)
链接：https://arxiv.org/pdf/2309.11131

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 利用余弦相似性增强双模态特征，返回单输出（两路增强特征之和）
- 通道自适配：两路输入通道数必须一致（由 parse_model 注入）
"""

__all__ = ['CMCE']


class CMCE(nn.Module):
    """Contrastive Multimodal Contrastive Enhancement.

    Args:
        channels (int | None): 通道数（parse_model 自动注入，内部未使用但保持接口兼容）。
    """

    def __init__(self, channels=None):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x1, x2=None):
        # 兼容 parse_model 输入为 [x1, x2] 的形式
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        assert x1.shape[1] == x2.shape[1], \
            f"CMCE: 两路输入通道数须一致, got {x1.shape[1]} vs {x2.shape[1]}"

        cos_sim = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)

        # 余弦相似性调整特征
        fa = x1 + x2 * cos_sim
        fb = x2 + x1 * cos_sim

        fa = self.relu(fa)
        fb = self.relu(fb)

        # 单输出：两路增强特征之和
        return fa + fb
