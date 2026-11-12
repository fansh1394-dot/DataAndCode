import math
import torch
import torch.nn as nn
import numbers

"""
Frequency Exhaustive Fusion Mechanism (FEFM)

论文：Complementary Advantages: Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising (CVPR 2025)
链接：https://arxiv.org/pdf/2412.16645v1

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 频域+空域联合跨模态融合（DCT 分块注意力）
- 通道数必须为 16 的倍数（dim // 16 = num_heads）
- 懒加载：torch_dct / einops 仅在 forward() 时导入，不使用时不需要安装
"""

__all__ = ['FEFM']


def _to_3d(x):
    from einops import rearrange
    return rearrange(x, 'b c h w -> b (h w) c')


def _to_4d(x, h, w):
    from einops import rearrange
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class _BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class _LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = _BiasFree_LayerNorm(dim)
        else:
            self.body = _WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


def _lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class FEFM(nn.Module):
    """Frequency Exhaustive Fusion Mechanism.

    Args:
        channels (int): 输入通道数（须为 16 的倍数，parse_model 注入）。
        bias (bool): 卷积偏置（默认 True）。
        depth (int): 层深索引，用于 lambda 初始化（默认 0）。
    """

    def __init__(self, channels, bias=True, depth=0):
        super().__init__()
        dim = channels
        self.num_heads = dim // 16
        self.lambda_init = _lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.to_hidden = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_hidden_nir = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw_nir = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
                                          groups=dim * 2, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.special = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_middle = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.pool1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))
        self.pool2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        self.norm = _LayerNorm(dim, LayerNorm_type='WithBias')
        self.patch_size = 8

    def forward(self, x, nir=None):
        if nir is None and isinstance(x, (list, tuple)):
            x, nir = x

        # 懒加载：仅在首次 forward 时导入（Python import 有缓存，后续开销极小）
        try:
            from einops import rearrange
            import torch_dct as dct
        except ImportError as e:
            raise ImportError(
                "FEFM 需要 einops 和 torch_dct，请运行: pip install einops torch_dct") from e

        b, c, h, w = x.shape

        hidden = self.to_hidden(x)
        nir_hidden = self.to_hidden_nir(nir)
        q = self.to_hidden_dw(hidden)
        k, v = self.to_hidden_dw_nir(nir_hidden).chunk(2, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)

        q_fft = dct.dct_2d(q_patch.float())
        k_fft = dct.dct_2d(k_patch.float())

        out1 = q_fft * k_fft

        q_fft = rearrange(q_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)',
                          head=self.num_heads)
        k_fft = rearrange(k_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)',
                          head=self.num_heads)

        out1 = rearrange(out1, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)',
                         head=self.num_heads)
        q_fft = torch.nn.functional.normalize(q_fft, dim=-1)
        k_fft = torch.nn.functional.normalize(k_fft, dim=-1)
        attn = (q_fft @ k_fft.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ out1)
        out = rearrange(out, 'b head c (h w patch1 patch2) -> b (head c) h w patch1 patch2',
                        head=self.num_heads,
                        h=h // self.patch_size, w=w // self.patch_size,
                        patch1=self.patch_size, patch2=self.patch_size)

        out = dct.idct_2d(out)
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
                        patch1=self.patch_size, patch2=self.patch_size)

        out = self.project_middle(out)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        output = self.pool1(q * out) + self.pool2(v - lambda_full * (v * out))
        output = self.project_out(output)
        return output
