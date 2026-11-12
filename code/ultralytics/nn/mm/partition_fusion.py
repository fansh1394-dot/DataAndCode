"""Channel-partitioned cross-modal fusion (改动 D3).

根因驱动
--------
消融诊断（``diag_channel_ablation.py``）给出了决定性证据：

  * 融合头 fuse 对 **IR 半边完全零响应**：把 IR 支路（层 L39）的 64 个通道
    **全部置零**，fuse 的 mAP50-95 仅从 0.6914 变到 0.6858（Δ=-0.006，噪声量级）；
    而同一时刻 ir 头自身从 0.6458 正常塌到 0.0000，证明置零确实生效。
  * 融合头对 RGB 半边的依赖是 **全通道渐进式**：置零 5%/10%/20% 通道几乎无影响，
    30% 时崩到 0.42，100% 时塌到 0.0024。
  * 梯度归因（``diag_fusion_grad_attribution.py``）：fuse 损失的梯度
    **75.6% 压在 RGB 支路**，仅 24.4% 流向 IR 支路。

三条证据共同指向：以 ``fused = rgb + gate*(ir - rgb)`` 为形式的算子
（GatedReliabilityFusion / DeformableCrossModalFusion 同属此族）存在**结构性漏洞**——
``gate`` 可整体趋 0，使 fused 退化为纯 RGB，IR 支路在**前向**上被旁路、
在**反向**上得不到梯度。A4/S3/D2-P1b 三次训练期干预全部失败，正是因为
它们约束的是"gate 的取值"，而 gate 本就有权关闭。

本模块的解法
------------
把输出通道 **硬性切分**，让 IR 通路在结构上不可被绕过：

    ouc 个通道 = [ RGB 专属 n_rgb 段 | IR 专属 n_ir 段 | 交互 n_mix 段 ]

* **RGB 专属段**：只取 rgb_warp（对齐后的 RGB 特征），IR 无写入路径。
* **IR 专属段**：只取 ir_attended（可变形采样 + 内容注意力聚合的 IR 特征），
  RGB 无写入路径；此段**没有任何 gate**，故无法被训练"关闭"。
* **交互段**：保留原 gated 形式 ``rgb + g*(ir - rgb)``，供模型自行选择混合比例。

fuse 头（层 54 的 Detect）随后用 3×3 卷积跨全部通道做特征抽取，
因此 IR 专属段的信号必然进入检测头的感受野 —— 这从结构上保证了
「融合分支必然使用 IR」，而非依赖训练把它"逼"出来。

与既有算子的兼容性
------------------
签名与 ``GatedReliabilityFusion`` / ``DeformableCrossModalFusion`` 前四个参数一致
（``inc, ouc, max_offset, base``），``parse_model`` 无需改动。
``split`` 为新增超参，取值形如 ``(0.5, 0.25, 0.25)`` 表示
RGB/IR/交互 三段占总通道的比例，须和为 1。

安全初始化
----------
offset 头零初始化（起点为恒等对齐，避免随机偏移抹平小目标高频细节）；
交互段 gate 用小随机初始化（避免 std 正则的零梯度鞍点，同 P0/L_gate 的教训）。
RGB/IR 专属段无 gate，故不存在"起点即闭合"的问题。
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["ChannelPartitionFusion"]


class ChannelPartitionFusion(nn.Module):
    """可变形跨模态对齐 + **通道级硬性模态分工**的融合算子。

    Args:
        inc: 两模态输入通道数 ``[c_rgb, c_ir]``。
        ouc: 输出通道数。
        max_offset: 归一化网格最大位移（特征像素单位）。
        base: 交互段残差基（``"rgb"``/``"mean"``），与既有算子语义一致。
        split: 三段通道占比 ``(rgb, ir, mix)``，三者之和须为 1。
        num_points: 可变形采样点数 K（IR 专属段与交互段各采样一次，共享 offset 头）。
        attn: 注意力粒度，``"channel"``（默认）或 ``"pixel"``。
    """

    def __init__(
        self,
        inc,
        ouc,
        max_offset: float = 0.08,
        base: str = "rgb",
        split: Sequence[float] = (0.5, 0.25, 0.25),
        num_points: int = 4,
        attn: str = "channel",
    ):
        super().__init__()
        if not isinstance(inc, (list, tuple)) or len(inc) != 2:
            raise ValueError(f"ChannelPartitionFusion expects inc=[c_rgb,c_ir], got {inc}")
        self.c_rgb, self.c_ir, self.ouc = int(inc[0]), int(inc[1]), int(ouc)
        self.max_offset = float(max_offset)
        self.num_points = int(num_points)
        self.base_mode = str(base).lower()
        self.attn_mode = str(attn).lower()
        if self.base_mode not in ("rgb", "mean"):
            raise ValueError(f"base must be 'rgb' or 'mean', got {base!r}")
        if self.attn_mode not in ("channel", "pixel"):
            raise ValueError(f"attn must be 'channel' or 'pixel', got {attn!r}")
        if self.num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {self.num_points}")

        # ---- 通道切分 ----
        sp = tuple(float(v) for v in split)
        if len(sp) != 3 or abs(sum(sp) - 1.0) > 1e-6:
            raise ValueError(f"split must be 3 values summing to 1, got {split}")
        n_rgb = int(round(self.ouc * sp[0]))
        n_ir = int(round(self.ouc * sp[1]))
        n_mix = self.ouc - n_rgb - n_ir
        if n_ir < 1:
            raise ValueError(
                f"IR 段通道数为 0（ouc={self.ouc}, split={sp}）—— 该配置等于放弃 IR 通路，"
                f"与模块设计目的相悖；请增大 ouc 或提高 IR 占比。"
            )
        self.n_rgb, self.n_ir, self.n_mix = n_rgb, n_ir, n_mix
        self.split = sp

        hidden = max(min(self.ouc, 128), 16)
        self.rgb_proj = Conv(self.c_rgb, self.ouc, 1)
        self.ir_proj = Conv(self.c_ir, self.ouc, 1)

        # ---- 可变形 offset 头（两条支路共用：RGB 对齐与 IR 采样）----
        self.offset = nn.Sequential(
            nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * self.num_points, 3, padding=1),
        )
        # ---- 内容注意力头（K 点聚合权重）----
        self.attn = nn.Sequential(
            nn.Conv2d(self.ouc * 2 + 2 * self.num_points, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, self.num_points, 1),
        )
        # ---- 交互段 gate（仅作用于 mix 通道）----
        if n_mix > 0:
            self.gate = nn.Sequential(
                nn.Conv2d(self.ouc * 2, hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, n_mix, 1),
            )
        else:
            self.gate = None

        # 输出投影：跨全部通道做特征抽取，使 IR 专属段必然进入 fuse 头感受野
        self.out = Conv(self.ouc, self.ouc, 3)

        # ---- 安全初始化 ----
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        nn.init.normal_(self.attn[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.attn[-1].bias, mean=0.0, std=1e-3)
        if self.gate is not None:
            nn.init.normal_(self.gate[-1].weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.gate[-1].bias, mean=0.0, std=1e-3)

        self.last_offset_abs_mean = None
        self.last_gate_mean = None
        self.last_gate_std = None
        self.last_attn_entropy = None
        self._gate_raw = None
        self._attn_raw = None

    @staticmethod
    def _base_grid(b, h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != 2:
            raise ValueError(
                f"ChannelPartitionFusion expects two feature maps, got {type(xs).__name__}"
            )
        rgb, ir = xs
        if rgb.shape[-2:] != ir.shape[-2:]:
            raise ValueError(f"feature spatial sizes differ: RGB={rgb.shape[-2:]} IR={ir.shape[-2:]}")

        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        b, c, h, w = rgb.shape
        pair = torch.cat((rgb, ir), dim=1)

        # ---- 1) K 点可变形偏移 ----
        offsets = torch.tanh(self.offset(pair)) * self.max_offset
        offsets = offsets.view(b, self.num_points, 2, h, w)
        base_grid = self._base_grid(b, h, w, pair.device, pair.dtype)
        sample_grid = (base_grid.unsqueeze(1) + offsets.permute(0, 1, 3, 4, 2)).clamp(-1, 1)

        # ---- 2) RGB 恒等对齐 / IR 可变形采样 ----
        # RGB 保持恒等网格（block 内 RGB 是主坐标系，不做位移以免破坏其几何）；
        # IR 在 K 个学习到的偏移点上采样，保留多假设失配。
        rgb_warp = rgb
        ir_expand = ir.unsqueeze(1).expand(b, self.num_points, c, h, w).reshape(
            b * self.num_points, c, h, w
        )
        grid_flat = sample_grid.reshape(b * self.num_points, h, w, 2)
        ir_sampled = F.grid_sample(
            ir_expand, grid_flat, mode="bilinear", align_corners=True, padding_mode="border"
        ).view(b, self.num_points, c, h, w)

        # ---- 3) 内容注意力聚合 K 个 IR 采样 ----
        attn_in = torch.cat((pair, offsets.reshape(b, 2 * self.num_points, h, w)), dim=1)
        attn = self.attn(attn_in).softmax(dim=1)  # [B,K,H,W]
        ir_attended = (ir_sampled * attn.unsqueeze(2)).sum(dim=1)  # [B,C,H,W]

        # ---- 4) 通道级硬性分工 ----
        # 三段拼接：IR 专属段无任何 gate，结构上不可被训练关闭。
        #
        # 判别实验（D3-P2）附加开关 ``_mm_force_ir``：把 RGB 专属段与交互段全部
        # 替换为 ir_attended 的对应切片，使该融合节点的**全部输出通道都只承载 IR 特征**。
        #
        # 与「置零 RGB 段」的关键差异：置零是分布外输入（幅值、统计都变了），
        # 下游卷积可以靠偏置兜底；而这里是**用 IR 特征原位填充**，幅值与语义均
        # 保持 IR 的正常分布，故下游卷积**无从绕过** —— 检测头拿到的任何通道
        # 都只含 IR 信息。
        #
        # 该开关用于回答：给 fuse 头一个「只能用 IR」的输入，它能否学会？
        #   能  -> 瓶颈在训练信号分配（需改损失）；
        #   不能 -> IR 特征对 fuse 头根本不可用（需查 IR 特征质量）。
        _force_ir = bool(getattr(self, "_mm_force_ir", False))

        if _force_ir:
            # 全部通道取 ir_attended（保持 IR 的自然分布，不做任何归一化）
            fused = ir_attended
            self.last_offset_abs_mean = offsets.detach().abs().mean()
            with torch.no_grad():
                p = attn.clamp_min(1e-9)
                self.last_attn_entropy = (
                    -(p * p.log()).sum(dim=1).mean() / math.log(self.num_points)
                    if self.num_points > 1
                    else torch.zeros((), device=rgb.device)
                )
            self._attn_raw = attn
            return self.out(fused)

        parts = []
        if self.n_rgb > 0:
            parts.append(rgb_warp[:, : self.n_rgb])
        parts.append(ir_attended[:, self.n_rgb : self.n_rgb + self.n_ir])
        if self.n_mix > 0:
            rgb_mix = rgb_warp[:, self.n_rgb + self.n_ir :]
            ir_mix = ir_attended[:, self.n_rgb + self.n_ir :]
            if self.gate is not None:
                g_logits = self.gate(pair)
                # channel 粒度时 gate 通道数与 n_mix 对齐；pixel 粒度同理逐通道
                g = torch.sigmoid(g_logits)
                base_mix = 0.5 * (rgb_mix + ir_mix) if self.base_mode == "mean" else rgb_mix
                mix = base_mix + g * (ir_mix - rgb_mix)
                self.last_gate_mean = g.detach().mean()
                self.last_gate_std = g.detach().std()
                self._gate_raw = g
            else:
                mix = rgb_mix
            parts.append(mix)
        fused = torch.cat(parts, dim=1)

        # ---- 诊断量 ----
        self.last_offset_abs_mean = offsets.detach().abs().mean()
        with torch.no_grad():
            p = attn.clamp_min(1e-9)
            self.last_attn_entropy = (
                -(p * p.log()).sum(dim=1).mean() / math.log(self.num_points)
                if self.num_points > 1
                else torch.zeros((), device=rgb.device)
            )
        self._attn_raw = attn
        return self.out(fused)
