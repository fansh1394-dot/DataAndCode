# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""双头 + 融合分支检测头聚合模块。

设计背景
--------
RGB 与 IR 是两个不同视场的相机，同一目标在两幅图中的像素位置**系统性偏移**
（Anti-UAV300 实测：归一化中心偏差中位 3.62 pp、最大 10.85 pp，100% 样本 > 1 pp）。
因此单个检测头无法同时满足两个需求：

1. 「融合结果要优于任一单模态」——需要融合通道有**独立监督**；
2. 「红外图上也要出准确的框」——需要 IR 通道用**本模态真实框**监督。

本模块持有三个**独立**的 Detect 头，按模态分别前向，并在出口把三路输出打包成
带键名的 dict，供损失层按模态取用各自标签：

    {"rgb": [...], "ir": [...], "fuse": [...]}

坐标约定
--------
- **融合头与 RGB 头共用 RGB 坐标系**（融合主坐标系为 RGB）→ 两者都用 RGB 标签监督；
- **IR 头使用 IR 坐标系**（来自 infrared.json）→ 用 ``bboxes_ir`` 监督。

实现要点（踩坑记录）
--------------------
在模型图中，三个 Detect 头与聚合头是**相邻的两级**：L52/53/54 是 Detect，
L55 是聚合头且 ``f = [52, 53, 54]``。因此聚合头拿到的输入**已经是三个头的输出**
（训练态为 ``list[Tensor]``，每个 ``(B, no, H, W)``），无需、也**不应该**再次调用 Detect。
早期实现曾在此重复前向，触发了两个隐蔽问题：

1. ``Detect.forward`` 内部 ``x[i] = torch.cat((...))`` 会**就地改写**传入列表，
   而该列表来自 ``_predict_once`` 的 ``y[]`` 缓存，导致缓存被污染为 ``no`` 通道；
2. 重复前向造成计算浪费，且训练/推理态返回值类型不一致。

因此当前实现只做「校验 + identity 打包」，零额外计算、零副作用。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["DualHeadFusionDetect"]


class DualHeadFusionDetect(nn.Module):
    """双头 + 融合分支检测聚合头。

    持有三个独立 ``Detect`` 子模块（RGB / IR / Fuse），但**不在 forward 中重复前向**：
    在 YAML 中三个 Detect 作为独立层先行计算，本模块仅接收其输出并打包成带键 dict。

    Args:
        rgb_detect (nn.Module): RGB 分支检测头。
        ir_detect (nn.Module): IR 分支检测头。
        fuse_detect (nn.Module): 融合分支检测头。
        branch_order (tuple): 三个分支的名字顺序，需与 YAML 输入顺序一致。
            默认 ``("rgb", "ir", "fuse")``。

    Attributes:
        nl (int): 单个头的检测层数（P3/P4/P5 → 3）。
        nc (int): 类别数。
        no (int): 每个 anchor 的输出通道数（``nc + reg_max*4``）。
        stride (torch.Tensor): 共享 stride，形状 ``(nl,)``。

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.head import Detect
        >>> agg = DualHeadFusionDetect(Detect(1, (8, 16, 32)), Detect(1, (8, 16, 32)), Detect(1, (8, 16, 32)))
        >>> feats = [[torch.randn(1, 65, 8, 8)], [torch.randn(1, 65, 8, 8)], [torch.randn(1, 65, 8, 8)]]
        >>> sorted(agg(feats).keys())
        ['fuse', 'ir', 'rgb']
    """

    # 与 Detect 保持一致的类属性，便于上游工具（stride 推断、export）无感识别
    end2end = False
    export = False
    format = None
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False
    inplace = True

    def __init__(
        self,
        rgb_detect: nn.Module,
        ir_detect: nn.Module,
        fuse_detect: nn.Module,
        branch_order: Tuple[str, str, str] = ("rgb", "ir", "fuse"),
        fuse_ir_detect: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if len(branch_order) != 3:
            raise ValueError(f"branch_order 必须包含 3 个分支名，实得 {branch_order}")

        self.branch_order = tuple(branch_order)
        # 三个头作为独立子模块注册，使参数被优化器与 state_dict 正确收录
        self.rgb_detect = rgb_detect
        self.ir_detect = ir_detect
        self.fuse_detect = fuse_detect
        self._branch_modules: Dict[str, nn.Module] = {
            "rgb": self.rgb_detect,
            "ir": self.ir_detect,
            "fuse": self.fuse_detect,
        }

        # ---- B1 (2026-09-27)：可选的第 4 头「fuse_ir」（IR 坐标系辅助监督）----
        # 设计意图：让融合特征在 IR 坐标系下也被要求「独立出准框」，
        # 从而在 RGB 失效时具备兜底能力（见 B1改造方案·融合头双坐标系监督.md）。
        #
        # 兼容性保证：fuse_ir_detect=None 时（默认），本模块行为与改造前**逐位一致**，
        # 且 branch_order 恒为 ("rgb","ir","fuse")，故已有 best.pt 可原样加载与推理。
        self.fuse_ir_detect = fuse_ir_detect
        if fuse_ir_detect is not None:
            # 仅追加到「训练分支」，不改变 branch_order（推理路径不变）
            self._branch_modules["fuse_ir"] = self.fuse_ir_detect
            self.fuse_ir_detect = fuse_ir_detect

        # 一致性校验：三个头必须共享 nl / nc，否则多路 loss 无法统一处理
        nls = {name: int(getattr(mod, "nl", 3)) for name, mod in self._branch_modules.items()}
        ncs = {name: int(getattr(mod, "nc", -1)) for name, mod in self._branch_modules.items()}
        if len(set(nls.values())) != 1:
            raise ValueError(f"DualHeadFusionDetect 要求三个头的 nl 一致，实得 {nls}")
        if len(set(ncs.values())) != 1:
            raise ValueError(f"DualHeadFusionDetect 要求三个头的 nc 一致，实得 {ncs}")

        self.nl = next(iter(nls.values()))
        self.nc = next(iter(ncs.values()))
        self.no = int(getattr(self.rgb_detect, "no", self.nc + 64))
        self.reg_max = int(getattr(self.rgb_detect, "reg_max", 16))
        # stride 由 DetectionModel 构建期逐个回填
        self.stride = torch.zeros(self.nl)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Union[List, Tuple],
    ) -> Dict[str, Union[List[torch.Tensor], Tuple, torch.Tensor]]:
        """校验并打包三路检测头输出。

        框架约定：``BaseModel._predict_once`` 把 ``m.f`` 指定的多个前序层输出收集为
        一个**列表**后单参数调用（``x = m(x)``）。在本架构中 ``m.f = [rgb, ir, fuse]``
        指向三个 Detect 层，故 ``x`` 形如 ``[rgb_out, ir_out, fuse_out]``。

        Args:
            x (list | tuple): 长度 3 的序列，依次为 RGB / IR / 融合三个检测头的输出。

        Returns:
            (dict): ``{"rgb": ..., "ir": ..., "fuse": ...}``。
        """
        if not isinstance(x, (list, tuple)) or len(x) != len(self._branch_modules):
            raise ValueError(
                f"DualHeadFusionDetect 需要长度 {len(self._branch_modules)} 的输入序列"
                f"（{list(self._branch_modules)}），"
                f"实得 {type(x).__name__} (len={len(x) if isinstance(x, (list, tuple)) else 'n/a'})"
            )
        return dict(zip(self._branch_modules.keys(), x))

    # ------------------------------------------------------------------ #
    # B1 接口
    # ------------------------------------------------------------------ #
    @property
    def train_branch_names(self) -> Tuple[str, ...]:
        """参与训练损失计算的分支名（含可选的 ``fuse_ir``）。

        与 :attr:`branch_order` 的区别：
        - ``branch_order`` 恒为 ``("rgb", "ir", "fuse")``，定义**推理**路径与输出顺序；
        - ``train_branch_names`` 额外包含只做监督的 ``fuse_ir``（若启用）。

        损失层应使用本属性，评测/推理层应使用 ``branch_order``。
        """
        return tuple(self._branch_modules.keys())

    @property
    def has_fuse_ir(self) -> bool:
        """是否启用了 IR 坐标系辅助头。"""
        return "fuse_ir" in self._branch_modules

    # ------------------------------------------------------------------ #
    # 辅助接口
    # ------------------------------------------------------------------ #
    def branch(self, name: str) -> nn.Module:
        """按名字取分支头模块。"""
        if name not in self._branch_modules:
            raise KeyError(f"未知分支 '{name}'，可选 {sorted(self._branch_modules)}")
        return self._branch_modules[name]

    def bias_init(self) -> None:
        """对三个头依次执行偏置初始化（若其实现提供）。"""
        for module in self._branch_modules.values():
            fn = getattr(module, "bias_init", None)
            if callable(fn):
                fn()

    def fuse(self) -> "DualHeadFusionDetect":
        """对三个头依次执行 fuse（若其实现提供），并返回自身。"""
        for module in self._branch_modules.values():
            fn = getattr(module, "fuse", None)
            if callable(fn):
                fn()
        return self

    def __repr__(self) -> str:  # pragma: no cover - 仅用于日志展示
        return (
            f"{self.__class__.__name__}(nl={self.nl}, nc={self.nc}, no={self.no}, "
            f"branches={list(self.branch_order)})"
        )
