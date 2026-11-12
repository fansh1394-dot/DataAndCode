# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO26 双头 + 融合分支检测头聚合模块。

设计背景
--------
与 :mod:`ultralytics.nn.mm.dual_head` 中的 ``DualHeadFusionDetect`` 目标一致：
RGB / IR 是两台视场不同的相机，同一目标在两幅图中的像素位置**系统性偏移**，
因此需要「每个模态各自一套检测头 + 一条融合分支」，并在出口把多路输出打包成
带键名的 dict，供损失层按模态取用各自的标签。

与 YOLO11 版本的差异
--------------------
YOLO26 的检测头 ``v26Detect`` 是**端到端**的（``end2end = True``），其 ``forward``
返回值随 ``training`` 标志和分支数变化：

======================  ==============================================
场景                     返回
======================  ==============================================
训练态                    ``{"one2many": List[Tensor], "one2one": List[Tensor]}``
推理态（非 export）        ``(y, {"one2many": ..., "one2one": ...})``
导出态（export=True）     ``Tensor (B, 300, 6)``
======================  ==============================================

其中 ``one2many`` 参与训练损失（密集监督），``one2one`` 是 NMS-free 推理路径。
YOLO11 的 ``Detect`` 只返回 ``List[Tensor]``（训练态）或 ``(y, List[Tensor])``
（推理态），结构简单一层，故原实现「校验 + identity 打包」即可。

本模块的职责是：**保留每路的 one2many / one2one 两套输出，不丢失任何信息**。

为什么不直接把 ``{"one2many":...}`` 再套一层 dict
---------------------------------------------------
损失层（``DualHeadDetectionLoss`` / ``DualHeadFusionCriterion``）以
``preds[name][i].shape[1]`` 读取通道数 ``no``、以 ``make_anchors(feats, stride)``
生成锚点，它需要的是 **``List[Tensor]`` 形态的 one2many 特征图**。
若把 v26 的 dict 原样塞进去，``feats[0]`` 会变成字符串键，直接崩。

因此本模块提供两种访问方式：

1. :meth:`forward` 返回**按分支名索引的原始输出**（``Dict[str, Any]``），
   每个 value 就是 ``v26Detect`` 的原始返回值，供``DualHeadFusionCriterion``
   在需要时同时取 one2many 与 one2one；
2. :meth:`one2many` / :meth:`one2one` 两个静态工具函数，从原始输出里安全地
   取出对应的 ``List[Tensor]``，屏蔽「训练态 dict / 推理态 tuple / 导出态 Tensor」
   三种形态的差异。

坐标约定（与 YOLO11 版一致）
----------------------------
- **融合头与 RGB 头共用 RGB 坐标系** → 两者都用 RGB 标签监督；
- **IR 头使用 IR 坐标系** → 用 ``bboxes_ir`` 监督。

实现要点（踩坑记录，继承自 YOLO11 版）
---------------------------------------
在模型图中，三个检测头与聚合头是**相邻两级**：前一级是 ``v26Detect``，
后一级是聚合头且 ``f = [rgb_idx, ir_idx, fuse_idx, ...]``。因此聚合头拿到的
输入**已经是各头的输出**，不应再次调用检测头。``v26Detect.forward`` 内部有
``x[i] = torch.cat(...)`` 的**就地改写**，重复前向会污染 ``_predict_once`` 的
缓存，故本模块同样只做「校验 + 打包」，零额外计算。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["DualHeadFusionDetectV26"]


class DualHeadFusionDetectV26(nn.Module):
    """YOLO26 版双头 + 融合分支聚合头。

    持有三个（或四个）独立 ``v26Detect`` 子模块（RGB / IR / Fuse / 可选 fuse_ir），
    但**不在 forward 中重复前向**：在 YAML 中这些头作为独立层先行计算，
    本模块仅接收其输出并按分支名打包。

    Args:
        rgb_detect (nn.Module): RGB 分支检测头（``v26Detect``）。
        ir_detect (nn.Module): IR 分支检测头。
        fuse_detect (nn.Module): 融合分支检测头。
        branch_order (tuple): 三个推理分支的名字顺序，需与 YAML 输入顺序一致。
            默认 ``("rgb", "ir", "fuse")``。
        fuse_ir_detect (nn.Module | None): 可选的第 4 头（IR 坐标系辅助监督）。

    Attributes:
        nl (int): 单个头的检测层数（P3/P4/P5 → 3）。
        nc (int): 类别数。
        no (int): 每层输出通道数（``nc + reg_max*4``；YOLO26 默认 ``nc + 4``）。
        reg_max (int): DFL 通道数，YOLO26 恒为 1。
        stride (torch.Tensor): 共享 stride，形状 ``(nl,)``。

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.head import v26Detect
        >>> h = lambda: v26Detect(nc=1, ch=(8, 16, 32))
        >>> agg = DualHeadFusionDetectV26(h(), h(), h())
        >>> sorted(agg.branch_order)
        ['fuse', 'ir', 'rgb']
    """

    # 与 v26Detect 保持一致的类属性，便于上游工具（stride 推断）无感识别。
    # 注意：这里刻意设为 True —— 聚合头内部持有的都是端到端头，
    # `DetectionModel.end2end` 由 `getattr(self.model[-1], "end2end", False)` 读取，
    # 若为 False 会让模型走非端到端分支（例如 stride 推断的解包逻辑）。
    end2end = True
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
        self.rgb_detect = rgb_detect
        self.ir_detect = ir_detect
        self.fuse_detect = fuse_detect
        self._branch_modules: Dict[str, nn.Module] = {
            "rgb": self.rgb_detect,
            "ir": self.ir_detect,
            "fuse": self.fuse_detect,
        }

        # 可选第 4 头：IR 坐标系辅助监督。不进入 branch_order（推理路径不变），
        # 仅参与训练损失，语义与 YOLO11 版完全一致。
        self.fuse_ir_detect = fuse_ir_detect
        if fuse_ir_detect is not None:
            self._branch_modules["fuse_ir"] = self.fuse_ir_detect

        # 一致性校验：各头必须共享 nl / nc，否则多路 loss 无法统一处理
        nls = {name: int(getattr(mod, "nl", 3)) for name, mod in self._branch_modules.items()}
        ncs = {name: int(getattr(mod, "nc", -1)) for name, mod in self._branch_modules.items()}
        if len(set(nls.values())) != 1:
            raise ValueError(f"DualHeadFusionDetectV26 要求各头的 nl 一致，实得 {nls}")
        if len(set(ncs.values())) != 1:
            raise ValueError(f"DualHeadFusionDetectV26 要求各头的 nc 一致，实得 {ncs}")

        self.nl = next(iter(nls.values()))
        self.nc = next(iter(ncs.values()))
        # YOLO26 的 reg_max=1 → no = nc + 4；用 getattr 兼容将来可能的多 bin 配置
        self.reg_max = int(getattr(self.rgb_detect, "reg_max", 1))
        self.no = int(getattr(self.rgb_detect, "no", self.nc + 4 * self.reg_max))
        # stride 由 DetectionModel 构建期逐个回填
        self.stride = torch.zeros(self.nl)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Union[List, Tuple],
    ) -> Dict[str, Any]:
        """校验并打包各分支检测头的原始输出。

        框架约定：``BaseModel._predict_once`` 把 ``m.f`` 指定的多个前序层输出收集为
        一个**列表**后单参数调用（``x = m(x)``）。本架构中 ``m.f`` 指向各个检测头层，
        故 ``x`` 形如 ``[rgb_out, ir_out, fuse_out(, fuse_ir_out)]``。

        每个 ``*_out`` 都是 ``v26Detect`` 的**原始返回值**，形态取决于训练/导出状态，
        见模块文档的表格。本方法**不做任何解包**，保留全部信息。

        Args:
            x (list | tuple): 与 ``_branch_modules`` 等长的序列。

        Returns:
            (dict): ``{"rgb": ..., "ir": ..., "fuse": ...}``（含 ``fuse_ir`` 若启用）。
        """
        if not isinstance(x, (list, tuple)) or len(x) != len(self._branch_modules):
            raise ValueError(
                f"DualHeadFusionDetectV26 需要长度 {len(self._branch_modules)} 的输入序列"
                f"（{list(self._branch_modules)}），"
                f"实得 {type(x).__name__} (len={len(x) if isinstance(x, (list, tuple)) else 'n/a'})"
            )
        return dict(zip(self._branch_modules.keys(), x))

    # ------------------------------------------------------------------ #
    # 输出解包工具：屏蔽 v26Detect 的三种返回形态
    # ------------------------------------------------------------------ #
    @staticmethod
    def one2many(raw: Any) -> Optional[List[torch.Tensor]]:
        """从 ``v26Detect`` 的原始输出中取出 one2many（训练用）特征图列表。

        兼容形态：

        - 训练态 ``dict``：直接取 ``raw["one2many"]``；
        - 推理态 ``(y, dict)``：取 ``raw[1]["one2many"]``；
        - 导出态 ``Tensor``：无 one2many（推理只需 one2one），返回 ``None``；
        - 原始 ``List[Tensor]``：说明上游头不是端到端头，原样返回。
        """
        if isinstance(raw, dict):
            return raw.get("one2many")
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
            return raw[1].get("one2many")
        if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], torch.Tensor):
            return list(raw)
        return None

    @staticmethod
    def one2one(raw: Any) -> Optional[List[torch.Tensor]]:
        """从 ``v26Detect`` 的原始输出中取出 one2one（NMS-free 推理用）特征图列表。"""
        if isinstance(raw, dict):
            return raw.get("one2one")
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
            return raw[1].get("one2one")
        # 导出态返回的是已后处理的 (B, 300, 6)，没有特征图列表
        return None

    # ------------------------------------------------------------------ #
    # B1 接口（与 YOLO11 版保持一致，供损失层/评测层复用）
    # ------------------------------------------------------------------ #
    @property
    def train_branch_names(self) -> Tuple[str, ...]:
        """参与训练损失计算的分支名（含可选的 ``fuse_ir``）。"""
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
        """对每个头依次执行偏置初始化（``v26Detect`` 继承自 ``Detect``）。"""
        for module in self._branch_modules.values():
            fn = getattr(module, "bias_init", None)
            if callable(fn):
                fn()

    def fuse(self) -> "DualHeadFusionDetectV26":
        """对每个头依次执行 fuse，并返回自身。

        注意：``v26Detect.fuse`` 会删掉 one2many 分支（``cv2/cv3 = None``），
        这是导出优化。训练中途不要调用；``DetectionModel.fuse()`` 只在推理前调用。
        """
        for module in self._branch_modules.values():
            fn = getattr(module, "fuse", None)
            if callable(fn):
                fn()
        return self

    def __repr__(self) -> str:  # pragma: no cover - 仅用于日志展示
        return (
            f"{self.__class__.__name__}(nl={self.nl}, nc={self.nc}, no={self.no}, "
            f"reg_max={self.reg_max}, branches={list(self.branch_order)})"
        )
