# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""双头 + 融合分支的**分模态**评测。

为什么需要单独的评测器
----------------------
双头架构一次前向至少输出三组预测（RGB / IR / 融合）；启用 B1/S3 四头监督时，
还会输出面向 IR 坐标系的 ``fuse_ir``。各分支必须用**各自坐标系的真实标签**评价，
才能回答两个核心问题：

1. **融合是否真的优于单模态？** —— 对比 ``fuse`` 与 ``rgb``/``ir`` 的 mAP。
2. **红外图上是否出得准？** —— 看 ``ir`` 头的 mAP（它用 IR 标签评测）。

标签对应关系（与训练一致）：

==========  ========================  ==================
检测头       评测标签                   坐标系
==========  ========================  ==================
rgb         ``batch["bboxes"]``        RGB
ir          ``batch["bboxes_ir"]``     IR
fuse        ``batch["bboxes"]``        RGB
fuse_ir     ``batch["bboxes_ir"]``     IR（B1/S3 可选）
==========  ========================  ==================

实现方式
--------
不改动 ``DetectionValidator`` 的整体流程，而是读取 RGB/IR 两套标签与帧级模态有效性，
对一次前向产生的各路预测分别累计指标。``fuse_ir`` 在存在四头训练配置时参与验证，
用于审计 only-IR 场景，但不属于默认部署输出。

使用方式::

    from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator
    validator = DualHeadDetectionValidator(args=args, save_dir=save_dir)
    stats = validator(model=model)

输出的 ``stats`` 除标准键外，额外包含::

    metrics/rgb/mAP50-95(B)    metrics/ir/mAP50-95(B)    metrics/fuse/mAP50-95(B)
    metrics/fuse_ir/mAP50-95(B)    # 仅四头配置
    metrics/rgb/mAP50(B)       ... 等
    metrics/fusion_gain_over_rgb    融合相对 RGB 的 mAP50-95 增益
    metrics/fusion_gain_over_ir     融合相对 IR  的 mAP50-95 增益
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import select_device

__all__ = ["DualHeadDetectionValidator", "BRANCH_LABEL_KEYS", "BRANCH_LOSS_NAMES"]

#: 各分支评测所用的 batch 标签键（须与 dual_loss.BRANCH_LABEL_KEYS 保持一致）
BRANCH_LABEL_KEYS: Dict[str, Tuple[str, str]] = {
    "rgb": ("bboxes", "cls"),
    "ir": ("bboxes_ir", "cls_ir"),
    "fuse": ("bboxes", "cls"),
    "fuse_ir": ("bboxes_ir", "cls_ir"),
}

#: 各分支的检测损失列名，顺序须与 ``_BranchDetectionLoss`` 返回的 (box, cls, dfl) 一致。
#: 训练侧 ``DualHeadDetectionLoss`` 按 branch_order 顺序拼接，故训练 CSV 的 loss 列
#: 天然是 ``3 * n_branches`` 项；验证器若要标注这些列，必须用同一份命名。
BRANCH_LOSS_NAMES: Dict[str, Tuple[str, str, str]] = {
    "rgb": ("box_loss_rgb", "cls_loss_rgb", "dfl_loss_rgb"),
    "ir": ("box_loss_ir", "cls_loss_ir", "dfl_loss_ir"),
    "fuse": ("box_loss_fuse", "cls_loss_fuse", "dfl_loss_fuse"),
    # B1/S3：第 4 头 fuse_ir 参与训练和分支验证，用于 IR 坐标系及 only-IR 审计；
    # 它不是默认部署输出。其损失项会出现在 loss_items 中，列名表必须收录，否则
    # 12 项 loss 无法被完整命名（会退化成 val_loss_0..11 占位列名并刷 WARNING）。
    "fuse_ir": ("box_loss_fuse_ir", "cls_loss_fuse_ir", "dfl_loss_fuse_ir"),
}


def _loss_names_for(branch_names: Sequence[str]) -> Tuple[str, ...]:
    """把分支名序列展开成 ``3 * len(branch_names)`` 项损失列名。

    未收录在 :data:`BRANCH_LOSS_NAMES` 中的分支名走兜底命名
    ``box_loss_<name> / cls_loss_<name> / dfl_loss_<name>``。
    """
    names: List[str] = []
    for name in branch_names:
        names.extend(
            BRANCH_LOSS_NAMES.get(name, (f"box_loss_{name}", f"cls_loss_{name}", f"dfl_loss_{name}"))
        )
    return tuple(names)


def _find_train_branch_names(model: Any) -> Optional[Tuple[str, ...]]:
    """从 trainer.model 上取出参与训练损失的分支名（含可选的 ``fuse_ir``）。

    为什么要单独写一层探测
    ----------------------
    ``trainer.model`` 在不同阶段指向不同对象：

    * 训练态下通常是 ``YOLOMM`` / ``Model`` **外壳**，``train_branch_names``
      定义在末层检测头 ``DualHeadFusionDetect`` 上，外壳上没有该属性，
      直接 ``getattr(model, "train_branch_names")`` 会拿到 ``None``；
    * EMA 路径下还可能是 ``.ema`` 包装。

    这正是 B1 训练中每轮验证都刷
    ``DualHead 验证损失项数(12)与已知分支名([['rgb','ir','fuse']])均不匹配``
    的原因——候选 1 从未生效，只剩 3 路的评测分支可比对。

    探测顺序：外壳自身 → 逐层解包（``.model``）→ 末层位置索引 → 遍历子模块。
    全部失败时返回 ``None``，由调用方退回评测分支命名。
    """
    seen: set = set()
    cur = model
    # 逐层解包外壳（YOLOMM.model -> Model.model -> nn.Sequential ...）
    for _ in range(4):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))

        names = getattr(cur, "train_branch_names", None)
        if names:
            return tuple(names)

        # 末层检测头（nn.Sequential 支持位置索引）
        try:
            names = getattr(cur[-1], "train_branch_names", None)
            if names:
                return tuple(names)
        except Exception:
            pass

        # 兜底：遍历子模块找 DualHeadFusionDetect
        try:
            for m in cur.modules():
                names = getattr(m, "train_branch_names", None)
                if names:
                    return tuple(names)
        except Exception:
            pass

        cur = getattr(cur, "model", None) or getattr(getattr(cur, "ema", None), "model", None)

    return None


class DualHeadDetectionValidator(DetectionValidator):
    """分模态评测器：对同一份三路预测，分别用各模态标签计算指标。

    Args:
        dataloader: 验证集 dataloader。
        save_dir (Path | str, optional): 结果保存目录。
        pbar (tqdm, optional): 进度条。
        args (dict, optional): 验证参数。
        _callbacks (dict, optional): 回调。
        branch_names (list[str], optional): 参与评测的分支名，默认 ``["rgb", "ir", "fuse"]``。

    Attributes:
        branch_metrics (dict[str, DetMetrics]): 每个分支一套指标累加器。
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        branch_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.branch_names = list(branch_names) if branch_names else ["rgb", "ir", "fuse"]
        self.branch_metrics: Dict[str, DetMetrics] = {}
        self.subset_metrics: Dict[str, DetMetrics] = {}
        # 决策级晚融合（帧级路由）累加器；未启用时保持 None，不产生任何额外指标。
        self.route_metrics: Optional[DetMetrics] = None
        self._route_stat = {"n": 0, "switched": 0, "correct": 0, "only_ir": 0, "only_ir_switched": 0}

    # ------------------------------------------------------------------ #
    # 初始化：为每个分支建立独立指标累加器
    # ------------------------------------------------------------------ #
    def init_metrics(self, model) -> None:
        """初始化主指标 + 每个分支一套指标。"""
        super().init_metrics(model)
        # B1/reliable 的 fuse_ir 虽不作为默认部署输出，但它是 only-IR 场景
        # 唯一具有 IR 坐标系监督的融合头，必须纳入验证才能形成闭环证据。
        train_names = _find_train_branch_names(model) or ()
        if "fuse_ir" in train_names and "fuse_ir" not in self.branch_names:
            self.branch_names.append("fuse_ir")
        self.branch_metrics = {}
        for name in self.branch_names:
            m = DetMetrics()
            m.names = self.names
            self.branch_metrics[name] = m
        self.subset_metrics = {}
        for name in self.branch_names:
            subsets = ("both", "only_ir") if name in {"ir", "fuse_ir"} else ("both", "only_rgb")
            for subset in subsets:
                m = DetMetrics()
                m.names = self.names
                self.subset_metrics[f"{name}/{subset}"] = m
        # 路由指标：需要一个跨帧混合标签坐标系的统一累加器，才能与
        # always_rgb 在同一 P-R 曲线上可比。
        if self._late_route_cfg()[0]:
            m = DetMetrics()
            m.names = self.names
            self.route_metrics = m
            LOGGER.info(
                "DualHeadDetectionValidator: 决策级晚融合已启用 "
                f"(tau_low={self._late_route_cfg()[1]:.2f}, tau_high={self._late_route_cfg()[2]:.2f}, "
                f"route_conf={self._late_route_cfg()[3]:.2f})"
            )
        LOGGER.info(f"DualHeadDetectionValidator: 分模态评测分支 = {self.branch_names}")

    # ------------------------------------------------------------------ #
    # 关键：preprocess 必须与多模态训练共用同一套精度契约
    # ------------------------------------------------------------------ #
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """多模态感知的预处理，并与训练侧保持完全一致的归一化语义。

        不能用 ``DetectionValidator.preprocess``（它对整张图做 ``/255``）：
        多模态 batch 是 ``(B, 3+Xch, H, W)``，legacy+legacy 契约下 float 输入
        只对 RGB 三通道 ``/255``、X 模态保持原值；整体 ``/255`` 会把 IR 通道
        压到与训练时不同的值域，导致验证 mAP 与训练曲线不可比。

        同时把 ``batch_idx_ir`` / ``bboxes_ir`` / ``cls_ir`` 一并搬到设备
        （IR 分支评测需要按 IR 自己的帧索引取标签）。
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        img = batch["img"].float()
        img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
        batch["img"] = img.half() if self.args.half else img

        for k in (
            "batch_idx",
            "cls",
            "bboxes",
            "batch_idx_ir",
            "bboxes_ir",
            "cls_ir",
            "valid_rgb",
            "valid_ir",
            "frame_kind",
        ):
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)

        xch = self.data.get("Xch", 3) if getattr(self, "data", None) else 3
        expected_channels = 3 + xch
        if batch["img"].shape[1] != expected_channels:
            LOGGER.warning(
                f"期望{expected_channels}通道输入（RGB:3 + X:{xch}），但收到 {batch['img'].shape[1]} 通道"
            )
        return batch

    # ------------------------------------------------------------------ #
    # 分支输出归一化：Detect 在 eval 下返回 (y, feats)，需取出 y
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unwrap_branch_preds(branch_out: Any) -> Any:
        """把单个分支的输出归一化成可送入 NMS 的张量。

        ``Detect.forward`` 在推理态返回 ``(y, x)``，其中 ``y`` 才是 NMS 需要的
        ``(B, 4+nc, N)`` 张量；训练态返回特征图列表。这里统一取 ``y``。
        """
        if isinstance(branch_out, (list, tuple)):
            # 优先取首个元素：推理态为 (y, feats)，y 在第一项
            first = branch_out[0]
            if isinstance(first, torch.Tensor):
                return first
            # 兜底：若是特征图列表（训练态误入），取第一个特征图
            return first
        return branch_out

    def postprocess(self, preds: Any) -> List[Dict[str, torch.Tensor]]:
        """NMS 后处理，兼容双头原始 dict 输出。

        正常验证流程里每个分支的张量是分别喂进来的；但 AFSS 的
        ``score_epoch`` 会直接把 ``model(batch["img"])`` 的**原始输出**交给
        ``validator.postprocess(preds)``，双头下那是 ``{"rgb":..,"ir":..,"fuse":..}``
        的 dict，直接送进 ``non_max_suppression`` 会 ``AttributeError: 'dict' object
        has no attribute 'shape'``。

        这里统一取主分支（融合头）张量再走父类 NMS：AFSS 打分只关心"这套样本
        对融合主坐标系是否学够了"，单分支语义与该机制一致。
        """
        if isinstance(preds, dict):
            preds = self._select_main_branch(preds)
        return super().postprocess(preds)

    # ------------------------------------------------------------------ #
    # batch 标签临时替换：让父类的累加逻辑直接复用
    # ------------------------------------------------------------------ #
    @staticmethod
    def _swap_labels(batch: Dict[str, Any], box_key: str, cls_key: str) -> Dict[str, Any]:
        """返回一个把 ``bboxes``/``cls``/``batch_idx`` 替换成指定模态标签的浅拷贝 batch。

        父类 ``update_metrics``/``_prepare_batch``/``_process_batch`` 只读
        ``batch["batch_idx"]`` / ``batch["cls"]`` / ``batch["bboxes"]``，
        因此临时替换这三项即可复用全部逻辑。

        ``batch_idx`` 必须与框、类同源。IR 与 RGB 的**每帧目标数可以不同**，
        数据层为此单独提供了 ``batch_idx_ir``（见 ``data/augment.py`` 与
        ``data/dataset.py`` 的 collate）。若 IR 分支沿用 RGB 的 ``batch_idx``，
        ``_prepare_batch`` 里 ``batch["batch_idx"] == si`` 的掩码会切错目标，
        IR 的 mAP 将失去意义。
        """
        swapped = dict(batch)
        # 帧索引与框同源：IR 用 batch_idx_ir，其余沿用 batch_idx。
        bi_key = "batch_idx_ir" if box_key == "bboxes_ir" else "batch_idx"
        boxes = batch.get(box_key)
        clses = batch.get(cls_key)
        indices = batch.get(bi_key)
        if all(isinstance(x, torch.Tensor) for x in (boxes, clses, indices)):
            # 防御性检查：标签三元组必须有相同的第一维，否则父类按
            # batch_idx 建 mask 时会产生难以解释的 shape mismatch。
            if boxes.shape[0] == clses.shape[0] == indices.shape[0]:
                swapped["bboxes"] = boxes
                swapped["cls"] = clses
                swapped["batch_idx"] = indices
                return swapped

        # 缺失某套标签时，把该分支视为空标签，并同步清空索引。
        # 不能沿用另一模态的 batch_idx，否则空 bboxes 会被错误索引。
        base_boxes = batch.get("bboxes")
        base_cls = batch.get("cls")
        base_idx = batch.get("batch_idx")
        swapped["bboxes"] = base_boxes[:0] if isinstance(base_boxes, torch.Tensor) else torch.zeros(0, 4)
        swapped["cls"] = base_cls[:0] if isinstance(base_cls, torch.Tensor) else torch.zeros(0, 1)
        swapped["batch_idx"] = base_idx[:0] if isinstance(base_idx, torch.Tensor) else torch.zeros(0, dtype=torch.long)
        return swapped

    @staticmethod
    def _select_valid_samples(
        batch: Dict[str, Any],
        preds: Sequence[Dict[str, torch.Tensor]],
        valid_key: str,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Optional[Dict[str, Any]]]:
        """Select samples that are valid for one modality and remap object indices.

        ``valid_rgb``/``valid_ir`` are frame-level supervision flags, not detection
        targets.  A frame with ``False`` must be removed from that branch's metric
        accumulator; treating its empty label file as a real background would count
        every prediction as a false positive.  Because YOLO stores all objects in one
        concatenated tensor, the surviving ``batch_idx`` values are remapped to the
        compact batch before calling :meth:`DetectionValidator.update_metrics`.

        The helper intentionally returns ``None`` for an empty subset.  Callers can
        then skip the metric update without manufacturing a fake negative image.
        """
        if not preds:
            return [], None

        image_count = len(preds)
        raw_valid = batch.get(valid_key)
        if raw_valid is None:
            valid = torch.ones(image_count, dtype=torch.bool, device=getattr(preds[0].get("conf"), "device", None))
        elif isinstance(raw_valid, torch.Tensor):
            valid = raw_valid.to(dtype=torch.bool).view(-1)
        else:
            valid = torch.as_tensor(raw_valid, dtype=torch.bool)
        if valid.numel() != image_count:
            raise ValueError(
                f"{valid_key} length ({valid.numel()}) does not match prediction batch ({image_count})"
            )

        keep = torch.where(valid)[0]
        if keep.numel() == image_count:
            return list(preds), batch
        if keep.numel() == 0:
            return [], None

        # Prediction lists and per-image metadata use the compact sample order.
        keep_list = [int(i) for i in keep.detach().cpu().tolist()]
        selected_preds = [preds[i] for i in keep_list]
        selected: Dict[str, Any] = dict(batch)
        batch_size = image_count

        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                # Object tensors (bboxes/cls/batch_idx) are handled below using their
                # frame-index tensor; all other tensors are per-image metadata.
                if key not in {"bboxes", "cls", "batch_idx", "bboxes_ir", "cls_ir", "batch_idx_ir"}:
                    selected[key] = value[keep.to(value.device)]
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                selected[key] = [value[i] for i in keep_list]

        # Remap the active detection labels.  ``_swap_labels`` always puts the
        # branch-specific frame index into ``batch_idx`` before this helper runs.
        old_idx = batch.get("batch_idx")
        if isinstance(old_idx, torch.Tensor):
            old_idx = old_idx.view(-1).long()
            object_keep = torch.zeros(old_idx.numel(), dtype=torch.bool, device=old_idx.device)
            keep_device = keep.to(old_idx.device)
            object_keep = torch.isin(old_idx, keep_device)
            remap = torch.full((batch_size,), -1, dtype=torch.long, device=old_idx.device)
            remap[keep_device] = torch.arange(keep.numel(), device=old_idx.device)
            selected["batch_idx"] = remap[old_idx[object_keep]]
            for key in ("bboxes", "cls", "segments", "obb", "masks"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and value.shape[0] == old_idx.numel():
                    selected[key] = value[object_keep]

        return selected_preds, selected

    def _update_branch_metrics(
        self,
        name: str,
        preds: List[Dict[str, torch.Tensor]],
        batch: Dict[str, Any],
    ) -> None:
        """Accumulate one branch only on frames with valid supervision."""
        box_key, cls_key = BRANCH_LABEL_KEYS.get(name, ("bboxes", "cls"))
        valid_key = "valid_ir" if name in {"ir", "fuse_ir"} else "valid_rgb"
        swapped = self._swap_labels(batch, box_key, cls_key)
        selected_preds, selected_batch = self._select_valid_samples(swapped, preds, valid_key)
        if selected_batch is None or not selected_preds:
            return
        self._accumulate(branch_metrics=self.branch_metrics[name], preds=selected_preds, batch=selected_batch)

    def _update_subset_metrics(
        self,
        name: str,
        preds: List[Dict[str, torch.Tensor]],
        batch: Dict[str, Any],
    ) -> None:
        """Accumulate both-visible and single-visible subsets for paper tables."""
        frame_kind = batch.get("frame_kind")
        if frame_kind is None:
            return  # 旧数据无可审计的困难子集身份
        if not isinstance(frame_kind, torch.Tensor):
            frame_kind = torch.as_tensor(frame_kind, device=self.device)
        frame_kind = frame_kind.view(-1)
        box_key, cls_key = BRANCH_LABEL_KEYS.get(name, ("bboxes", "cls"))
        valid_key = "valid_ir" if name in {"ir", "fuse_ir"} else "valid_rgb"
        raw_valid = batch.get(valid_key)
        if raw_valid is None:
            raw_valid = torch.ones_like(frame_kind, dtype=torch.bool)
        elif not isinstance(raw_valid, torch.Tensor):
            raw_valid = torch.as_tensor(raw_valid, device=frame_kind.device, dtype=torch.bool)
        else:
            raw_valid = raw_valid.to(frame_kind.device, dtype=torch.bool).view(-1)

        subset_codes = {"both": 0}
        subset_codes["only_ir" if name in {"ir", "fuse_ir"} else "only_rgb"] = (
            2 if name in {"ir", "fuse_ir"} else 1
        )
        swapped = self._swap_labels(batch, box_key, cls_key)
        for subset, code in subset_codes.items():
            key = f"{name}/{subset}"
            if key not in self.subset_metrics:
                continue
            subset_batch = dict(swapped)
            subset_batch["__subset_valid"] = raw_valid & (frame_kind == code)
            selected_preds, selected_batch = self._select_valid_samples(
                subset_batch, preds, "__subset_valid"
            )
            if selected_batch is not None and selected_preds:
                self._accumulate(self.subset_metrics[key], selected_preds, selected_batch)

    # ------------------------------------------------------------------ #
    # 决策级晚融合：帧级置信度路由（不训练、不改模型）
    # ------------------------------------------------------------------ #
    def _late_route_cfg(self) -> Tuple[bool, float, float, float]:
        """读取路由开关与阈值，缺省关闭，保证历史实验逐位不受影响。"""
        args = getattr(self, "args", None)
        enabled = bool(getattr(args, "late_route", False)) if args is not None else False
        tau_low = float(getattr(args, "late_route_tau_low", 0.30) or 0.30)
        tau_high = float(getattr(args, "late_route_tau_high", 0.50) or 0.50)
        route_conf = float(getattr(args, "late_route_conf", 0.25) or 0.25)
        return enabled, tau_low, tau_high, route_conf

    @staticmethod
    def _frame_conf(pred: Dict[str, torch.Tensor], floor: float) -> float:
        """单帧最高置信度（低于 floor 视为该模态"沉默"，记 0）。"""
        conf = pred.get("conf")
        if conf is None or not isinstance(conf, torch.Tensor) or conf.numel() == 0:
            return 0.0
        top = float(conf.max())
        return top if top >= floor else 0.0

    @staticmethod
    def _flag_tensor(raw: Any, n: int, device) -> torch.Tensor:
        """把 valid_* 归一化成长度 n 的 bool 张量；缺失时视为全 True。"""
        if raw is None:
            return torch.ones(n, dtype=torch.bool, device=device)
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, device=device)
        raw = raw.to(device=device, dtype=torch.bool).view(-1)
        if raw.numel() != n:  # 长度不符时不猜测语义，按“全部有效”处理
            LOGGER.warning(f"valid 标记长度({raw.numel()})与批大小({n})不符，按全有效处理")
            return torch.ones(n, dtype=torch.bool, device=device)
        return raw

    def _update_route_metrics(
        self,
        rgb_preds: List[Dict[str, torch.Tensor]],
        ir_preds: List[Dict[str, torch.Tensor]],
        batch: Dict[str, Any],
    ) -> None:
        """按帧把 RGB 头 / IR 头的**后处理结果**路由到同一套指标里。

        判据只用两路各自的最高置信度：

        * ``c_rgb >= tau_low`` —— RGB 该帧"有话说"，留在 RGB（含 both 帧）；
        * ``c_rgb < tau_low`` 且 ``c_ir > tau_high`` —— RGB 沉默而 IR 确信，切 IR。

        关键点：路由只是"选一路输出"，不涉及任何坐标映射。选 RGB 的帧用
        ``labels/`` 计分，选 IR 的帧用 ``labels_ir/`` 计分，两组样本累加进
        **同一个** ``DetMetrics``，得到跨模态混合口径的全局 P-R 曲线，
        因此可直接与 ``always_rgb`` 对比。

        Args:
            rgb_preds: RGB 头每帧后处理结果（xyxy + conf + cls，letterbox 已还原）。
            ir_preds: IR 头每帧后处理结果，长度必须与 ``rgb_preds`` 相同。
            batch: 当前验证批次（需含 ``valid_rgb``/``valid_ir``，理想情况含 ``frame_kind``）。
        """
        enabled, tau_low, tau_high, route_conf = self._late_route_cfg()
        if not enabled or self.route_metrics is None:
            return
        n = len(rgb_preds)
        if n == 0 or len(ir_preds) != n:
            return

        device = batch["img"].device if isinstance(batch.get("img"), torch.Tensor) else self.device
        use_ir = torch.tensor(
            [
                self._frame_conf(rgb_preds[i], route_conf) < tau_low
                and self._frame_conf(ir_preds[i], route_conf) > tau_high
                for i in range(n)
            ],
            dtype=torch.bool,
            device=device,
        )

        # 路由质量统计（论文里的"路由准确率"）：frame_kind 编码 0=both, 1=only_rgb, 2=only_ir
        st = self._route_stat
        st["n"] += n
        st["switched"] += int(use_ir.sum().item())
        kinds = batch.get("frame_kind")
        if kinds is not None:
            kinds = torch.as_tensor(kinds, device=device).view(-1)
            if kinds.numel() == n:
                expect_ir = kinds == 2
                st["correct"] += int((use_ir == expect_ir).sum().item())
                st["only_ir"] += int(expect_ir.sum().item())
                st["only_ir_switched"] += int((use_ir & expect_ir).sum().item())

        valid_rgb = self._flag_tensor(batch.get("valid_rgb"), n, device)
        valid_ir = self._flag_tensor(batch.get("valid_ir"), n, device)
        for mask, box_key, cls_key, preds in (
            (~use_ir, "bboxes", "cls", rgb_preds),
            (use_ir, "bboxes_ir", "cls_ir", ir_preds),
        ):
            keep = mask & (valid_ir if box_key == "bboxes_ir" else valid_rgb)
            if not bool(keep.any()):
                continue
            sub = dict(self._swap_labels(batch, box_key, cls_key))
            sub["__route_valid"] = keep
            sel_preds, sel_batch = self._select_valid_samples(sub, preds, "__route_valid")
            if sel_batch is not None and sel_preds:
                self._accumulate(self.route_metrics, sel_preds, sel_batch)

    def _route_stats(self) -> Dict[str, float]:
        """汇总路由指标，并给出相对 always_rgb 的增益。"""
        if self.route_metrics is None:
            return {}
        try:
            self.route_metrics.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
            res = self.route_metrics.results_dict
        except Exception as e:  # pragma: no cover - 路由样本为空时
            LOGGER.warning(f"路由指标计算失败: {type(e).__name__}: {e}")
            return {}

        out: Dict[str, float] = {f"metrics/route/{k}": float(v) for k, v in res.items()}
        st = self._route_stat
        if st["n"]:
            out["metrics/route_switch_rate"] = st["switched"] / st["n"]
            out["metrics/route_accuracy"] = st["correct"] / st["n"]
        if st["only_ir"]:
            out["metrics/route_only_ir_hit_rate"] = st["only_ir_switched"] / st["only_ir"]

        base = self.branch_metrics.get("rgb")
        if base is not None:
            try:
                base.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
                out["metrics/route_gain_over_rgb"] = float(
                    res.get("metrics/mAP50-95(B)", 0.0)
                ) - float(base.results_dict.get("metrics/mAP50-95(B)", 0.0))
            except Exception:
                pass

        LOGGER.info(
            "DualHead 决策级路由 mAP50-95(B) = "
            f"{float(res.get('metrics/mAP50-95(B)', 0.0)):.5f}  "
            f"增益(vs RGB) = {out.get('metrics/route_gain_over_rgb', 0.0):+.5f}  "
            f"切到IR = {out.get('metrics/route_switch_rate', 0.0) * 100:.2f}%  "
            f"路由准确率 = {out.get('metrics/route_accuracy', 0.0) * 100:.2f}%"
        )
        return out

    # ------------------------------------------------------------------ #
    # 损失列命名：必须与训练侧实际产出的项数对齐
    # ------------------------------------------------------------------ #
    def _resolve_val_loss_names(self, trainer) -> Tuple[str, ...]:
        """解析验证阶段要标注的损失列名，与 ``trainer.loss_items`` 的实际长度对齐。

        为什么不能沿用 ``trainer.loss_names``
        ------------------------------------
        双头 + 融合分支下，``DualHeadDetectionLoss`` 返回
        ``3 * n_branches`` 项（rgb/ir/fuse 各 box/cls/dfl），即 9 项；
        而 ``trainer.loss_names`` 只有 3 项（单头时代的 ``box/cls/dfl``）。
        若用 3 项去标注 9 项的张量，长度不匹配会直接抛错——这正是
        Kaggle 上验证期崩溃的直接原因。

        这里改为**以 ``loss_items`` 的实际长度为唯一真相源**：

        1. 先用模型的 ``train_branch_names``（含可选的 ``fuse_ir``）比对长度；
        2. 再退回当前评测分支 ``self.branch_names``（四头配置可含 ``fuse_ir``）；
        3. 都不匹配时按实际项数生成占位列名并给出 WARNING。

        B1 (2026-09-27) 修正
        --------------------
        改造前只比对 ``self.branch_names``（恒为 rgb/ir/fuse 三路）。
        启用 B1 后模型有 4 个训练分支（``3 * 4 = 12`` 项 loss），
        与 ``3 * 3 = 9`` 不等，于是每轮验证都会退化成占位列名并刷 WARNING。
        现改为**优先取模型训练分支**：``fuse_ir`` 会参与四头配置的分支验证，
        且其损失项确实存在于张量中，列名必须如实标注。
        """
        loss_items = getattr(trainer, "loss_items", None)
        n_actual = int(loss_items.numel()) if isinstance(loss_items, torch.Tensor) else 0

        # 候选分支顺序：模型训练分支（含 fuse_ir）优先，其次评测分支
        candidates: List[Tuple[str, ...]] = []
        model = getattr(trainer, "model", None)
        train_names = _find_train_branch_names(model)
        if train_names:
            candidates.append(tuple(train_names))
        candidates.append(tuple(self.branch_names))

        if n_actual:
            for names in candidates:
                if n_actual == 3 * len(names) and len(names) > 0:
                    return _loss_names_for(names)

            LOGGER.warning(
                f"DualHead 验证损失项数({n_actual})与已知分支名"
                f"({[list(c) for c in candidates]})均不匹配，按实际项数生成占位列名。"
            )
            return tuple(f"val_loss_{i}" for i in range(n_actual))

        # loss_items 不可用（未进入训练态）：退回 trainer.loss_names
        return tuple(getattr(trainer, "loss_names", ()))

    def _label_val_loss_items(self, loss_items: torch.Tensor, prefix: str = "val") -> Dict[str, float]:
        """把 ``self.loss`` 按已解析的列名打包成 dict（长度已保证一致）。"""
        keys = [f"{prefix}/{x}" for x in getattr(self, "_val_loss_names", ())]
        values = [round(float(x), 5) for x in loss_items]
        if len(keys) != len(values):  # 防御：列名与张量仍不一致时按实际项数截断/补齐
            keys = [f"{prefix}/val_loss_{i}" for i in range(len(values))]
        return dict(zip(keys, values))

    # ------------------------------------------------------------------ #
    # 主流程：一次推理，三套指标
    # ------------------------------------------------------------------ #
    def __call__(self, trainer=None, model=None):
        """执行分模态验证。"""
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # 以 loss_items 实际长度建累加器（双头下为 3*n_branches），维度无关
            self._val_loss_names = self._resolve_val_loss_names(trainer)
            self.loss = torch.zeros(len(self._val_loss_names), device=trainer.device, dtype=trainer.loss_items.dtype)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            model = self._load_model_for_eval(model)

        from ultralytics.utils.torch_utils import de_parallel

        self.init_metrics(de_parallel(model))
        self.jdict = []
        self.phase_times = [0.0] * 4  # 分段耗时累加器（每次验证重置）

        desc = self.get_desc()
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))

        for batch_i, batch in enumerate(bar):
            # 分段计时：trainer._measure_validation 会读 validator.speed 的四项，
            # 缺失键会直接 KeyError，故必须填充（与 DetectionValidator 同契约）。
            dt = tuple(Profile(device=self.device) for _ in range(4))

            with dt[0]:
                batch = self.preprocess(batch)
            with dt[1]:
                preds_raw = model(batch["img"])
            with dt[2]:
                if self.training:
                    # 双头下 loss() 返回 3*n_branches 项；仅在项数匹配时累计，
                    # 避免因列名解析偏差把不匹配的张量加进累加器而崩。
                    loss_items = model.loss(batch, preds_raw)[1]
                    if loss_items.numel() == self.loss.numel():
                        self.loss += loss_items.to(self.loss.dtype)
                    else:
                        LOGGER.warning(
                            f"验证损失项数变化（累加器 {self.loss.numel()} vs 本批 {loss_items.numel()}），"
                            f"本批跳过累计。"
                        )

            # 三路输出分别后处理为 List[dict]（xyxy + conf + cls）
            branch_preds: Dict[str, List[Dict[str, torch.Tensor]]] = {}
            with dt[3]:
                if isinstance(preds_raw, dict):
                    for name in self.branch_names:
                        if name not in preds_raw:
                            continue
                        branch_preds[name] = self.postprocess(self._unwrap_branch_preds(preds_raw[name]))
                else:
                    # 非双头模型：退化为标准单路评测
                    branch_preds = {"fuse": self.postprocess(preds_raw)}
            self.phase_times = [getattr(self, "phase_times", [0.0] * 4)[i] + x.dt for i, x in enumerate(dt)]

            # 主指标沿用 fuse（融合主坐标系 = RGB），但只统计 RGB 有效帧。
            # 无效模态的空标签不是“真实背景”，不能进入主 mAP 的 FP 分母。
            main_key = "fuse" if "fuse" in branch_preds else next(iter(branch_preds))
            main_box_key, main_cls_key = BRANCH_LABEL_KEYS.get(main_key, ("bboxes", "cls"))
            main_valid_key = "valid_ir" if main_key == "ir" else "valid_rgb"
            main_batch_source = self._swap_labels(batch, main_box_key, main_cls_key)
            main_preds, main_batch = self._select_valid_samples(
                main_batch_source, branch_preds[main_key], main_valid_key
            )
            if main_batch is not None and main_preds:
                self.update_metrics(main_preds, main_batch)
            if self.args.plots and batch_i < 3:
                # 可视化仍使用原 batch，便于定位 only-RGB/only-IR 样本。
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, branch_preds[main_key], batch_i)

            # 各分支用各自模态标签累加指标
            for name, preds in branch_preds.items():
                self._update_branch_metrics(name, preds, batch)
                self._update_subset_metrics(name, preds, batch)

            # 决策级晚融合：同一次前向的 rgb/ir 两路结果按帧路由（默认关闭）
            if "rgb" in branch_preds and "ir" in branch_preds:
                self._update_route_metrics(branch_preds["rgb"], branch_preds["ir"], batch)

            bar.set_description(desc)

        stats = self.get_stats()
        n_img = len(self.dataloader.dataset) if getattr(self.dataloader, "dataset", None) else 1
        self.speed = dict(
            zip(
                ("preprocess", "inference", "loss", "postprocess"),
                (t / n_img * 1e3 for t in getattr(self, "phase_times", [0.0] * 4)),
            )
        )
        self.finalize_metrics()

        # 附加分模态与困难子集指标（先 process，随后日志可读）。
        stats.update(self._branch_stats())
        stats.update(self._route_stats())
        stats.update(self._subset_stats())
        # P0（C2）：复合主指标 + 困难子集并列口径。避免只报 fuse 而掩盖
        # 「融合在单模态可见帧上失效」这一论文核心结论。
        stats.update(self._composite_stats(stats))
        self.print_results()

        if self.training:
            model.float()
            return {**stats}
        return stats

    def _load_model_for_eval(self, model):
        """独立评测路径：把权重路径或 None 统一解析成可前向的模型。

        ``final_eval()`` 会以 ``validator(model=best.pt)`` 的形式调用，
        此时 ``model`` 是 ``Path`` 而非 ``nn.Module``，必须当作 ``weights``
        交给 ``AutoBackend``；否则后续 ``de_parallel(model)`` / ``model.names``
        会直接对 Path 取属性而报 AttributeError。
        """
        import torch as _t

        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils import DEFAULT_CFG

        if model is None or isinstance(model, (str, Path)):
            args = getattr(self, "args", None) or DEFAULT_CFG
            return AutoBackend(
                weights=model or getattr(args, "model", None),
                device=select_device(getattr(args, "device", None), getattr(args, "batch", 1)),
                dnn=bool(getattr(args, "dnn", False)),
                data=getattr(args, "data", None),
                fp16=bool(getattr(args, "half", False)),
                fuse=False,
            )
        return model

    # ------------------------------------------------------------------ #
    # 分支指标累加：复用父类 update_metrics 的匹配逻辑
    # ------------------------------------------------------------------ #
    def _accumulate(self, branch_metrics: DetMetrics, preds, batch) -> None:
        """把一批预测按其模态标签累加到指定指标累加器。

        复用 ``DetectionValidator.update_metrics`` 的统计量（confusion_matrix / seen /
        stats / _process_batch），但把结果写入 ``branch_metrics`` 而非 ``self.metrics``。
        """
        # 临时切换 self.metrics，复用父类逻辑后切回
        saved = self.metrics
        saved_cm = self.confusion_matrix
        saved_seen = self.seen
        saved_plots = self.args.plots
        saved_json = self.args.save_json
        saved_txt = self.args.save_txt
        try:
            self.metrics = branch_metrics
            # Branch metrics are in-memory comparisons.  They must not append
            # duplicate JSON/TXT predictions and must not mutate the main
            # confusion matrix while reusing DetectionValidator.update_metrics.
            self.args.plots = False
            self.args.save_json = False
            self.args.save_txt = False
            cm_names = list(self.names.values()) if hasattr(self.names, "values") else list(self.names)
            self.confusion_matrix = ConfusionMatrix(names=cm_names)
            self.update_metrics(preds, batch)
        finally:
            self.metrics = saved
            self.confusion_matrix = saved_cm
            self.seen = saved_seen
            self.args.plots = saved_plots
            self.args.save_json = saved_json
            self.args.save_txt = saved_txt

    # ------------------------------------------------------------------ #
    # 汇总分模态结果
    # ------------------------------------------------------------------ #
    def _branch_stats(self) -> Dict[str, float]:
        """把各分支 mAP 汇总为扁平字典，并计算融合增益。

        注意：``DetMetrics`` 的 ``box``（承载 mAP）是在 ``process()`` 中由累积的
        ``stats`` 计算出来的。标准验证流程里由 ``DetectionValidator.get_stats()``
        触发一次；分支指标是独立累加器，必须在此显式 ``process()``，
        否则 ``results_dict`` 会返回全 0（stats 已在但未计算成 mAP）。
        """
        out: Dict[str, float] = {}
        key_vals: Dict[str, float] = {}

        for name, m in self.branch_metrics.items():
            try:
                # 从累积 stats 计算 mAP（幂等；stats 为空时内部会直接返回）
                m.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
                res = m.results_dict
            except Exception as e:  # pragma: no cover - 指标为空时
                LOGGER.warning(f"分支 '{name}' 指标计算失败: {type(e).__name__}: {e}")
                continue
            for k, v in res.items():
                out[f"metrics/{name}/{k}"] = float(v)
            key_vals[name] = float(res.get("metrics/mAP50-95(B)", 0.0))

        fuse = key_vals.get("fuse")
        if fuse is not None:
            if "rgb" in key_vals:
                out["metrics/fusion_gain_over_rgb"] = fuse - key_vals["rgb"]
            if "ir" in key_vals:
                out["metrics/fusion_gain_over_ir"] = fuse - key_vals["ir"]

        # 日志打印
        if key_vals:
            LOGGER.info("DualHead 分模态 mAP50-95(B): " + "  ".join(
                f"{k}={v:.5f}" for k, v in key_vals.items()
            ))
            if "fuse" in key_vals and "rgb" in key_vals:
                gain_rgb = fuse - key_vals["rgb"]
                flag = "✔ 融合更优" if gain_rgb > 0 else "✘ 融合未超过 RGB"
                LOGGER.info(f"DualHead 融合增益(vs RGB) = {gain_rgb:+.5f}  {flag}")
            if "fuse" in key_vals and "ir" in key_vals:
                gain_ir = fuse - key_vals["ir"]
                flag = "✔ 融合更优" if gain_ir > 0 else "✘ 融合未超过 IR"
                LOGGER.info(f"DualHead 融合增益(vs IR ) = {gain_ir:+.5f}  {flag}")

        return out

    def _subset_stats(self) -> Dict[str, float]:
        """Return auditable both/only-RGB/only-IR metrics per coordinate head."""
        out: Dict[str, float] = {}
        summaries = []
        for key, metrics in self.subset_metrics.items():
            try:
                metrics.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
                res = metrics.results_dict
            except Exception:
                continue
            branch, subset = key.split("/", 1)
            for metric_name, value in res.items():
                out[f"metrics/subset/{branch}/{subset}/{metric_name}"] = float(value)
            summaries.append(
                f"{branch}/{subset}={float(res.get('metrics/mAP50-95(B)', 0.0)):.5f}"
            )
        if summaries:
            LOGGER.info("DualHead 困难子集 mAP50-95(B): " + "  ".join(summaries))
        return out

    def _composite_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        """P0（C2）：复合主指标与困难子集并列口径。

        背景：``primary_metric`` 原先只取 ``fuse_ir/only_ir`` 单一子集，或只看
        ``fuse``。而困难子集实测 ``fuse``（0.029）远低于 ``ir``（0.351）、
        ``fuse_ir``（0.392）——只看 ``fuse`` 会掩盖「融合在单模态可见帧上失效」
        这一论文核心结论，只看单一子集又不足以反映双坐标系能力。

        本方法在**不删除任何既有键**的前提下，附加：

        * ``metrics/fusion_dual_coord_hm``：``fuse`` 与 ``fuse_ir`` mAP50-95 的
          调和平均（两个坐标系的综合；任一为 0 或缺失则回退为存在项）。
        * ``metrics/hard/{branch}/{subset}_mAP50-95``：困难子集的并列口径，
          供论文主表使用（fuse_ir/only_ir、ir/only_ir、fuse/only_rgb）。
        """
        out: Dict[str, float] = {}

        fuse_map = stats.get("metrics/fuse/metrics/mAP50-95(B)")
        fuse_ir_map = stats.get("metrics/fuse_ir/metrics/mAP50-95(B)")
        if fuse_map is not None and fuse_ir_map is not None:
            eps = 1e-9
            out["metrics/fusion_dual_coord_hm"] = (
                2.0 * fuse_map * fuse_ir_map / (fuse_map + fuse_ir_map + eps)
            )
        elif fuse_map is not None:
            out["metrics/fusion_dual_coord_hm"] = float(fuse_map)

        for subset_key in ("fuse_ir/only_ir", "ir/only_ir", "fuse/only_rgb"):
            key = f"metrics/subset/{subset_key}/metrics/mAP50-95(B)"
            if key in stats:
                out[f"metrics/hard/{subset_key}_mAP50-95"] = float(stats[key])

        if out.get("metrics/fusion_dual_coord_hm") is not None:
            LOGGER.info(
                f"DualHead 复合主指标 fusion_dual_coord_hm = "
                f"{out['metrics/fusion_dual_coord_hm']:.5f}  "
                f"(fuse={fuse_map}, fuse_ir={fuse_ir_map})"
            )
        return out

    def print_results(self) -> None:
        """打印主指标 + 各分支指标。"""
        super().print_results()
        for name, m in self.branch_metrics.items():
            try:
                self.logger.info(f"--- 分支 [{name}] ---")
                self.logger.info(str(m))
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------ #
    # 前向输出归一化：双头输出是 dict，父类若干入口按张量处理
    # ------------------------------------------------------------------ #
    def _select_main_branch(self, preds_raw: Any) -> Any:
        """从双头输出中取主分支（融合头）张量，供父类风格的入口复用。

        融合主坐标系为 RGB，且论文主指标取自融合头，故 AFSS 打分等
        复用父类流程的入口统一以 ``fuse`` 为准。
        """
        if isinstance(preds_raw, dict):
            key = "fuse" if "fuse" in preds_raw else next(iter(preds_raw))
            return self._unwrap_branch_preds(preds_raw[key])
        return preds_raw

    # ------------------------------------------------------------------ #
    # AFSS 按样本打分：与 detect 适配器契约对齐（供 AFSS 启用时复用）
    # ------------------------------------------------------------------ #
    def afss_score_sample(self, pred: Any, batch: Dict[str, Any], si: int) -> Dict[str, Any]:
        """构造单样本 AFSS 打分行。

        ``pred`` 由调用方保证已是**单样本的后处理结果**（``{"bboxes","conf","cls"}``）：
        ``afss_score_batch`` 负责把双头 dict 归一化并按 batch 维拆开。
        """
        from ultralytics.engine.afss.tasks.detect import build_detect_afss_score_row

        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")

        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)
        tp = result["tp"]
        matched = int(tp[:, 0].sum()) if len(tp) else 0
        return build_detect_afss_score_row(
            im_file=str(batch["im_file"][si]),
            matched=matched,
            pred_count=int(len(predn["cls"])),
            gt_count=int(len(pbatch["cls"])),
            task_name="detect",
        )

    def afss_score_batch(self, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对整批打分。

        AFSS 的 ``score_epoch`` 传入的是**未后处理**的双头 dict，
        此时先取主分支（融合头）做一次 NMS，得到 ``List[dict]``（长度 = batch），
        再逐样本打分。若传入的已是后处理结果则直接逐样本使用。
        """
        if isinstance(preds, dict):
            preds = self._select_main_branch(preds)
        # 每个元素已是 {"bboxes","conf","cls"} 的后处理 dict 时不再重复 NMS
        if isinstance(preds, (list, tuple)) and preds and isinstance(preds[0], dict) and "conf" in preds[0]:
            per_sample = list(preds)
        else:
            per_sample = self.postprocess(preds)
        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(per_sample)]

    def score_sample(self, pred: Any, batch: Dict[str, Any], si: int) -> Dict[str, Any]:
        """Backward-compatible AFSS sample helper for legacy scorer calls."""
        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Backward-compatible AFSS batch helper for legacy scorer calls."""
        return self.afss_score_batch(preds, batch)
