# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random

import torch
import torch.nn as nn
from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.nn.mm.pruning.trainer import PrunedTrainerMixin
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.nn.mm.complexity import (
    compute_default_multimodal_complexity_report,
    log_default_complexity,
)
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    normalize_modality_token,
    resolve_requested_modality_token,
    resolve_modality_pair,
    set_single_modal_build_metadata_in_ckpt,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
    validate_single_modal_and_modality_exclusive,
)
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime


class MultiModalDetectionTrainer(DetectionTrainer):
    _tid = "ed99bd31"
    """
    多模态检测训练器，基于Input字段路由系统的RGB+X模态训练流程。

    核心特色:
    - 支持配置驱动的多模态路由 (通过第5字段: 'RGB', 'X', 'Dual')
    - 早期融合: 6通道RGB+X输入统一处理
    - 中期融合: 独立RGB和X路径特征提取后融合
    - 晚期融合: 高层语义特征融合
    
    这个类继承DetectionTrainer，集成MultiModalRouter实现灵活的多模态数据流控制。
    支持RGB+深度、RGB+热红外等任意X模态组合的完整训练流程。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化多模态检测训练器。
        
        Args:
            cfg (str | DictConfig, optional): 配置文件路径或配置字典
            overrides (dict, optional): 配置覆盖参数
            _callbacks (list, optional): 回调函数列表
        """
        if overrides is None:
            overrides = {}
        validate_single_modal_and_modality_exclusive(overrides)
        overrides.setdefault("task", "detect")  # 仅当子类未设置时默认为detect（如semseg会设置'semseg'）
        super().__init__(cfg, overrides, _callbacks)
        
        # Get modality parameter from standard cfg system (与推理器保持一致)
        # Modality validation is handled by cfg system, no local validation needed
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        
        # Initialize modality-specific attributes
        self.is_dual_modal = self.modality is None
        self.is_modality_ablation = self.modality is not None
        self.is_single_modal = self.is_modality_ablation

        # Initialize logging control flags
        self._multimodal_config_logged = False  # 控制多模态配置日志只记录一次

        # # Log initialization with modality information
        # if self.modality:
        #     LOGGER.info(f"初始化MultiModalDetectionTrainer - 模态消融: {self.modality}-only")
        # else:
        #     LOGGER.info("初始化MultiModalDetectionTrainer - 双模态训练模式")

        # ---- Distillation runtime (lazy-initialized after model is built) ----
        self.distill_runtime = None
        self._distill_student_collector = None
        self._distill_cfg = self._parse_distill_arg()
        self._sync_afss_task_name()
        self.afss_config = AFSSConfig.from_args(self.args)
        self.afss_runtime = None
        if self.afss_config.enabled:
            self.add_callback("on_train_start", self._afss_on_train_start)
            self.add_callback("on_train_epoch_start", self._afss_on_train_epoch_start)
            self.add_callback("on_train_epoch_end", self._afss_on_train_epoch_end)
            LOGGER.info(
                "AFSS 配置已启用: task=%s warmup=%d, update_interval=%d, score_conf=%.3f, score_iou=%.3f, "
                "ema_alpha=%.3f, state_dir=%s",
                self.afss_config.task_name,
                self.afss_config.warmup_epochs,
                self.afss_config.state_update_interval,
                self.afss_config.score_conf,
                self.afss_config.score_iou,
                self.afss_config.state_ema_alpha,
                self.afss_config.state_dir,
            )

    def _sync_afss_task_name(self):
        """Keep AFSS task wiring aligned with trainer task for adapter-based runtime."""
        task_name = str(getattr(self.args, "task", "detect"))
        setattr(self.args, "afss_task_name", task_name)

    def _setup_train(self, world_size):
        """Build model, then initialize distillation runtime if configured."""
        import time
        td = getattr(self, "td", None)

        super()._setup_train(world_size)

        t0 = time.time()
        self._init_distill_runtime()
        # Distillation epoch-level accumulator (always present for safety)
        self._distill_loss_names = ()
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0
        self._apply_freeze_for_fusion_probe()
        if self.distill_runtime is not None:
            from ultralytics.nn.mm.distill.adapters import YOLOMMDetectDistillAdapter
            self._distill_adapter = YOLOMMDetectDistillAdapter(
                runtime=self.distill_runtime,
                config=self.distill_runtime.config,
                student_model=de_parallel(self.model),
                trainer=self,
            )
            # Do NOT extend self.loss_names -- progress bar only shows base detection loss.
            # Store distill column names separately for CSV and epoch summary.
            _, mode = self._distill_cfg
            distill_names = []
            if mode in ("output", "both"):
                distill_names.extend(["d_out", "d_out_cls", "d_out_loc"])
            if mode in ("feature", "both"):
                distill_names.extend(["d_feat", "d_feat_fg", "d_feat_bg", "d_feat_cwd", "d_feat_ctx"])
            self._distill_loss_names = tuple(distill_names)
            self.add_callback("on_train_epoch_end", self._log_distill_epoch_summary)
        if td is not None and td.enabled:
            td.log_event("setup.init_distill_runtime", elapsed=time.time() - t0)

        # 融合颈 RGB 半边随机屏蔽（路线二）：必须放在模型构建完成之后
        self._init_fusion_rgb_mask()
        # D2-P1 通路级 dropout（逐样本强置融合算子 gate=1）：同样需模型构建完成之后
        self._init_gate_dropout()
        # D3-P2 判别实验：强制融合节点输出全部来自 IR（用 IR 特征原位填充而非置零）
        self._init_force_ir()

    def _init_fusion_rgb_mask(self):
        """装配「融合颈 Concat RGB 半边随机屏蔽」（路线二）。

        背景：诊断脚本 `scripts/diag_fusion_attribution.py` 发现融合头存在伪融合
        退化 —— 在融合颈 Concat（层 46/48/50）把 RGB 半边置零后，fuse 头损失直接
        归零（0.05 -> 0.00），而 rgb/ir 头完全不变。说明融合分支的有效信息全部来自
        RGB 半边，IR 半边被优化成了近零通路。

        本函数不改结构、不改损失，只在训练时按概率切断融合分支的 RGB 支路，迫使
        反向传播给 IR 通路分配梯度 —— 从「通路层面」而非「监督层面」打破伪融合。

        抽样粒度（scope）：
          - batch （旧口径）：每个 forward 掷一次硬币，整批同屏蔽。实测 p=0.1/0.2
            下力度不足 —— 共享权重的融合头在多数 batch 上靠 RGB 就能压低 loss，
            剩下少数 batch 的惩罚不足以抵消，捷径依旧（A4/S3 均未破局）。
          - sample（逐样本）：逐样本掷签，同一 batch 内混有屏蔽/未屏蔽帧，融合头在
            同一次 forward 内就必须同时支持 RGB 与 IR，无法把 RGB 当捷径专精。
            这是目前针对该失效机理对症的改动，建议新实验显式选用（默认仍为 batch，
            以免改变 A4/A5 这对已配对消融的语义）。

        验收必须用 drop_rgb_term 探针（scripts/diag_fusion_attribution.py），
        而非只看 mAP：改造后 fuse 在掩码下应接近基准，而非塌到 0.07。

        不变量：既有权重（best.pt）的推理行为必须逐位不变。故：
          - 开关默认关闭（p=0.0），且只在显式传入时才装配；
          - 真正生效还需模型处于 train 模式（见 DetectionModel._predict_once）；
          - val/predict/导出全部走 eval 模式，屏蔽恒不激活。
        """
        p = float(getattr(self.args, "mm_fusion_rgb_mask_p", 0.0) or 0.0)
        epoch_start = int(getattr(self.args, "mm_fusion_rgb_mask_epoch_start", 0) or 0)
        fill = str(getattr(self.args, "mm_fusion_rgb_mask_fill", "zero") or "zero")
        scope = str(getattr(self.args, "mm_fusion_rgb_mask_scope", "batch") or "batch")

        if p <= 0.0:
            LOGGER.info("融合颈 RGB 屏蔽：未启用（mm_fusion_rgb_mask_p=%.3f）", p)
            return

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"mm_fusion_rgb_mask_p 必须在 [0, 1]，实得 {p}")
        if epoch_start < 0:
            raise ValueError(f"mm_fusion_rgb_mask_epoch_start 必须 >= 0，实得 {epoch_start}")
        if fill not in ("zero", "channel_mean"):
            raise ValueError(f"mm_fusion_rgb_mask_fill 只能是 'zero' 或 'channel_mean'，实得 {fill}")
        if scope not in ("batch", "sample"):
            raise ValueError(f"mm_fusion_rgb_mask_scope 只能是 'batch' 或 'sample'，实得 {scope}")

        if not self._is_dual_head_architecture():
            LOGGER.warning(
                "融合颈 RGB 屏蔽：p=%.3f 但当前模型末层不是 DualHeadFusionDetect / "
                "DualHeadFusionDetectV26，本项不会生效（仅双头融合配置支持）", p
            )
            return

        # self.model 是 DetectionModel（其 .model 才是 nn.Sequential）；
        # 兼容外部再包一层外壳的情况，逐层解包直到拿到 Sequential。
        inner = self.model
        for _ in range(3):
            nxt = getattr(inner, "model", None)
            if isinstance(nxt, torch.nn.Sequential):
                inner = nxt
                break
            if nxt is None:
                break
            inner = nxt
        if not isinstance(inner, torch.nn.Sequential):
            LOGGER.warning("融合颈 RGB 屏蔽：无法定位模型 nn.Sequential，本项不会生效")
            return

        hits = [m.i for m in inner if getattr(m, "_mm_fusion_concat", False)]
        if not hits:
            # 绝不能静默降级：用户以为调了参数其实没调，是最坏的一类 bug
            LOGGER.warning(
                "融合颈 RGB 屏蔽：p=%.3f 但未识别到任何融合节点层，屏蔽不会生效！"
                "请检查模型构建日志中的 '融合颈 RGB 屏蔽：识别到融合节点层 ...' 告警",
                p,
            )
            return

        # 开关挂在 nn.Sequential 上（_predict_once 里的 self 即 DetectionModel，
        # 而它通过 self.model 迭代，故这里挂在 inner 上由 DetectionModel 读取）。
        # 注：_predict_once 的 self 是 DetectionModel，因此属性需同时挂在
        # DetectionModel 与 inner 两级，兼容两种读取路径。
        arm_epoch = 0 if epoch_start > 0 else 1
        for target in (inner, self.model):
            target._mm_fusion_mask_enabled = True
            target._mm_fusion_mask_p = p if arm_epoch == 1 else 0.0
            target._mm_fusion_mask_target_p = p
            target._mm_fusion_mask_epoch_start = epoch_start
            target._mm_fusion_mask_fill = fill
            target._mm_fusion_mask_scope = scope
            target._mm_fusion_mask_current = False

        if epoch_start > 0:
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start_fusion_mask)
            LOGGER.info(
                "融合颈 RGB 屏蔽：启用 p=%.3f（scope=%s），但延迟到 epoch %d 之后才真正生效"
                "（前 %d 个 epoch 保持基线分布）",
                p, scope, epoch_start, epoch_start,
            )
        else:
            LOGGER.info("融合颈 RGB 屏蔽：启用 p=%.3f（scope=%s，立即生效）", p, scope)

        LOGGER.info(
            "融合颈 RGB 屏蔽：命中融合 Concat 层 %s，fill=%s，scope=%s。"
            "期望现象：训练日志中 fuse 头 loss 明显高于未开启时（0.05 -> 0.1 量级），"
            "且 rgb 头 loss 不受影响；若 fuse loss 反而下降，说明屏蔽未生效。",
            hits, fill, scope,
        )

    def _on_train_epoch_start_fusion_mask(self, trainer):
        """按 epoch 门控：延迟生效模式到点后才把 p 从 0 提升到目标值。"""
        inner = getattr(self, "model", None)
        targets = [t for t in (inner, getattr(inner, "model", None)) if t is not None]
        if not targets:
            return
        start = getattr(targets[0], "_mm_fusion_mask_epoch_start", 0)
        if trainer.epoch < start:
            return
        if getattr(targets[0], "_mm_fusion_mask_p", 0.0) > 0.0:
            return  # 已生效，避免重复打印
        target_p = getattr(targets[0], "_mm_fusion_mask_target_p", 0.0)
        for t in targets:
            t._mm_fusion_mask_p = target_p
        LOGGER.info("融合颈 RGB 屏蔽：epoch %d 起正式生效（p=%.3f）", trainer.epoch, target_p)

    def _init_gate_dropout(self):
        """D2-P1 通路级 dropout：逐样本强置融合算子 gate=1，使 fused 走纯 IR 通路。

        背景与设计依据
        --------------
        D0 判别实验（v2_D0_c4_ircoord_e2e_s0）给出一个关键事实：fuse 头的训练
        cls_loss 降到 0.388、与 C4 联合训练的 fuse_ir 头（0.390）完全持平，但
        伪融合探针下 fuse=0.0007 而 fuse_ir=0.6485 —— **损失相等，通路依赖相反**。
        说明融合特征里 RGB 与 IR 信息并存，损失函数对「用哪条通路」无任何约束，
        fuse 头总能找到 RGB 捷径。三条既往修复路线（S9_fix 监督、C4 算子、冻结
        判别）都只改「标签」或「算子」，从未改「损失对通路的约束」。

        本方法在融合算子的输出表达式上直接干预：C4 的 fused 为
            fused = (1-gate)*rgb + gate*ir_attended      （base_mode="rgb"）
        逐样本以概率 p 把 gate 强置为 1 -> fused = ir_attended，即完全消去 RGB
        加性项。这与诊断探针 drop_rgb_term 在同一数学位置、同一语义，训练扰动
        与评测判据构成同构因果链。

        与既有 ``mm_fusion_rgb_mask`` 的区别：后者在融合节点**输入端**置零
        rgb_half，但 rgb_proj(0)=bias≠0，RGB 分量仍留在 fused 中，gate 仍可依赖
        它，屏蔽被绕过（A4/S3 于 p=0.1/0.2 下均未破局）。本项在表达式层面消去，
        物理上无法绕过。

        不变量：开关默认关闭（p=0.0）；仅在 train 模式且显式置位时抽签，
        eval/val/predict/导出恒不激活，既有 best.pt 推理行为逐位不变。
        """
        p = float(getattr(self.args, "mm_gate_drop_p", 0.0) or 0.0)
        if p <= 0.0:
            LOGGER.info("通路级 dropout：未启用（mm_gate_drop_p=%.3f）", p)
            return
        if not 0.0 < p <= 1.0:
            raise ValueError(f"mm_gate_drop_p 必须在 (0, 1]，实得 {p}")

        # 确认模型里确实存在可干预的融合算子，避免「以为调了参数其实没调」
        inner = self.model
        for _ in range(3):
            nxt = getattr(inner, "model", None)
            if isinstance(nxt, torch.nn.Sequential):
                inner = nxt
                break
            if nxt is None:
                break
            inner = nxt
        if not isinstance(inner, torch.nn.Sequential):
            LOGGER.warning("通路级 dropout：无法定位模型 nn.Sequential，本项不会生效")
            return
        from ultralytics.nn.mm.deform_fusion import DeformableCrossModalFusion

        hits = [m.i for m in inner if isinstance(m, DeformableCrossModalFusion)]
        if not hits:
            LOGGER.warning(
                "通路级 dropout：p=%.3f 但未识别到 DeformableCrossModalFusion 融合算子，本项不会生效！"
                "该开关仅支持 --config dual-fusion-deform。",
                p,
            )
            return

        # 属性需同时挂在 DetectionModel（self.model）与内层 Sequential 上，
        # 兼容 _predict_once 的两种读取路径（与 mm_fusion_rgb_mask 同理）。
        for target in (inner, self.model):
            target._mm_gate_drop_p = p
        LOGGER.info(
            "通路级 dropout：启用 p=%.3f（逐样本），命中融合算子层 %s。"
            "期望现象：训练时约 %.0f%% 样本的 fused 走纯 IR；若伪融合探针 fuse 仍贴地板，"
            "说明瓶颈比「通路约束」更靠后。",
            p, hits, p * 100,
        )

    def _init_force_ir(self):
        """D3-P2 判别实验：强制融合节点输出的**全部通道**只承载 IR 特征。

        为什么需要这个实验
        ------------------
        既有六条修复路线（S9_fix 监督 / C4 算子 / 冻结判别 / D0 坐标系 /
        D2-P1b 训练期通路扰动 / D3-P1 通道分工）全部无效。D3-P1 尤其说明问题：
        即便在拼接层面把通道硬性划给 IR，下游 ``out`` 是 3x3 卷积，**会跨通道
        自由加权**，fuse 头照样能把 IR 段的权重压到 0 —— 结构上的"硬连接"在
        可学习卷积面前形同虚设。

        本实验绕开所有间接手段，直接把融合算子的输出**整体替换为 IR 特征**
        （``fused = ir_attended``，用 IR 原位填充而非置零，故幅值与语义都保持
        IR 的自然分布，下游卷积无从绕过）。此时 fuse 头拿到的每一个通道都只含
        IR 信息，问题被逼到最纯粹的形态：

            给 fuse 头一个「只能用 IR」的输入，它究竟能不能学会？

        判读（配合冻结实验使用）
          * 冻结骨干+颈，只训 fuse 头，若 mAP 显著高于地板（>0.15 量级）
            -> IR 特征可用，瓶颈在**训练信号分配**（需改损失/监督协议）
          * 仍贴地板（≈0.001）
            -> IR 特征对 fuse 头**根本不可用**（需回头查 IR 特征质量或头结构）

        不变量：默认关闭；仅在该开关显式置位时生效，且**训练与验证均生效**
        （本实验的目的就是评测这个受限条件下的表现，故不能只在训练时开）。
        """
        if not bool(getattr(self.args, "mm_force_ir_fusion", False)):
            LOGGER.info("D3-P2 强制 IR 融合：未启用（mm_force_ir_fusion=False）")
            return

        inner = self.model
        for _ in range(3):
            nxt = getattr(inner, "model", None)
            if isinstance(nxt, torch.nn.Sequential):
                inner = nxt
                break
            if nxt is None:
                break
            inner = nxt
        if not isinstance(inner, torch.nn.Sequential):
            LOGGER.warning("D3-P2 强制 IR 融合：无法定位模型 nn.Sequential，本项不会生效")
            return

        try:
            from ultralytics.nn.mm.partition_fusion import ChannelPartitionFusion
        except Exception as e:
            LOGGER.warning("D3-P2 强制 IR 融合：无法导入 ChannelPartitionFusion（%s）", e)
            return

        hits = [m.i for m in inner if isinstance(m, ChannelPartitionFusion)]
        if not hits:
            LOGGER.warning(
                "D3-P2 强制 IR 融合：未识别到 ChannelPartitionFusion 融合算子，本项不会生效！"
                "该开关仅支持 --config dual-fusion-partition。"
            )
            return

        # 直接把标志位挂到各融合算子实例上（forward 内读取），
        # 同时挂到 Sequential 与 DetectionModel 以便审计。
        for m in inner:
            if isinstance(m, ChannelPartitionFusion):
                m._mm_force_ir = True
        for target in (inner, self.model):
            target._mm_force_ir_fusion = True

        LOGGER.info(
            "D3-P2 强制 IR 融合：已启用，命中融合算子层 %s。"
            "期望现象：fuse 头输入的全部通道只含 IR 特征；"
            "若 fuse mAP 仍贴地板，则说明 IR 特征对 fuse 头不可用（而非训练信号问题）。",
            hits,
        )

    def _apply_freeze_for_fusion_probe(self):
        """判别实验：冻结骨干与 RGB/IR 头，只训练「融合颈 + 融合头」。

        背景
        ----
        伪融合（fuse 头只读 RGB 半边）已排除两整类解释：
          - 监督侧（S9_fix / B4-b：掩膜样本改用 IR 标签）—— 无效
          - 对齐算子侧（C4：K 点可变形交叉注意力）—— 无效
        而 IR 半边本身是可解码的（ir 头 mAP50-95≈0.64）。剩下的可疑环节是
        ``fuse`` 头的监督坐标系：它以 RGB 坐标系的标签监督，跨模态偏移使 IR
        特征无法在该坐标系下落准，梯度持续惩罚 IR 通路，网络的最优解就是
        「压掉 IR、纯走 RGB」。

        本方法用最少的改动做一次判别：冻结 RGB/IR 骨干与两个单模态头，仅放开
        融合颈（含融合算子）与 ``fuse`` 检测头，并配合
        ``--mm-dual-w-fuse-ir-label`` 把 fuse 头改用 IR 坐标系标签监督。
        若掩码探针下 fuse 头显著回升，则证明瓶颈在监督坐标系而非算子。

        冻结范围（白名单式判定，绝不猜测）：
          - 从 ``DualHeadFusionDetect.fuse_detect`` 反查其在 ``nn.Sequential`` 中的
            真实层下标（检测头是独立顶层层，非命名子模块）
          - 逐个检测头输入索引向前回溯，命中的 ``Concat`` / ``Reliable*`` 融合节点
            及其下游 C3k2 融合颈 → 解冻（``scope=neck_head``）
          - ``fuse_ir`` 头（若存在）保持冻结：它已在 IR 坐标系下监督且表现正常，
            放开只会引入噪声
          - 其余全部冻结（``.dfl`` 由 trainer 恒定冻结）
        """
        if not bool(getattr(self.args, "mm_freeze_for_fusion_probe", False)):
            return
        if not self._is_dual_head_architecture():
            raise ValueError(
                "--mm-freeze-for-fusion-probe 仅支持双头 + 融合分支架构"
                "（--config dual-fusion-*），当前模型末层不是 DualHeadFusionDetect / "
                "DualHeadFusionDetectV26。"
            )

        inner = getattr(self, "model", None)
        inner = getattr(inner, "model", inner)
        if not isinstance(inner, torch.nn.Sequential):
            raise ValueError("--mm-freeze-for-fusion-probe 无法定位模型 nn.Sequential。")

        agg = inner[-1]
        fuse_detect = getattr(agg, "fuse_detect", None)
        if fuse_detect is None:
            raise ValueError("--mm-freeze-for-fusion-probe：模型中不存在 fuse_detect 头。")

        # fuse 头在 inner 中的真实下标：检测头由 parse_model 逐个 append 成独立层
        # （rgb/ir/fuse/fuse_ir → 52/53/54/55），并非以 "fuse_detect" 命名的子模块，
        # 故参数名形如 "model.54.cv3..."。必须从模块身份反查下标，不能按名字匹配。
        fuse_detect_idx = None
        for i, mod in enumerate(inner):
            if mod is fuse_detect:
                fuse_detect_idx = i
                break
        if fuse_detect_idx is None:
            raise ValueError(
                "--mm-freeze-for-fusion-probe：无法在 nn.Sequential 中定位 fuse_detect 层下标。"
            )

        # 收集需要解冻的层下标：融合算子 + 其下游融合颈（到 fuse 头输入为止）
        thaw_idx = set()
        try:
            src_list = list(fuse_detect.f)
        except TypeError:
            src_list = [fuse_detect.f]
        for s in src_list:
            if not isinstance(s, int) or not 0 <= s < len(inner):
                continue
            thaw_idx.add(s)
            cur = inner[s]
            # 沿 f 指针向前回溯（-1 表示上一层），沿途解冻融合算子层
            seen = set()
            while True:
                f = getattr(cur, "f", -1)
                prev = (cur.i - 1) if f == -1 else f
                if not isinstance(prev, int) or not 0 <= prev < len(inner) or prev in seen:
                    break
                seen.add(prev)
                pmod = inner[prev]
                if getattr(pmod, "_mm_fusion_concat", False):
                    thaw_idx.add(prev)
                    cur = pmod
                    continue
                break

        # 冻结范围：scope="head" 时只解冻 fuse 头自身（特征完全冻结），
        # 用于把「特征不可用」与「检测头学不动」彻底分开；
        # scope="neck_head"（默认）额外解冻融合算子与融合颈。
        scope = str(getattr(self.args, "mm_freeze_probe_scope", "neck_head") or "neck_head")
        if scope == "head":
            thaw_idx = {fuse_detect_idx}
        else:
            thaw_idx.add(fuse_detect_idx)

        # 白名单式设置 requires_grad：只有融合颈/融合头相关层的参数解冻
        # 参数名形如 "model.<idx>.<...>"；逐层按下标前缀匹配，避免子串误命中。
        prefixes = tuple(f"model.{i}." for i in sorted(thaw_idx))
        n_thaw = n_freeze = 0
        thaw_names = []
        for name, param in inner.named_parameters():
            full = f"model.{name}"
            is_thaw = any(full.startswith(p) for p in prefixes)
            if is_thaw:
                param.requires_grad = True
                n_thaw += param.numel()
                if len(thaw_names) < 12:
                    thaw_names.append(full)
            else:
                param.requires_grad = False
                n_freeze += param.numel()

        LOGGER.info(
            "融合探针冻结：scope=%s，解冻 %d 个参数（%s），冻结 %d 个参数。",
            scope, n_thaw,
            "仅 fuse 头" if scope == "head" else "融合颈 + fuse 头",
            n_freeze,
        )
        LOGGER.info("融合探针冻结：解冻层下标 %s", sorted(thaw_idx))
        for nm in thaw_names:
            LOGGER.info("  解冻示例参数: %s", nm)
        LOGGER.info(
            "融合探针冻结：fuse 头标签坐标系由 --mm-dual-w-fuse-ir-label 控制"
            "（开启时为 IR 坐标系）。"
        )

        # 冻结语义必须包含 BN 与 dropout：requires_grad=False 只锁权重，
        # train 模式下 BN 仍会用 batch 统计归一化并按 momentum 更新
        # running_mean/running_var（一个 epoch 即漂移殆尽），dropout 也会随机
        # 扰动冻结特征。本实验从 close_mosaic 收尾的 checkpoint 起训，mosaic
        # 重新开启会放大统计漂移，导致冻结头/冻结特征在验证时系统性失真。
        # 故把冻结范围内所有顶层模块永久置于 eval 模式，并注册 batch 级回调，
        # 在 trainer 的 model.train() 之后重新断言（epoch 级回调顺序无法保证，
        # batch 级最稳妥；每批几十次 .eval() 幂等置位，开销可忽略）。
        frozen_modules = [mod for i, mod in enumerate(inner) if i not in thaw_idx]
        frozen_sensitive = []
        for mod in frozen_modules:
            mod.eval()
            for sub in mod.modules():
                if isinstance(
                    sub,
                    (
                        nn.modules.batchnorm._BatchNorm,
                        nn.Dropout,
                        nn.Dropout1d,
                        nn.Dropout2d,
                        nn.Dropout3d,
                    ),
                ):
                    frozen_sensitive.append(sub)
        if frozen_sensitive:
            self._frozen_eval_modules = tuple(frozen_sensitive)

            def _reassert_frozen_eval(trainer, _mods=tuple(frozen_sensitive)):
                for m in _mods:
                    m.eval()

            self.add_callback("on_train_batch_start", _reassert_frozen_eval)
            LOGGER.info(
                "融合探针冻结：%d 个 BN/Dropout 已永久置于 eval 模式"
                "（冻结特征连同归一化统计完全固定）。",
                len(frozen_sensitive),
            )

    def compute_batch_loss(self, batch):
        """Compute batch loss, injecting distillation loss if enabled.

        Returns only base detection loss items for progress bar display.
        Distill items are accumulated separately for epoch-level summary.
        """
        if self.distill_runtime is None:
            return self.model(batch)

        student_model = de_parallel(self.model)
        self._distill_student_collector.reset()

        td = getattr(self, "td", None)
        if td is None or not td.enabled:
            det_loss, det_items, student_preds = student_model.distill_forward(batch)
            teacher_outputs = self.distill_runtime.run_teachers(batch)
            self._distill_adapter.set_epoch_state(self.epoch, self.epochs)
            student_features = self._distill_student_collector.features
            distill_loss, distill_items = self._distill_adapter.compute_distill_loss(
                student_preds, student_features, teacher_outputs
            )
            total_loss = det_loss + distill_loss
            self._accumulate_distill_items(distill_items)
            return total_loss, det_items

        with td.span("batch.forward.student", sync_cuda=True, epoch=self.epoch):
            det_loss, det_items, student_preds = student_model.distill_forward(batch)

        with td.span("batch.forward.teacher", sync_cuda=True, epoch=self.epoch):
            teacher_outputs = self.distill_runtime.run_teachers(batch)

        self._distill_adapter.set_epoch_state(self.epoch, self.epochs)
        student_features = self._distill_student_collector.features
        with td.span("batch.forward.distill", sync_cuda=True, epoch=self.epoch):
            distill_loss, distill_items = self._distill_adapter.compute_distill_loss(
                student_preds, student_features, teacher_outputs
            )

        total_loss = det_loss + distill_loss
        self._accumulate_distill_items(distill_items)
        return total_loss, det_items

    def _accumulate_distill_items(self, distill_items):
        """Accumulate distill items for epoch-level summary and CSV."""
        _zero = 0.0
        distill_values = {}
        if "d_out" in distill_items:
            distill_values["d_out"] = float(distill_items["d_out"])
            distill_values["d_out_cls"] = float(distill_items.get("distill_output_cls", _zero))
            distill_values["d_out_loc"] = float(distill_items.get("distill_output_loc", _zero))
        if "distill_feature" in distill_items:
            distill_values["d_feat"] = float(distill_items["distill_feature"])
            distill_values["d_feat_fg"] = float(distill_items.get("distill_feature_fg", _zero))
            distill_values["d_feat_bg"] = float(distill_items.get("distill_feature_bg", _zero))
            distill_values["d_feat_cwd"] = float(distill_items.get("distill_feature_cwd", _zero))
            distill_values["d_feat_ctx"] = float(distill_items.get("distill_feature_ctx", _zero))
        for k, v in distill_values.items():
            self._distill_epoch_accum[k] = self._distill_epoch_accum.get(k, 0.0) + v
        self._distill_epoch_count += 1

    def _log_distill_epoch_summary(self, trainer):
        """Callback: log epoch-level distill summary after training batches."""
        if not self._distill_loss_names or self._distill_epoch_count == 0:
            return
        n = self._distill_epoch_count
        parts = [f"{name}={self._distill_epoch_accum.get(name, 0.0) / n:.4f}"
                 for name in self._distill_loss_names]
        LOGGER.info(f"  Distill Epoch {self.epoch + 1}/{self.epochs}: {'  '.join(parts)}")

    def save_metrics(self, metrics):
        """Override: inject distill epoch averages into CSV for plotting."""
        if self._distill_loss_names and self._distill_epoch_count > 0:
            n = self._distill_epoch_count
            for name in self._distill_loss_names:
                key = f"train/{name}"
                if key not in metrics:
                    metrics[key] = round(self._distill_epoch_accum.get(name, 0.0) / n, 5)
        super().save_metrics(metrics)
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0

    def _ensure_afss_runtime(self, dataset):
        """Initialize AFSS runtime once training dataset is available."""
        if not self.afss_config.enabled or self.afss_runtime is not None:
            return self.afss_runtime
        self._sync_afss_task_name()
        self.afss_config = AFSSConfig.from_args(self.args)
        self.afss_runtime = AFSSRuntime.from_dataset(
            dataset=dataset,
            args=self.args,
            save_dir=self.save_dir,
            resume=bool(getattr(self.args, "resume", False)),
        )
        return self.afss_runtime

    def _afss_on_train_start(self, trainer):
        """Callback: announce AFSS bootstrap runtime."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_start()

    def _afss_on_train_epoch_start(self, trainer):
        """Callback: refresh AFSS bootstrap epoch selection."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_start(trainer.epoch)

    def _afss_on_train_epoch_end(self, trainer):
        """Callback: persist AFSS state snapshot and trigger scheduled scoring."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_end(
                trainer.epoch,
                trainer=trainer,
                validator=trainer.validator,
            )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader with optional AFSS sampler injection for training."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        sampler_override = None
        if mode == "train" and self.afss_config.enabled:
            runtime = self._ensure_afss_runtime(dataset)
            sampler_override = runtime.create_sampler(rank=rank, shuffle=shuffle)
        return build_dataloader(
            dataset,
            batch_size,
            workers,
            shuffle,
            rank,
            sampler_override=sampler_override,
        )

    def _parse_multimodal_config(self):
        """
        解析和验证数据配置文件中的多模态设置。
        
        解析data.yaml中的modalities和models字段，确保配置正确性，
        提供默认配置和友好的错误信息。
        
        优先支持用户指定的 runtime 模态消融参数。
        
        Returns:
            dict: 解析后的多模态配置
            
        Raises:
            ValueError: 当多模态配置不正确时
        """
        if self.modality:
            primary_modality, secondary_modality = resolve_modality_pair(
                getattr(self, "data", None), default_secondary=self._determine_x_modality_from_data()
            )
            if self.modality and self.modality.lower() == primary_modality.lower():
                config = {
                    'models': [primary_modality, secondary_modality],
                    'modalities': {
                        primary_modality: get_modality_path_from_data(self.data, primary_modality),
                        secondary_modality: get_modality_path_from_data(self.data, secondary_modality),
                    },
                }
                LOGGER.info(f"{primary_modality}模态消融，动态确定secondary模态: {secondary_modality}")
            else:
                actual_secondary = resolve_requested_modality_token(
                    self.modality,
                    getattr(self, "data", None),
                    default_x=secondary_modality,
                )
                config = {
                    'models': [primary_modality, actual_secondary],
                    'modalities': {
                        primary_modality: get_modality_path_from_data(self.data, primary_modality),
                        actual_secondary: get_modality_path_from_data(self.data, actual_secondary),
                    },
                }
                if self.modality == "X":
                    LOGGER.info(f"X模态消融: {actual_secondary}-only (从'X'解析)")
                else:
                    LOGGER.info(f"X模态消融: {actual_secondary}-only")
            return config

        config = self._get_default_multimodal_config()

        if not self.data:
            LOGGER.warning("训练器未提供数据配置，使用默认多模态配置: rgb+depth")
            return config

        _rgb_modality, x_modality = resolve_modality_pair(config['models'], strict=True)
        LOGGER.info(f"✅ 使用用户配置的X模态: {x_modality} (配置驱动，支持任意模态类型)")
        return config

    def _get_x_modality_path(self, modality_name):
        """
        获取指定模态的实际路径。
        
        优先从data.yaml的modality字段读取，
        如果不存在则使用默认格式 'images_{modality_name}'。
        
        Args:
            modality_name (str): 模态名称（如 'ir', 'depth', 'thermal'）
            
        Returns:
            str: 模态对应的目录路径
        """
        return get_modality_path_from_data(self.data, modality_name)
    
    def _determine_x_modality_from_data(self):
        """
        智能确定X模态类型，避免硬编码depth。

        优先级:
        1. 从共享解析组件读取已配置的真实X模态
        2. 从数据目录结构推断
        3. 最后使用depth作为默认值

        Returns:
            str: X模态类型标识符
        """
        x_modality = extract_x_modality_name_from_data(getattr(self, "data", None))
        if x_modality != 'unknown':
            LOGGER.info(f"从数据配置解析X模态: {x_modality}")
            return x_modality

        if self.data and 'path' in self.data:
            try:
                import os
                data_path = self.data['path']
                if os.path.exists(data_path):
                    for item in os.listdir(data_path):
                        if item.startswith('images_') and item != 'images':
                            x_modality = item.replace('images_', '')
                            LOGGER.info(f"从目录结构推断X模态: {x_modality}")
                            return x_modality
            except Exception as e:
                LOGGER.debug(f"目录结构推断失败: {e}")

        LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
        return 'depth'

    def _get_default_multimodal_config(self):
        """
        获取默认的多模态配置，优先从数据配置文件读取。

        Returns:
            dict: 默认多模态配置
        """
        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                rgb_modality, x_modality = resolve_modality_pair(modality_used, strict=True)
                resolved_models = [rgb_modality, x_modality]
                LOGGER.info(f"从modality_used配置读取模态组合: {resolved_models}")
                return {
                    'models': resolved_models,
                    'modalities': {
                        mod: get_modality_path_from_data(self.data, mod)
                        for mod in resolved_models
                    },
                }

        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                rgb_modality, x_modality = resolve_modality_pair(models, strict=True)
                resolved_models = [rgb_modality, x_modality]
                LOGGER.info(f"从models配置读取模态组合: {resolved_models}")
                return {
                    'models': resolved_models,
                    'modalities': {
                        modality: get_modality_path_from_data(self.data, modality)
                        for modality in resolved_models
                    },
                }

        rgb_modality, x_modality = resolve_modality_pair(
            getattr(self, "data", None), default_secondary=self._determine_x_modality_from_data()
        )
        config = {
            'models': [rgb_modality, x_modality],
            'modalities': {
                rgb_modality: get_modality_path_from_data(self.data, rgb_modality),
                x_modality: get_modality_path_from_data(self.data, x_modality),
            },
        }
        LOGGER.info(f"生成默认多模态配置: {rgb_modality}+{x_modality}")
        return config

    def _validate_modality_compatibility(self):
        """
        验证用户指定的modality参数与数据配置的兼容性。
        
        Raises:
            ValueError: 当modality参数与可用数据不兼容时
        """
        if not self.modality:
            return

        available_modalities = []
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        elif self.data and 'models' in self.data:
            available_modalities = self.data['models']
        elif self.data and 'modality_used' in self.data:
            available_modalities = self.data['modality_used']

        if available_modalities:
            try:
                resolved = resolve_requested_modality_token(
                    self.modality,
                    available_modalities,
                    strict=True,
                    default_x=resolve_modality_pair(available_modalities)[1],
                )
            except ValueError as exc:
                raise ValueError(
                    f"指定的modality '{self.modality}' 不在可用模态列表中: {available_modalities}。"
                    f"请检查数据配置或modality参数。"
                ) from exc

            if self.modality == "X":
                LOGGER.info(f"✅ 模态兼容性验证通过: '{self.modality}' 映射到 {resolved}")
            else:
                LOGGER.info(f"✅ 模态兼容性验证通过: {self.modality} → {resolved}，可用模态 {available_modalities} 中")
        else:
            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")

    def preprocess_batch(self, batch):
        """多模态感知的预处理。

        Legacy+legacy: 恢复 29e006c^ 语义 — uint8 全通道 /255; float 仅 RGB /255。
        Mixed/high_precision: 按 contract 逐模态处理。
        顺序: to(device) -> contract-aware preprocess -> half() -> multi-scale
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype  # 捕获原始 dtype（uint8/float）再转 float
        img = batch["img"].float()

        if self.args.debug:
            cond = img[:, :3].max() > 1.5
            LOGGER.warning(
                f"[PLOT-DBG] preprocess condition: img[:,:3].max()={img[:, :3].max():.4f} > 1.5 = {cond}"
            )
            for idx in range(img.shape[0]):
                ri = img[idx, :3]
                xi = img[idx, 3:]
                flag = " *** BLACK ***" if ri.max() < 0.01 else ""
                LOGGER.warning(
                    f"[PLOT-DBG] PRE  idx={idx} | RGB [{ri.min():.4f}, {ri.max():.4f}]{flag} | "
                    f"X [{xi.min():.4f}, {xi.max():.4f}] | dtype={ri.dtype}"
                )

        # contract-driven preprocessing: legacy+legacy 恢复 29e006c^; mixed 按 per-modality 处理
        img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
        batch["img"] = img

        if self.args.debug:
            for idx in range(img.shape[0]):
                ri = img[idx, :3]
                xi = img[idx, 3:]
                flag = " *** BLACK ***" if ri.max() < 0.01 else ""
                LOGGER.warning(
                    f"[PLOT-DBG] POST idx={idx} | RGB [{ri.min():.4f}, {ri.max():.4f}]{flag} | "
                    f"X [{xi.min():.4f}, {xi.max():.4f}] | dtype={ri.dtype}"
                )

        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        return batch

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建多模态数据集，支持RGB+X模态的图像数据。

        Args:
            img_path (str): RGB图像路径
            mode (str): 数据集模式 ('train', 'val', 'test')
            batch (int, optional): 批次大小

        Returns:
            Dataset: 多模态数据集实例
        """
        # 解析多模态配置
        self.multimodal_config = self._parse_multimodal_config()

        # 验证modality兼容性
        self._validate_modality_compatibility()

        # 获取X模态信息
        _rgb_modality, x_modality = resolve_modality_pair(self.multimodal_config['models'], strict=True)
        x_modality_dir = self.multimodal_config['modalities'][x_modality]

        # 构建多模态数据集
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal_image=True,  # 启用图像多模态
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, 'enable_self_modal_generation', False),
            precision_report_dir=self.save_dir,
        )

    def _is_dual_head_architecture(self) -> bool:
        """当前模型是否为「双头 + 融合分支」架构（YOLO11 的 DualHeadFusionDetect 或 YOLO26 的 V26 版）。

        必须同时识别两代聚合头：只判 DualHeadFusionDetect 会让 YOLO26 的
        ``--mm-fusion-rgb-mask`` 等开关在 ``:206`` 处直接 return（仅打一条 warning），
        表现为「参数传了但实验组之间没有变量差」，是极难发现的静默失效。
        """
        try:
            from ultralytics.nn.mm.dual_head import DualHeadFusionDetect
        except Exception:  # pragma: no cover - 依赖缺失
            return False
        dual_classes = [DualHeadFusionDetect]
        try:
            from ultralytics.nn.mm.dual_head_v26 import DualHeadFusionDetectV26

            dual_classes.append(DualHeadFusionDetectV26)
        except Exception:  # pragma: no cover - v26 依赖缺失
            pass
        try:
            inner = getattr(self, "model", None)
            inner = getattr(inner, "model", inner)  # 解包 YOLOMM/Model 外壳
            return isinstance(inner[-1], tuple(dual_classes))
        except Exception:
            return False

    def _dual_head_loss_names(self) -> tuple:
        """双头架构下 ``loss_items`` 的列名（每个分支各 box/cls/dfl 三项）。

        顺序须与 ``DualHeadDetectionLoss.forward`` 的拼接顺序一致，即末层检测头的
        ``train_branch_names``：未挂 fuse_ir 时为 3 项（9 列），B1 配置下为 4 项（12 列）。

        注意：这里**必须按实际参与训练的分支动态展开**。早期版本硬编码
        ``("rgb", "ir", "fuse")``，而 ``dual_loss`` 在 B1 配置下会返回 12 项
        ``loss_items``；``label_loss_items`` 里的 ``dict(zip(keys, items))`` 以较短侧
        为准，会把 ``box/cls/dfl_loss_fuse_ir`` 三项静默丢弃 —— 训练本身没漏
        （梯度照常回传），但 results.csv 与进度条永远看不到 fuse_ir 的损失。
        """
        from ultralytics.nn.mm.dual_val import BRANCH_LOSS_NAMES

        branches = None
        try:
            inner = getattr(self, "model", None)
            inner = getattr(inner, "model", inner)  # 解包 YOLOMM/Model 外壳
            branches = getattr(inner[-1], "train_branch_names", None)
        except Exception:  # pragma: no cover - 结构异常时退回默认三路
            branches = None
        if not branches:
            branches = ("rgb", "ir", "fuse")

        names = []
        for branch in branches:
            names.extend(BRANCH_LOSS_NAMES.get(branch, ()))
        return tuple(names)

    def get_validator(self):
        """
        获取多模态检测验证器。

        双头 + 融合分支架构下必须改用 ``DualHeadDetectionValidator``：该架构一次前向
        输出三组预测（RGB / IR / 融合），需要用**各自模态的标签**分别算 mAP，标准
        ``MultiModalDetectionValidator`` 只按 ``batch["bboxes"]`` 评一套，且其验证损失
        校验假定 3 项（单头），会与双头实际的 9 项冲突并在验证期抛错。

        Returns:
            DetectionValidator: 分模态验证器（双头）或多模态验证器（其余架构）。
        """
        # Validator always uses standard detection loss_names (no distill columns).
        # Distillation loss columns are only for training progress bar, set in
        # _setup_train after runtime init.
        if self._is_dual_head_architecture():
            # 双头：loss_items 为 3*n_branches 项，训练侧表头/CSV 也须同步展开，
            # 否则 label_loss_items 会把 9 项 zip 到 3 个键上而丢掉 6 项。
            self.loss_names = self._dual_head_loss_names()
            from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator

            LOGGER.info(f"双头 + 融合分支：启用分模态评测器，loss_names={list(self.loss_names)}")
            return DualHeadDetectionValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")

        # 延迟导入避免循环依赖
        from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator

        return MultiModalDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """
        绘制多模态训练样本 — 改为薄包装，全部委托 MultiModalPlotter。

        支持：
        - 多模态训练（未传 modality）：输出 RGB、X 与并排对比图
        - 模态消融（传入 modality）：rgb 仅输出 RGB；X 仅输出对应 X 模态
        """
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            ensure_batch_idx_long,
        )

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3

        # X 模态可能使用独立的标注坐标系（非对齐双光谱：X 用 bboxes_ir/cls_ir/batch_idx_ir）。
        # 只有把 X 自己的标注画到 X 面板上，框才会贴在目标上；否则会系统性偏离。
        bboxes_x = batch.get("bboxes_ir")
        bboxes_x = bboxes_x if isinstance(bboxes_x, torch.Tensor) and bboxes_x.numel() > 0 else None
        cls_x = batch.get("cls_ir")
        cls_x = cls_x.squeeze(-1) if (bboxes_x is not None and isinstance(cls_x, torch.Tensor)) else None
        batch_idx_x = batch.get("batch_idx_ir")
        batch_idx_x = (
            ensure_batch_idx_long(batch_idx_x)
            if (bboxes_x is not None and isinstance(batch_idx_x, torch.Tensor))
            else None
        )

        plotter = MultiModalPlotter(self.save_dir, on_plot=self.on_plot)
        plotter.plot_batch(
            stage="train", source="labels", images=batch["img"],
            batch_idx=ensure_batch_idx_long(batch["batch_idx"]),
            cls=batch["cls"].squeeze(-1), bboxes=batch["bboxes"],
            bboxes_x=bboxes_x, cls_x=cls_x, batch_idx_x=batch_idx_x,
            xch=xch,
            x_modality_name=MultiModalPlotter.resolve_x_modality_name(
                getattr(self, 'modality', None),
                getattr(self, 'data', None),
                getattr(self, 'x_modality_name', None),
            ),
            precision_contract=extract_precision_contract_from_batch(batch),
            modality_mode=MultiModalPlotter.determine_modality_mode(
                getattr(self, 'modality', None)
            ),
            paths=batch["im_file"], batch_ni=ni,
            threaded=True,  # 异步绘图，恢复旧版 @threaded 行为，避免阻塞训练循环
        )

    def plot_metrics(self):
        """
        绘制多模态训练指标图表。

        继承父类的指标绘制功能，添加多模态特定的指标可视化。
        """
        from ultralytics.utils.plotting import plot_distill_results

        # 调用父类的指标绘制（生成标准 results.png）
        super().plot_metrics()

        # 蒸馏训练时生成独立的蒸馏指标图
        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="yolomm", on_plot=self.on_plot)

        LOGGER.info("多模态训练指标绘制完成")
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        获取多模态检测模型，确保使用正确的通道数。
        
        重写父类方法以支持多模态输入的动态通道数配置。
        
        Args:
            cfg (str, optional): 模型配置文件路径
            weights (str, optional): 预训练权重路径
            verbose (bool): 是否打印详细信息
            
        Returns:
            DetectionModel: 配置了正确通道数的检测模型
        """
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        
        # 计算正确的输入通道数（双模态训练 与 模态消融 均按 Pch+Xch 构建）
        x_channels = self.data.get('Xch', 3)
        channels = 3 + x_channels  # RGB(3) + X(Xch)
        if verbose and RANK in {-1, 0}:
            tag = "消融" if self.modality else "多模态"
            LOGGER.info(f"{tag}模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch总输入")
        
        # 构建多模态 cfg_dict（build-time single_modal注入 + dataset_config注入）
        from ultralytics.nn.mm.utils import prepare_multimodal_build_cfg

        # 续训时 args 由 ckpt 的 train_args 全量恢复，其中 single_modal='rgb' 会让
        # 构建期注入再次作用到已含路由的 yaml 上并抛 ValueError。续训的架构已由
        # ckpt 定型，注入必须跳过；用一个浅代理屏蔽该字段，不改 args 本身。
        from types import SimpleNamespace

        args_for_build = self.args
        if getattr(self.args, "resume", None) and getattr(self.args, "single_modal", None):
            LOGGER.info(
                f"续训：跳过 single_modal={self.args.single_modal!r} 的构建期注入（结构取自 ckpt）"
            )
            args_for_build = SimpleNamespace(**vars(self.args))
            args_for_build.single_modal = None

        cfg_dict = prepare_multimodal_build_cfg(cfg, self.data, args_for_build, LOGGER)
        model = DetectionModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        
        # 更新multimodal_router的dataset_config（如果存在）
        if hasattr(model, 'multimodal_router') and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"已更新MultiModalRouter的数据集配置，Xch={self.data.get('Xch', 3)}")
        # 注入运行时模态消融参数到路由器
        if hasattr(model, 'mm_router') and model.mm_router and self.modality:
            # 仅设置参数，不在训练预处理阶段做置零/拼接，统一由路由处理
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, 'ablation_strategy', None),
                seed=getattr(self.args, 'seed', None),
            )
        
        if weights:
            model.load(weights)

        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(getattr(self, "data", None)))

        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
            log_default_complexity(model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")
        
        return model

    def _inject_checkpoint_metadata(self, ckpt):
        """把多模态配置写进 checkpoint 字典（父类序列化前调用）。

        必须在内存里改：早先的实现是 ``super().save_model()`` 落盘后立刻
        ``torch_load(self.last)`` 读回再改写，overlayfs 上会读到尚未就绪的文件，
        抛 ``PytorchStreamReader failed reading zip archive: invalid header``
        直接中断训练（冒烟日志 error_report_26.09.29_4）。改成序列化前注入后，
        last/best/epochN 三份都自带元数据，也不再每轮多读+多写一遍 ~58MB。
        """
        x_modality_name = extract_x_modality_name_from_data(getattr(self, "data", None))
        set_x_modality_name_in_ckpt_metadata(ckpt, x_modality_name)
        set_single_modal_build_metadata_in_ckpt(ckpt, model=self.model, args=self.args)
        if hasattr(self, 'multimodal_config'):
            ckpt['multimodal_config'] = self.multimodal_config
        ckpt['modality'] = self.modality  # 保存 runtime 模态消融信息
        return ckpt

    def final_eval(self):
        """
        执行最终评估，包含多模态特定的评估指标。

        注意：父类final_eval()没有返回值，所以这里主要是执行评估并记录多模态信息。
        """
        # 执行标准评估（父类方法没有返回值）
        super().final_eval()
        if not getattr(self, "_checkpoint_written", False) or getattr(self, "validator", None) is None:
            return

        # 导出 LLM 友好的 JSON 格式验证结果
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        # 记录多模态特定信息
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            # 记录模态信息
            _rgb_modality, x_modality = resolve_modality_pair(self.multimodal_config['models'], strict=True)
            if self.modality:
                LOGGER.info(f"最终评估完成 - 模态消融: {self.modality}-only")
            else:
                LOGGER.info(f"最终评估完成 - 双模态训练: RGB+{x_modality}")
        else:
            LOGGER.info("最终评估完成 - 多模态训练")

    # ---- Distillation helpers ------------------------------------------------

    def _parse_distill_arg(self):
        """Parse ``self.args.distill`` and return ``(yaml_path, mode)`` or None."""
        distill = getattr(self.args, "distill", None)
        if distill is None:
            return None
        if not isinstance(distill, (list, tuple)) or len(distill) != 2:
            raise ValueError(
                "distill must be [yaml_path, mode] (list/tuple of length 2), "
                f"got {type(distill).__name__}: {distill}"
            )
        yaml_path, mode = str(distill[0]), str(distill[1]).lower()
        if mode not in ("output", "feature", "both"):
            raise ValueError(f"distill mode must be output/feature/both, got '{mode}'")
        LOGGER.info(f"Distillation enabled: yaml={yaml_path}, mode={mode}")
        return yaml_path, mode

    def _init_distill_runtime(self):
        """Build distill runtime after student model is available (called in train())."""
        if self._distill_cfg is None:
            return
        yaml_path, mode = self._distill_cfg
        from ultralytics.nn.mm.distill.schema import load_distill_config
        from ultralytics.nn.mm.distill.runtime import DistillRuntime

        config = load_distill_config(yaml_path)
        student = de_parallel(self.model)
        self.distill_runtime = DistillRuntime(
            config=config,
            mode=mode,
            family="yolomm",
            student_model=student,
            device=student.device if hasattr(student, 'device') else next(student.parameters()).device,
        )
        self._distill_student_collector = self.distill_runtime.register_student_hooks(student)


# ---------------------------------------------------------------------------
# Pruned model fine-tuning
# ---------------------------------------------------------------------------

def require_pruned_checkpoint(ckpt: dict) -> dict:
    """Validate that a checkpoint carries the minimal pruned marker.

    Args:
        ckpt: Loaded checkpoint dict.

    Returns:
        The ``prune_info`` sub-dict.

    Raises:
        ValueError: If the checkpoint is not a valid pruned checkpoint.
    """
    prune_info = ckpt.get("prune_info")
    if not isinstance(prune_info, dict) or prune_info.get("is_pruned") is not True:
        raise ValueError(
            'finetrain 仅接受带 `prune_info={"is_pruned": True}` 的剪枝权重。'
        )
    return prune_info

class PrunedMultiModalDetectionTrainer(PrunedTrainerMixin, MultiModalDetectionTrainer):
    """Detect-task pruned finetune trainer. Other tasks use make_pruned_trainer_class()."""
