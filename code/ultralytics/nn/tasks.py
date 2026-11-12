# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
# Extra modules conditional import（默认禁用）
# from ultralytics.nn.extra_modules import *
# from ultralytics.nn.backbone.convnextv2 import *
# from ultralytics.nn.backbone.fasternet import *
# from ultralytics.nn.backbone.efficientViT import *
# from ultralytics.nn.backbone.EfficientFormerV2 import *
# from ultralytics.nn.backbone.VanillaNet import *
# from ultralytics.nn.backbone.revcol import *
# from ultralytics.nn.backbone.lsknet import *
# from ultralytics.nn.backbone.SwinTransformer import *
# from ultralytics.nn.backbone.repvit import *
# from ultralytics.nn.backbone.CSwomTramsformer import *
# from ultralytics.nn.backbone.UniRepLKNet import *
# from ultralytics.nn.backbone.TransNext import *
# from ultralytics.nn.backbone.rmt import *
# from ultralytics.nn.backbone.pkinet import *
# from ultralytics.nn.backbone.mobilenetv4 import *
# from ultralytics.nn.backbone.starnet import *
# from ultralytics.nn.backbone.inceptionnext import *
# from ultralytics.nn.extra_modules.mobileMamba.mobilemamba import *
# from ultralytics.nn.backbone.MambaOut import *
# from ultralytics.nn.backbone.overlock import *
# from ultralytics.nn.backbone.lsnet import *
# except:
#     pass
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.mm.utils import MM_METADATA_KEY, MM_X_MODALITY_KEY, has_multimodal_config, sync_x_modality_name_to_model
from ultralytics.nn.backbone import (
    TimmBackbone,
    convnextv2_atto,
    convnextv2_base,
    convnextv2_femto,
    convnextv2_huge,
    convnextv2_large,
    convnextv2_nano,
    convnextv2_pico,
    convnextv2_tiny,
    repvit_m0_9,
    repvit_m1_0,
    repvit_m1_1,
    repvit_m1_5,
    repvit_m2_3,
    efficientformerv2_s0,
    efficientformerv2_s1,
    efficientformerv2_s2,
    efficientformerv2_l,
    EfficientViT_M0,
    EfficientViT_M1,
    EfficientViT_M2,
    EfficientViT_M3,
    EfficientViT_M4,
    EfficientViT_M5,
    SwinTransformer_Tiny,
)
from ultralytics.nn.modules import (
    # non-fusion modules (keep explicit import)
    AIFI,
    AIFI_DyT,
    AIFI_EDFFN,
    AIFI_LPE,
    AIFI_Mona,
    AIFI_RepBN,
    AIFI_SEFFN,
    AIFI_SEFN,
    TransformerEncoderLayer_AdditiveTokenMixer,
    TransformerEncoderLayer_ASSA,
    TransformerEncoderLayer_ASSA_SEFN,
    TransformerEncoderLayer_ASSA_SEFN_Mona,
    TransformerEncoderLayer_ASSA_SEFN_Mona_DyT,
    TransformerEncoderLayer_DAttention,
    TransformerEncoderLayer_DHSA,
    TransformerEncoderLayer_DPB,
    TransformerEncoderLayer_EfficientAdditiveAttnetion,
    TransformerEncoderLayer_HiLo,
    TransformerEncoderLayer_LocalWindowAttention,
    TransformerEncoderLayer_MSLA,
    TransformerEncoderLayer_MSMHSA,
    TransformerEncoderLayer_Pola,
    TransformerEncoderLayer_Pola_EDFFN_Mona_DyT,
    TransformerEncoderLayer_Pola_SEFFN_Mona_DyT,
    TransformerEncoderLayer_Pola_SEFN,
    TransformerEncoderLayer_Pola_SEFN_Mona,
    TransformerEncoderLayer_Pola_SEFN_Mona_DyT,
    TransformerEncoderLayer_TSSA,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    OBB26,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    ACA,
    BasicBlock,
    Blocks,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvNormLayer,
    ConvTranspose,
    FourierConv,
    Detect,
    FSConv,
    v8Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    MCFGatedFusion,
    # YOLO-RD modules (kept explicit only for non-fusion entries)
    LRPCHead,
    Pose,
    Pose26,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRBottleNeck,
    RTDETRDecoder,
    RTDETROBB,
    SCDown,
    Segment,
    SemSegHead,
    Depth,
    Segment26,
    MRCB,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
    v26Detect,
    get_activation,
)

# Import all fusion modules from the dedicated fusion package to avoid missing symbols
# and keep tasks.py decoupled from individual fusion module names.
from ultralytics.nn.modules.fusion import *  # noqa: F401,F403
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    E2ELoss,
    PoseLoss26,
    SemSegLoss,
    DepthLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
    resolve_detection_loss_profile,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)

# 多模态组件导入 (仅导入已迁移的组件)
try:
    from ultralytics.nn.mm import MultiModalRouter, MultiModalConfigParser
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

_LSCD_IMPORT_ERROR = None
# LSCD 检测头导入 (轻量化共享卷积检测头)
try:
    from ultralytics.nn.Head import (
        Detect_LSCD,
        Segment_LSCD,
        Pose_LSCD,
        OBB_LSCD,
        Conv_GN,
        Scale,
    )
    LSCD_AVAILABLE = True
except ImportError as _err:
    _LSCD_IMPORT_ERROR = _err
    LSCD_AVAILABLE = False
    LOGGER.warning(f"LSCD head import failed: {_err}")
    def _missing_lscd(*args, **kwargs):
        raise RuntimeError(f"LSCD head modules unavailable: {_err}")

    for _name in ['Detect_LSCD', 'Segment_LSCD', 'Pose_LSCD', 'OBB_LSCD', 'Conv_GN', 'Scale']:
        globals()[_name] = _missing_lscd

# LSPCD 检测头导入 (轻量共享部分卷积检测头)
_LSPCD_IMPORT_ERROR = None
try:
    from ultralytics.nn.Head.lspcd import (
        Detect_LSPCD,
        Segment_LSPCD,
        Segment26_LSPCD,
        OBB_LSPCD,
        OBB26_LSPCD,
        Pose_LSPCD,
        Pose26_LSPCD,
    )
    LSPCD_AVAILABLE = True
except ImportError as _err:
    _LSPCD_IMPORT_ERROR = _err
    LSPCD_AVAILABLE = False
    LOGGER.warning(f"LSPCD head import failed: {_err}")
    def _missing_lspcd(*args, **kwargs):
        raise RuntimeError(f"LSPCD head modules unavailable: {_err}")
    for _name in ['Detect_LSPCD', 'Segment_LSPCD', 'Segment26_LSPCD', 'OBB_LSPCD', 'OBB26_LSPCD', 'Pose_LSPCD', 'Pose26_LSPCD']:
        globals()[_name] = _missing_lspcd

# YOLO11 检测头变体导入（不需要编译/不依赖自定义 CUDA 扩展）
# 注意：这些头会被 YAML 直接引用，因此必须在 tasks.py 全局可见（供 parse_model 的 globals()[m] 解析）。
from ultralytics.nn.Head.yolo11_head_variants import (
    Detect_AFPN_P2345,
    Detect_AFPN_P2345_Custom,
    Detect_AFPN_P345,
    Detect_AFPN_P345_Custom,
    DetectAux,
    Detect_Efficient,
    Detect_LADH,
    Detect_LQE,
    Detect_LSCD_LQE,
    Detect_LSCSBD,
    Detect_LSDECD,
    Detect_MultiSEAM,
    Detect_RSCD,
    Detect_SEAM,
    OBB_LADH,
    OBB_LQE,
    OBB_LSCD_LQE,
    OBB_LSCSBD,
    OBB_LSDECD,
    OBB_RSCD,
    Pose_LADH,
    Pose_LQE,
    Pose_LSCD_LQE,
    Pose_LSCSBD,
    Pose_LSDECD,
    Pose_RSCD,
    Segment_Efficient,
    Segment_LADH,
    Segment_LQE,
    Segment_LSCD_LQE,
    Segment_LSCSBD,
    Segment_LSDECD,
    Segment_RSCD,
)

# 双头 + 融合分支聚合头（RGB/IR/Fuse 三检测头）
# 注意：必须在 tasks.py 全局可见，供 parse_model 通过 YAML 模块名解析与构建。
try:
    from ultralytics.nn.mm.dual_head import DualHeadFusionDetect
    # YOLO26 版：内部持有 v26Detect（end2end=True），保留 one2many / one2one 两套输出
    from ultralytics.nn.mm.dual_head_v26 import DualHeadFusionDetectV26
    from ultralytics.nn.mm.reliable_fusion import ReliabilityAwareFusion
    from ultralytics.nn.mm.gated_fusion import GatedReliabilityFusion
    from ultralytics.nn.mm.deform_fusion import DeformableCrossModalFusion
    from ultralytics.nn.mm.partition_fusion import ChannelPartitionFusion
    from ultralytics.nn.mm.misalign_gate import MisalignGatedConcat

    DUAL_HEAD_AVAILABLE = True
except Exception as _dual_head_err:  # pragma: no cover - 依赖缺失时降级
    DualHeadFusionDetect = None
    DualHeadFusionDetectV26 = None
    ReliabilityAwareFusion = None
    GatedReliabilityFusion = None
    DeformableCrossModalFusion = None
    ChannelPartitionFusion = None
    DUAL_HEAD_AVAILABLE = False
    LOGGER.warning(f"DualHeadFusionDetect 导入失败，双头+融合架构不可用: {_dual_head_err}")

# 多路损失（dual_loss）依赖 ultralytics.utils.loss，与 nn.tasks 存在导入环，
# 故仅在 init_criterion() 中按需导入，此处只保留占位名。
DualHeadDetectionLoss = None
DualHeadFusionCriterion = None

_SOEP_IMPORT_ERROR = None
# SOEP 颈部模块导入 (小目标增强金字塔)
try:
    from ultralytics.nn.Neck import (
        SPDConv,
        FGM,
        OmniKernel,
        CSPOmniKernel,
        SNI,
        GSConvE,
        MFM,
    )
    SOEP_AVAILABLE = True
except ImportError as _err:
    _SOEP_IMPORT_ERROR = _err
    SOEP_AVAILABLE = False
    LOGGER.warning(f"SOEP neck import failed: {_err}")
    def _missing_soep(*args, **kwargs):
        raise RuntimeError(f"SOEP neck modules unavailable: {_err}")

    for _name in ['SPDConv', 'FGM', 'OmniKernel', 'CSPOmniKernel', 'SNI', 'GSConvE', 'MFM']:
        globals()[_name] = _missing_soep

_C3K2_IMPORT_ERROR = None

# C3k2 Extraction 模块导入 (C3k2变体模块) — 精确子模块导入，避免聚合导入连带失败
try:
    from ultralytics.nn.extraction.c3k2_variants import (
        # Batch 1
        C3k2_Faster,
        C3k2_PConv,
        C3k2_ODConv,
        C3k2_Faster_EMA,
        C3k2_DBB,
        C3k2_WDBB,
        C3k2_DeepDBB,
        # Batch 2
        C3k2_CloAtt,
        C3k2_SCConv,
        C3k2_ScConv,
        C3k2_EMSC,
        C3k2_EMSCP,
        # Batch 3
        C3k2_ContextGuided,
        C3k2_MSBlock,
        C3k2_EMBC,
        C3k2_EMA,
        # Batch 4
        C3k2_DLKA,
        C3k2_DAttention,
        C3k2_Parc,
        C3k2_DWR,
        C3k2_RFAConv,
        # Batch 5
        C3k2_RFCBAMConv,
        C3k2_RFCAConv,
        C3k2_FocusedLinearAttention,
        C3k2_MLCA,
        C3k2_AKConv,
        # Batch 6
        C3k2_UniRepLKNetBlock,
        C3k2_DRB,
        C3k2_DWR_DRB,
        C3k2_AggregatedAtt,
        C3k2_SWC,
        # Batch 7
        C3k2_iRMB,
        C3k2_iRMB_Cascaded,
        C3k2_iRMB_DRB,
        C3k2_iRMB_SWC,
        C3k2_DynamicConv,
        # Batch 8
        C3k2_GhostDynamicConv,
        C3k2_RVB,
        C3k2_RVB_SE,
        C3k2_RVB_EMA,
        # Batch 9
        C3k2_PKIModule,
        C3k2_PPA,
        C3k2_Faster_CGLU,
        C3k2_Star,
        # Batch 10
        C3k2_Star_CAA,
        C3k2_EIEM,
        C3k2_DEConv,
        # Batch 11
        C3k2_gConv,
        C3k2_AdditiveBlock,
        C3k2_AdditiveBlock_CGLU,
        # Batch 12 - 新迁移
        C3k2_RetBlock,
        C3k2_Heat,
        C3k2_WTConv,
        C3k2_FMB,
        C3k2_MSMHSA_CGLU,
        C3k2_MogaBlock,
        C3k2_SHSA,
        C3k2_SHSA_CGLU,
        C3k2_MutilScaleEdgeInformationEnhance,
        C3k2_MutilScaleEdgeInformationSelect,
        C3k2_FFCM,
        C3k2_SMAFB,
        C3k2_SMAFB_CGLU,
        C3k2_MSM,
        C3k2_HDRAB,
        C3k2_RAB,
        C3k2_LFE,
        C3k2_IDWC,
        C3k2_IDWB,
        C3k2_CAMixer,
        C3k2_LFEM,
        C3k2_LEGM,
    )
    C3K2_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _C3K2_IMPORT_ERROR = _err
    # 为后续逻辑提供占位，避免 NameError
    C3k2_DAttention = C3k2_Parc = C3k2_FocusedLinearAttention = None
    C3K2_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"C3k2 extraction modules import failed: {_err}")

_C2PSA_IMPORT_ERROR = None
# C2PSA Extraction 模块导入 (C2PSA变体模块) — 精确子模块导入
try:
    from ultralytics.nn.extraction.c2psa_variants import (
        # Batch 1 - 基础类和第一批变体
        C2PSA,
        C2fPSA,
        C2BRA,
        BRABlock,
        # Batch 2 - 注意力机制变体
        C2CGA,
        CGABlock,
        C2DA,
        DABlock,
        C2DPB,
        DPBlock,
        C2Pola,
        Polalock,
        C2TSSA,
        TSSAlock,
        # Batch 3 - 注意力机制变体 + 归一化增强变体
        C2ASSA,
        ASSAlock,
        C2MSLA,
        MSLAlock,
        C2PSA_DYT,
        PSABlock_DYT,
        C2TSSA_DYT,
        TSSAlock_DYT,
        C2Pola_DYT,
        Polalock_DYT,
        # Batch 4 - FFN增强变体
        C2PSA_FMFFN,
        PSABlock_FMFFN,
        C2PSA_CGLU,
        PSABlock_CGLU,
        C2PSA_SEFN,
        PSABlock_SEFN,
        C2PSA_SEFFN,
        PSABlock_SEFFN,
        C2PSA_EDFFN,
        PSABlock_EDFFN,
        # Batch 5 - Mona模块化注意力归一化 + 复合增强变体
        C2PSA_Mona,
        PSABlock_Mona,
        C2TSSA_DYT_Mona,
        TSSAlock_DYT_Mona,
        C2TSSA_DYT_Mona_SEFN,
        TSSAlock_DYT_Mona_SEFN,
        C2TSSA_DYT_Mona_SEFFN,
        TSSAlock_DYT_Mona_SEFFN,
        C2TSSA_DYT_Mona_EDFFN,
        TSSAlock_DYT_Mona_EDFFN,
    )
    C2PSA_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _C2PSA_IMPORT_ERROR = _err
    C2PSA_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"C2PSA extraction modules import failed: {_err}")
    # 占位符，确保后续引用时抛出明确错误而非 NameError
    def _missing_c2psa(*args, **kwargs):
        raise RuntimeError(f"C2PSA extraction modules unavailable: {_err}")

    for _name in [
        'C2PSA', 'C2fPSA', 'C2BRA', 'BRABlock', 'C2CGA', 'CGABlock', 'C2DA', 'DABlock', 'C2DPB', 'DPBlock',
        'C2Pola', 'Polalock', 'C2TSSA', 'TSSAlock', 'C2ASSA', 'ASSAlock', 'C2MSLA', 'MSLAlock', 'C2PSA_DYT',
        'PSABlock_DYT', 'C2TSSA_DYT', 'TSSAlock_DYT', 'C2Pola_DYT', 'Polalock_DYT', 'C2PSA_FMFFN', 'PSABlock_FMFFN',
        'C2PSA_CGLU', 'PSABlock_CGLU', 'C2PSA_SEFN', 'PSABlock_SEFN', 'C2PSA_SEFFN', 'PSABlock_SEFFN', 'C2PSA_EDFFN',
        'PSABlock_EDFFN', 'C2PSA_Mona', 'PSABlock_Mona', 'C2TSSA_DYT_Mona', 'TSSAlock_DYT_Mona', 'C2TSSA_DYT_Mona_SEFN',
        'TSSAlock_DYT_Mona_SEFN', 'C2TSSA_DYT_Mona_SEFFN', 'TSSAlock_DYT_Mona_SEFFN', 'C2TSSA_DYT_Mona_EDFFN', 'TSSAlock_DYT_Mona_EDFFN',
    ]:
        globals()[_name] = _missing_c2psa

_SPPF_IMPORT_ERROR = None
# SPPF Extraction 模块导入（SPPF 及空间池化/融合变体） — 精确子模块导入
try:
    from ultralytics.nn.extraction.sppf_base import (
        # Batch 1 - 标准 SPPF 变体
        SPPF_LSKA,
        # Batch 2 - GOLD-YOLO 聚合/融合
        PyramidPoolAgg,
        PyramidPoolAgg_PCE,
        SimFusion_3in,
        SimFusion_4in,
        AdvPoolFusion,
        # Batch 3 - 注入/小波模块
        IFM,
        InjectionMultiSum_Auto_pool,
        WaveletPool,
        WaveletUnPool,
    )
    SPPF_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _SPPF_IMPORT_ERROR = _err
    SPPF_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"SPPF extraction modules import failed: {_err}")
    def _missing_sppf(*args, **kwargs):
        raise RuntimeError(f"SPPF extraction modules unavailable: {_err}")

    for _name in ['SPPF_LSKA', 'PyramidPoolAgg', 'PyramidPoolAgg_PCE', 'SimFusion_3in', 'SimFusion_4in', 'AdvPoolFusion', 'IFM', 'InjectionMultiSum_Auto_pool', 'WaveletPool', 'WaveletUnPool']:
        globals()[_name] = _missing_sppf

# Other Base Extraction 模块导入（独立模块）— MROD-YOLO RFEM
_OTHERBASE_IMPORT_ERROR = None
try:
    from ultralytics.nn.extraction.other_base import (
        RFEM,
    )
    OTHERBASE_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _OTHERBASE_IMPORT_ERROR = _err
    OTHERBASE_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"Other base extraction modules import failed: {_err}")
    def _missing_otherbase(*args, **kwargs):
        raise RuntimeError(f"Other base extraction modules unavailable: {_err}")

    for _name in ['RFEM']:
        globals()[_name] = _missing_otherbase

# LoGStem 模块导入 — 基于高斯-拉普拉斯算子的 Stem 模块
try:
    from ultralytics.nn.extraction.logstem import LoGStem, LoGStem2x, DRFD, Cut
except ImportError:
    def LoGStem(*args, **kwargs):
        raise ImportError("LoGStem module not found. Please check installation.")
    def LoGStem2x(*args, **kwargs):
        raise ImportError("LoGStem2x module not found. Please check installation.")

# 下采样模块导入（Downsample modules）— 条件导入，缺失时提供占位
_DOWNSAMPLE_IMPORT_ERROR = None
try:
    from ultralytics.nn.sample.lawds import LAWDS
    from ultralytics.nn.sample.edge_lawds import EdgeLAWDS
    from ultralytics.nn.sample.freq_lawds import FreqLAWDS
    from ultralytics.nn.sample.hwd import HWD
    from ultralytics.nn.sample.router_lawds import RouterLAWDS
    from ultralytics.nn.sample.v7down import V7DownSampling
    DOWNSAMPLE_AVAILABLE = True
except ImportError as _err:
    _DOWNSAMPLE_IMPORT_ERROR = _err
    DOWNSAMPLE_AVAILABLE = False
    LOGGER.warning(f"Downsample modules import failed: {_err}")
    def _missing_downsample(*args, **kwargs):
        raise RuntimeError(f"Downsample modules unavailable: {_err}")
    for _name in ['LAWDS', 'EdgeLAWDS', 'FreqLAWDS', 'HWD', 'RouterLAWDS', 'V7DownSampling']:
        globals()[_name] = _missing_downsample

# 上采样模块导入（Upsample modules）— 条件导入，缺失时提供占位
_UPSAMPLE_IMPORT_ERROR = None
try:
    from ultralytics.nn.sample.carafe import CARAFE
    from ultralytics.nn.sample.dysample import DySample
    from ultralytics.nn.sample.dsub import DSUB
    from ultralytics.nn.sample.converse2d_up import Converse2D_Up
    from ultralytics.nn.sample.eucb_sc import EUCB_SC
    UPSAMPLE_AVAILABLE = True
except ImportError as _err:
    _UPSAMPLE_IMPORT_ERROR = _err
    UPSAMPLE_AVAILABLE = False
    LOGGER.warning(f"Upsample modules import failed: {_err}")
    def _missing_upsample(*args, **kwargs):
        raise RuntimeError(f"Upsample modules unavailable: {_err}")
    for _name in ['CARAFE', 'DySample', 'DSUB', 'Converse2D_Up', 'EUCB_SC']:
        globals()[_name] = _missing_upsample

# C2f Extraction 模块导入（Batch 01）— 不做自动降级：导入失败应直接显式报错
from ultralytics.nn.extraction.c2f_variants import (
    C2f_CAMixer,
    C2f_Heat,
    C2f_FMB,
    C2f_MSMHSA_CGLU,
    C2f_MogaBlock,
    C2f_SHSA,
    C2f_SHSA_CGLU,
    C2f_HDRAB,
    C2f_RAB,
    C2f_FFCM,
    C2f_SMAFB,
    C2f_SMAFB_CGLU,
    C2f_AP,
    C2f_CSI,
    C2f_gConv,
    C2f_FCA,
    C2f_FDConv,
    C2f_FDT,
    C2f_FourierConv,
    C2f_GlobalFilter,
    C2f_LSBlock,
    C2f_Strip,
    C2f_StripCGLU,
    C2f_wConv,
    # Batch 03
    C2f_FasterFDConv,
    C2f_FasterSFSConv,
    C2f_Faster_KAN,
    C2f_FAT,
    C2f_SMPCGLU,
    C2f_DBlock,
    # Batch 04 (partial)
    C2f_AdditiveBlock,
    C2f_AdditiveBlock_CGLU,
    C2f_IEL,
    C2f_DTAB,
    C2f_PFDConv,
    C2f_SFSConv,
    C2f_PSFSConv,
    C2f_EBlock,
    # Batch 04 (no-CUDA extras)
    C2f_HFERB,
    C2f_JDPM,
    C2f_ETB,
    C2f_SFHF,
    C2f_MSM,
    C2f_ELGCA,
    C2f_ELGCA_CGLU,
    C2f_LEGM,
    C2f_LFEM,
    C2f_ESC,
    C2f_KAT,
)

# Public 即插即用注意力模块导入 — 条件导入，缺失时提供占位
_PUBLIC_ATTN_IMPORT_ERROR = None
try:
    from ultralytics.nn.public.ca import CoordAtt
    from ultralytics.nn.public.deformable_lka import DeformableLKA
    from ultralytics.nn.public.ema_attn import EMA_Attention
    from ultralytics.nn.public.lsk import LSKBlock, LSKBlock_SA
    # 新一批 public 即插即用模块
    from ultralytics.nn.public.adaptive_rectangular_conv import AdaptiveRectangularConv
    from ultralytics.nn.public.cghalfconv import CGHalfConv
    from ultralytics.nn.public.convutr import ConvUtr
    from ultralytics.nn.public.fdsm import FDSM
    from ultralytics.nn.public.fremlp import FreMLP
    from ultralytics.nn.public.gated_bottleneck_conv import GatedBottleneckConv
    from ultralytics.nn.public.inverse_conv2d import InverseConv2DLayer
    from ultralytics.nn.public.mkpconv import MKPConv
    from ultralytics.nn.public.mrfa import MRFA
    from ultralytics.nn.public.poly_kernel_inception import Poly_Kernel_Inception_Block
    from ultralytics.nn.public.shift_module import Shift_Module
    from ultralytics.nn.public.tbfe import TBFE
    from ultralytics.nn.public.fcb import FCB
    PUBLIC_ATTN_AVAILABLE = True
except ImportError as _err:
    _PUBLIC_ATTN_IMPORT_ERROR = _err
    PUBLIC_ATTN_AVAILABLE = False
    LOGGER.warning(f"Public attention modules import failed: {_err}")

    def _missing_public_attn(*args, **kwargs):
        raise RuntimeError(f"Public attention modules unavailable: {_err}")

    for _name in ['CoordAtt', 'DeformableLKA', 'EMA_Attention', 'LSKBlock', 'LSKBlock_SA',
                  'AdaptiveRectangularConv', 'CGHalfConv', 'ConvUtr', 'FDSM', 'FreMLP',
                  'GatedBottleneckConv', 'InverseConv2DLayer', 'MKPConv', 'MRFA',
                  'Poly_Kernel_Inception_Block', 'Shift_Module', 'TBFE', 'FCB']:
        globals()[_name] = _missing_public_attn

# ===== Attention 族类模块导入 =====
_ATTENTION_IMPORT_ERROR = None
try:
    from ultralytics.nn.attention.channel import (
        ACA, BinaryAttention, CASAB, CascadedGroupAtt, CascadedGroupAttention, CBSA,
        DHPF, MaskUnitAttention, MCA, SimAM,
        ChannelTransformerAttention, FCAttention,
        ChannelReductionAttention,
        DeformableInteractiveAttention, DualPoolCoordAttention,
        GCBAM, LocalAttention, SCAM,
        HighFreqDCTChannelAttention,
        SMSSA, PCSA,
    )
    from ultralytics.nn.attention.frequency import (
        ContrastDrivenFeatureAggregation, CTA, SFA, FSSA,
        CPIA_SA, KSFA, FSA,
        FSAS, MFCA,
    )
    from ultralytics.nn.attention.window import (
        BiLevelRoutingAttention, DilatedGCSA, DilatedMWSA, DPWA,
        DWM_MSA, DHOGSA, PatchSA, Token_Selective_Attention,
        LocalRegionSelfAttention, StripeAttentionBlock,
    )
    from ultralytics.nn.attention.global_ import (
        EfficientGlobalSA, LRSA, MALA, TAB,
        GLSA, LWGA, CFBlock,
        CAFM, DynamicRangeHistogramSelfAttention, EASA,
        LSKNetAttention, MLKA, SHSA,
        DynamicSpatialAttention, MaskAttention, MCC,
        Multipole_Attention, SPRS, TopK_Sparse_Attention,
        HDPA, ACFM, ESSA,
    )
    from ultralytics.nn.extraction import (
        ChannelReallocationFFN, HFFE, StarBlock,
        CED, LDE, LRCED, Multi_Scale_Feed_Forward_Network, PCFNBlock,
        SAFM, GSAU, LKAT, DualScaleGatedFFN, DFFN,
        BidomainNonlinearMapping, RCSABlock, MDCR,
    )
    ATTENTION_AVAILABLE = True
except ImportError as _err:
    _ATTENTION_IMPORT_ERROR = _err
    ATTENTION_AVAILABLE = False
    LOGGER.warning(f"Attention family modules import failed: {_err}")

    def _missing_attention(*args, **kwargs):
        raise RuntimeError(f"Attention family modules unavailable: {_err}")

    for _name in [
        'ACA', 'BinaryAttention', 'CASAB', 'CascadedGroupAtt', 'CascadedGroupAttention', 'CBSA',
        'DHPF', 'MaskUnitAttention', 'MCA', 'SimAM',
        'ContrastDrivenFeatureAggregation', 'CTA', 'SFA', 'FSSA',
        'CPIA_SA', 'KSFA', 'FSA',
        'BiLevelRoutingAttention', 'DilatedGCSA', 'DilatedMWSA', 'DPWA',
        'DWM_MSA', 'DHOGSA', 'PatchSA', 'Token_Selective_Attention',
        'EfficientGlobalSA', 'LRSA', 'MALA', 'TAB',
        'GLSA', 'LWGA', 'CFBlock',
        'CAFM', 'ChannelTransformerAttention', 'DynamicRangeHistogramSelfAttention',
        'EASA', 'FCAttention',
        'ChannelReductionAttention', 'FSAS', 'LSKNetAttention', 'MLKA', 'SHSA',
        'ChannelReallocationFFN', 'HFFE', 'StarBlock',
        'CED', 'LDE', 'LRCED', 'Multi_Scale_Feed_Forward_Network', 'PCFNBlock',
        # 新一批 attention 模块
        'DeformableInteractiveAttention', 'DualPoolCoordAttention', 'GCBAM',
        'LocalAttention', 'SCAM', 'DynamicSpatialAttention', 'MaskAttention',
        'MCC', 'Multipole_Attention', 'SPRS', 'TopK_Sparse_Attention',
        'LocalRegionSelfAttention', 'StripeAttentionBlock',
        'HighFreqDCTChannelAttention',
        'SMSSA', 'PCSA', 'MFCA', 'HDPA', 'ACFM', 'ESSA',
        'SAFM', 'GSAU', 'LKAT', 'DualScaleGatedFFN', 'DFFN',
        'BidomainNonlinearMapping', 'RCSABlock', 'MDCR',
    ]:
        globals()[_name] = _missing_attention

# ===== Mamba / SSM 族类模块导入 =====
# 纯 PyTorch 类（GatedCNNBlock_BCHW / MobileMambaModule）与 mamba_ssm 依赖类统一容错导入；
# 缺 mamba_ssm 时 5 个依赖类不导出（占位为 _missing_mamba），但 import 不崩溃。
_MAMBA_IMPORT_ERROR = None
try:
    from ultralytics.nn.mamba import (
        GatedCNNBlock_BCHW,
        MobileMambaModule,
        MambaVisionBlock,
        TransMambaBlock,
        VSSBlock,
        Mamba2Block,
        EVSSM,
    )
    MAMBA_AVAILABLE = True
except ImportError as _err:
    _MAMBA_IMPORT_ERROR = _err
    MAMBA_AVAILABLE = False
    LOGGER.warning(f"Mamba/SSM family modules import failed: {_err}")

    def _missing_mamba(*args, **kwargs):
        raise RuntimeError(f"Mamba/SSM family modules unavailable: {_err}")

    for _name in [
        'GatedCNNBlock_BCHW', 'MobileMambaModule',
        'MambaVisionBlock', 'TransMambaBlock',
        'VSSBlock', 'Mamba2Block', 'EVSSM',
    ]:
        globals()[_name] = _missing_mamba

_NECK_IMPORT_ERROR = None
# Neck 模块导入（AFPN/HSFPN/CFPT/BiPAN 等） — 精确子模块导入
try:
    from ultralytics.nn.Neck import (
        AFPN_P345,
        AFPN_P345_Custom,
        AFPN_P2345,
        AFPN_P2345_Custom,
        HFP,
        SDP,
        SDP_Improved,
        ChannelAttention_HSFPN,
        ELA_HSFPN,
        CA_HSFPN,
        CAA_HSFPN,
        CrossLayerSpatialAttention,
        CrossLayerChannelAttention,
        FreqFusion,
        LocalSimGuidedSampler,
        Fusion,
        SDI,
        CSPStage,
        BiFusion,
        OREPANCSPELAN4,
        SBA,
        EUCB,
        MSDC,
        MSCB,
        CSP_MSCB,
        # MSIA 多尺度迭代聚合模块 (MROD-YOLO)
        MCA,
        MSIA,
        # CTrans 跨尺度通道 Transformer (AAAI 2022)
        ChannelTransformer,
        # HyperComputeModule 超图计算模块 (TPAMI 2025)
        HyperComputeModule,
        HyPConv,
        # FDPN 频率动态金字塔网络
        FocusFeature,
        DynamicFrequencyFocusFeature,
        AlignmentGuidedFocusFeature,
        # GoldYOLO Transformer 聚合注入融合
        # 注意: AdvPoolFusion/IFM/InjectionMultiSum_Auto_pool/PyramidPoolAgg/SimFusion_3in/SimFusion_4in
        # 已在 sppf_base 中注册，此处仅补充 TopBasicLayer
        TopBasicLayer,
        # SlimNeck 轻量化 Neck
        GSConv,
        GSBottleneck,
        GSBottleneckC,
        VoVGSCSP,
        HaarWavelet,
        WFU,
        # ASF 注意力尺度融合 (arXiv 2312.06458)
        Add,
        ScalSeq,
        Zoom_cat,
        asf_attention_model,
    )
    NECK_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _NECK_IMPORT_ERROR = _err
    NECK_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"Neck modules import failed: {_err}")
    def _missing_neck(*args, **kwargs):
        raise RuntimeError(f"Neck modules unavailable: {_err}")

    for _name in [
        'AFPN_P345', 'AFPN_P345_Custom', 'AFPN_P2345', 'AFPN_P2345_Custom', 'HFP', 'SDP', 'SDP_Improved',
        'ChannelAttention_HSFPN', 'ELA_HSFPN', 'CA_HSFPN', 'CAA_HSFPN', 'CrossLayerSpatialAttention',
        'CrossLayerChannelAttention', 'FreqFusion', 'LocalSimGuidedSampler', 'Fusion', 'SDI', 'CSPStage', 'BiFusion',
        'OREPANCSPELAN4', 'SBA', 'EUCB', 'MSCB', 'CSP_MSCB', 'MCA', 'MSIA',
        # CTrans
        'ChannelTransformer',
        # HyperCompute
        'HyperComputeModule', 'HyPConv',
        # FDPN
        'FocusFeature', 'DynamicFrequencyFocusFeature', 'AlignmentGuidedFocusFeature',
        # GoldYOLO
        'AdvPoolFusion', 'IFM', 'InjectionMultiSum_Auto_pool',
        'PyramidPoolAgg', 'SimFusion_3in', 'SimFusion_4in', 'TopBasicLayer',
        # SlimNeck
        'GSConv', 'GSBottleneck', 'GSBottleneckC', 'VoVGSCSP',
        # ASF
        'Add', 'ScalSeq', 'Zoom_cat', 'asf_attention_model',
    ]:
        globals()[_name] = _missing_neck

# ===== Head Class Sets (align with upstream behavior) =====
# 统一以集合方式识别检测/分割/姿态/旋转头，便于在 parse_model/_apply/stride 推断等位置一致处理。
DETECT_CLASS: tuple = (
    Detect,
    v8Detect,
    WorldDetect,
    YOLOEDetect,
    v10Detect,
    v26Detect,
    ImagePoolingAttn,
)
SEGMENT_CLASS: tuple = (
    Segment,
    Segment26,
    YOLOESegment,
)
POSE_CLASS: tuple = (Pose, Pose26)
OBB_CLASS: tuple = (OBB, OBB26)
SEMSEG_CLASS: tuple = (SemSegHead,)
DEPTH_CLASS: tuple = (Depth,)

# 动态扩展 LSCD 系列头
if LSCD_AVAILABLE:
    DETECT_CLASS = DETECT_CLASS + (Detect_LSCD,)
    SEGMENT_CLASS = SEGMENT_CLASS + (Segment_LSCD,)
    POSE_CLASS = POSE_CLASS + (Pose_LSCD,)
    OBB_CLASS = OBB_CLASS + (OBB_LSCD,)

# 动态扩展 LSPCD 系列头（轻量共享部分卷积检测头）
if LSPCD_AVAILABLE:
    DETECT_CLASS = DETECT_CLASS + (Detect_LSPCD,)
    SEGMENT_CLASS = SEGMENT_CLASS + (Segment_LSPCD, Segment26_LSPCD)
    POSE_CLASS = POSE_CLASS + (Pose_LSPCD, Pose26_LSPCD)
    OBB_CLASS = OBB_CLASS + (OBB_LSPCD, OBB26_LSPCD)

# YOLO11 头部变体（无需编译）
DETECT_CLASS = DETECT_CLASS + (
    Detect_AFPN_P345,
    Detect_AFPN_P345_Custom,
    Detect_AFPN_P2345,
    Detect_AFPN_P2345_Custom,
    Detect_Efficient,
    DetectAux,
    Detect_SEAM,
    Detect_MultiSEAM,
    Detect_LADH,
    Detect_LSCSBD,
    Detect_LSDECD,
    Detect_RSCD,
    Detect_LQE,
    Detect_LSCD_LQE,
)
SEGMENT_CLASS = SEGMENT_CLASS + (
    Segment_Efficient,
    Segment_LADH,
    Segment_LSCSBD,
    Segment_LSDECD,
    Segment_RSCD,
    Segment_LQE,
    Segment_LSCD_LQE,
)
POSE_CLASS = POSE_CLASS + (
    Pose_LADH,
    Pose_LSCSBD,
    Pose_LSDECD,
    Pose_RSCD,
    Pose_LQE,
    Pose_LSCD_LQE,
)
OBB_CLASS = OBB_CLASS + (
    OBB_LADH,
    OBB_LSCSBD,
    OBB_LSDECD,
    OBB_RSCD,
    OBB_LQE,
    OBB_LSCD_LQE,
)

# ===== Neck Module Class Sets =====
NECK_CLASS: tuple = ()
if NECK_EXTRACTION_AVAILABLE:
    NECK_CLASS = (
        AFPN_P345,
        AFPN_P345_Custom,
        AFPN_P2345,
        AFPN_P2345_Custom,
        HFP,
        SDP,
        SDP_Improved,
        ChannelAttention_HSFPN,
        ELA_HSFPN,
        CA_HSFPN,
        CAA_HSFPN,
        CrossLayerSpatialAttention,
        CrossLayerChannelAttention,
        FreqFusion,
        LocalSimGuidedSampler,
        Fusion,
        SDI,
        CSPStage,
        BiFusion,
        OREPANCSPELAN4,
        SBA,
        EUCB,
        MSCB,
        CSP_MSCB,
        # CTrans 跨尺度通道 Transformer
        ChannelTransformer,
        # HyperComputeModule 超图计算
        HyperComputeModule,
        HyPConv,
        # FDPN 频率动态金字塔
        FocusFeature,
        DynamicFrequencyFocusFeature,
        AlignmentGuidedFocusFeature,
        # GoldYOLO 聚合注入融合
        AdvPoolFusion,
        IFM,
        InjectionMultiSum_Auto_pool,
        PyramidPoolAgg,
        SimFusion_3in,
        SimFusion_4in,
        TopBasicLayer,
        # SlimNeck 轻量化
        GSConv,
        GSBottleneck,
        GSBottleneckC,
        VoVGSCSP,
        # ASF 注意力尺度融合
        Add,
        ScalSeq,
        Zoom_cat,
        asf_attention_model,
    )

# ===== C3k2 Module Class Sets =====
# 统一以集合方式识别C3k2及其变体，便于在 parse_model/repeat_modules 等位置一致处理。
C3K2_CLASS: tuple = (C3k2,)

# 动态扩展 C3k2 Extraction 系列模块
if C3K2_EXTRACTION_AVAILABLE:
    C3K2_CLASS = C3K2_CLASS + (
        # Batch 1
        C3k2_Faster, C3k2_PConv, C3k2_ODConv, C3k2_Faster_EMA,
        C3k2_DBB, C3k2_WDBB, C3k2_DeepDBB,
        # Batch 2
        C3k2_CloAtt, C3k2_SCConv, C3k2_ScConv,
        C3k2_EMSC, C3k2_EMSCP,
        # Batch 3
        C3k2_ContextGuided, C3k2_MSBlock, C3k2_EMBC, C3k2_EMA,
        # Batch 4
        C3k2_DLKA, C3k2_DAttention, C3k2_Parc, C3k2_DWR, C3k2_RFAConv,
        # Batch 5
        C3k2_RFCBAMConv, C3k2_RFCAConv,
        C3k2_FocusedLinearAttention, C3k2_MLCA, C3k2_AKConv,
        # Batch 6
        C3k2_UniRepLKNetBlock, C3k2_DRB, C3k2_DWR_DRB,
        C3k2_AggregatedAtt, C3k2_SWC,
        # Batch 7
        C3k2_iRMB, C3k2_iRMB_Cascaded, C3k2_iRMB_DRB,
        C3k2_iRMB_SWC, C3k2_DynamicConv,
        # Batch 8
        C3k2_GhostDynamicConv, C3k2_RVB, C3k2_RVB_SE, C3k2_RVB_EMA,
        # Batch 9
        C3k2_PKIModule, C3k2_PPA, C3k2_Faster_CGLU, C3k2_Star,
        # Batch 10
        C3k2_Star_CAA, C3k2_EIEM, C3k2_DEConv,
        # Batch 11
        C3k2_gConv, C3k2_AdditiveBlock, C3k2_AdditiveBlock_CGLU,
        # Batch 12
        C3k2_RetBlock, C3k2_Heat, C3k2_WTConv, C3k2_FMB,
        C3k2_MSMHSA_CGLU, C3k2_MogaBlock, C3k2_SHSA, C3k2_SHSA_CGLU,
        C3k2_MutilScaleEdgeInformationEnhance, C3k2_MutilScaleEdgeInformationSelect, C3k2_FFCM,
        C3k2_SMAFB, C3k2_SMAFB_CGLU,
        C3k2_MSM, C3k2_HDRAB, C3k2_RAB, C3k2_LFE,
        C3k2_IDWC, C3k2_IDWB, C3k2_CAMixer,
        C3k2_LFEM, C3k2_LEGM,  # LEGNet series
    )

# ===== SPPF Variant Class Set =====
SPPF_CLASS: tuple = (SPPF,)
if SPPF_EXTRACTION_AVAILABLE:
    SPPF_CLASS = SPPF_CLASS + (SPPF_LSKA,)

# ===== Upsample Module Class Set =====
# 上采样模块集合，c2=ch[f]，args=[c2, *args]
UPSAMPLE_CLASS: tuple = ()
if UPSAMPLE_AVAILABLE:
    UPSAMPLE_CLASS = (
        CARAFE,
        DySample,
        DSUB,
        Converse2D_Up,
        EUCB_SC,
    )

# ===== C2PSA Module Class Sets =====
# 统一以集合方式识别 C2PSA 及其变体，便于在 parse_model/repeat_modules 等位置一致处理。
C2PSA_CLASS: tuple = ()
if C2PSA_EXTRACTION_AVAILABLE:
    C2PSA_CLASS = (C2PSA, C2fPSA) + (
        # Batch 1/2/3 - 注意力主干变体
        C2BRA, C2CGA, C2DA, C2DPB, C2Pola, C2TSSA, C2ASSA, C2MSLA,
        C2PSA_DYT, C2TSSA_DYT, C2Pola_DYT,
        # Batch 4 - FFN 增强
        C2PSA_FMFFN, C2PSA_CGLU, C2PSA_SEFN, C2PSA_SEFFN, C2PSA_EDFFN,
        # Batch 5 - Mona 复合增强
        C2PSA_Mona, C2TSSA_DYT_Mona, C2TSSA_DYT_Mona_SEFN, C2TSSA_DYT_Mona_SEFFN, C2TSSA_DYT_Mona_EDFFN,
    )

# 万物皆可融 Block 类 — 从 extraction 包导入
from ultralytics.nn.extraction.block_fusion import C3k_Block, C3_Block, C2f_Block, C3k2_Block

# =========================================================================
# 万物皆可融 Block 集合 — 独立于 base_modules，拥有专属 parse_model 分支
# 支持通过 YAML dict 格式动态指定子模块: {'module': '类名', 'selfatt': bool, 'param': {...}}
# 注意: 此集合不合并进 base_modules/repeat_modules
# =========================================================================
# ===== Attention 族类模块集合 =====
# 注意力模块统一特征: c2=c1(不改变通道), 构造函数第一参数为 dim
ATTENTION_CLASS: tuple = ()
if ATTENTION_AVAILABLE:
    ATTENTION_CLASS = (
        # channel (12 + 4 new)
        ACA, BinaryAttention, CASAB, CascadedGroupAtt, CascadedGroupAttention, CBSA,
        DHPF, MaskUnitAttention, MCA, SimAM,
        ChannelTransformerAttention, FCAttention, ChannelReductionAttention,
        DualPoolCoordAttention, GCBAM, LocalAttention, SCAM,
        # frequency (8)
        ContrastDrivenFeatureAggregation, CTA, SFA, FSSA,
        CPIA_SA, KSFA, FSA, FSAS,
        # window (8 + 2 new)
        BiLevelRoutingAttention, DilatedGCSA, DilatedMWSA, DPWA,
        DWM_MSA, DHOGSA, PatchSA, Token_Selective_Attention,
        LocalRegionSelfAttention, StripeAttentionBlock,
        # global (13 + 6 new)
        EfficientGlobalSA, LRSA, MALA, TAB,
        GLSA, LWGA, CFBlock,
        CAFM, DynamicRangeHistogramSelfAttention, EASA,
        LSKNetAttention, MLKA, SHSA,
        DynamicSpatialAttention, MaskAttention, MCC,
        Multipole_Attention, SPRS, TopK_Sparse_Attention,
        # extraction 同通道 block (8)
        ChannelReallocationFFN, HFFE, StarBlock,
        CED, LDE, LRCED, Multi_Scale_Feed_Forward_Network, PCFNBlock,
        HighFreqDCTChannelAttention,
        SMSSA, PCSA,
        MFCA,
        HDPA, ACFM, ESSA,
        SAFM, GSAU, LKAT, DualScaleGatedFFN, DFFN,
        BidomainNonlinearMapping, RCSABlock, MDCR,
    )

# 新一批 public 同通道模块（channel-preserving, c2=c1, 第一参数 dim）
# 与 ATTENTION_CLASS 同模式但走独立元组以保留 public 模块分组语义
_PUBLIC_PRESERVE_CLASS: tuple = ()
if PUBLIC_ATTN_AVAILABLE:
    _PUBLIC_PRESERVE_CLASS = (
        CGHalfConv, FreMLP, GatedBottleneckConv, MRFA, TBFE, MKPConv,
        FCB,  # 傅里叶卷积块（channel-preserving）
    )

# ===== Mamba / SSM 族类模块集合 =====
# 与 ATTENTION_CLASS 同模式: c2=c1(不改变通道), 构造函数第一参数为 dim
# 覆盖纯 PyTorch 类（GatedCNNBlock_BCHW / MobileMambaModule）与 mamba_ssm 依赖类
MAMBA_CLASS: tuple = ()
if MAMBA_AVAILABLE:
    MAMBA_CLASS = (
        # 纯 PyTorch（无 mamba_ssm 依赖）
        GatedCNNBlock_BCHW,
        MobileMambaModule,
        # mamba_ssm 依赖
        MambaVisionBlock,
        TransMambaBlock,
        VSSBlock,
        Mamba2Block,
        EVSSM,
    )

block_repeat_modules = frozenset({C3k_Block, C3_Block, C2f_Block, C3k2_Block})

def apply_fusion_rgb_mask(x, fill="zero", keep=None):
    """Return a new layer-input list with the fusion Concat's RGB half masked.

    Used by ``DetectionModel._predict_once`` when the fusion-neck RGB-half masking
    (route 2) is armed. ``x`` is the *input list* of a fusion ``Concat`` module,
    i.e. ``[rgb_half, ir_half]``, where index 0 is the half coming from the RGB
    neck (see the ``_mm_fusion_concat`` marker written in ``parse_model``).

    Args:
        x (list | tuple): Input list of two feature tensors ``[rgb_half, ir_half]``.
        fill (str): ``'zero'`` zeroes the RGB half (same as the diagnostic script),
            ``'channel_mean'`` replaces it with its own per-channel spatial mean
            (keeps the anti-zero-mean bias of the following BatchNorm meaningful).
        keep (torch.Tensor | None): Per-sample retention flags broadcastable to
            ``(B, 1, 1, 1)``; ``True`` keeps the RGB half for that sample and
            ``False`` masks it. ``None`` (default) masks the whole batch, i.e. the
            original per-batch semantics is preserved bit-exactly.

    Returns:
        (list): ``[masked_rgb_half, ir_half]``, or the original ``x`` when the
            input does not match the expected two-tensor fusion layout.
    """
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return x
    a, b = x
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return x
    if fill == "channel_mean":
        # expand() 返回视图，后续可能的 inplace 操作要求连续张量
        a_masked = a.mean(dim=(2, 3), keepdim=True).expand_as(a).contiguous()
    else:
        a_masked = torch.zeros_like(a)
    if keep is None:
        return [a_masked, b]
    # 逐样本混合：keep=1 取原值、keep=0 取 masked。用乘加而非布尔索引/原地写，
    # 既与批量口径共用同一算子，也不破坏 autograd 链路。
    k = keep.to(device=a.device, dtype=a.dtype).reshape(-1, 1, 1, 1)
    return [a * k + a_masked * (1.0 - k), b]


class BaseModel(torch.nn.Module):
    """
    Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion,
    information display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # ===== MULTIMODAL EXTENSION START - 多模态路由初始化 =====
        mm_router = None
        mm_routing_enabled = False
        mm_input_sources = None

        # Check if this model has a persistent router (RTDETRDetectionModel)
        if hasattr(self, 'mm_router') and self.mm_router is not None:
            # Use persistent router from model initialization
            mm_router = self.mm_router
            mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
            if profile:
                LOGGER.info("MultiModal: 使用持久化router")
        elif MULTIMODAL_AVAILABLE:
            try:
                from ultralytics.nn.mm import MultiModalRouter
                # Create temporary router for other model types
                config_dict = getattr(self, 'yaml', None)
                mm_router = MultiModalRouter(config_dict, verbose=profile)
                mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
                if profile:
                    LOGGER.info("MultiModal: 使用临时router")
            except Exception as e:
                if profile:
                    LOGGER.warning(f"MultiModal routing initialization failed: {e}")
        # ===== MULTIMODAL EXTENSION END =====

        # ===== FUSION-NECK RGB-HALF MASKING START - 逐 batch / 逐样本抽样（路线二）=====
        # 抽样粒度由 _mm_fusion_mask_scope 决定（默认 batch）：
        #   batch  —— 每个 forward 独立掷一次硬币，整批（全部融合 Concat 层）共用同一
        #             决定。若逐层独立抽样，同一 batch 内 P3/P4/P5 可能有的屏蔽有的
        #             不屏蔽，融合头仍能从"未屏蔽那一层"拿到 RGB 信息，退化通路永远
        #             得不到惩罚，屏蔽失效。三层同屏蔽才能保证"本 batch 的融合头只能
        #             读 IR 半边"。
        #   sample —— 逐样本掷签，同一 batch 内混有屏蔽 / 未屏蔽帧。融合头在同一次
        #             forward 内就必须同时支持 RGB 与 IR，无法把 RGB 当捷径专精；
        #             实测 batch 口径在 p=0.1/0.2 下力度不足（伪融合依旧），逐样本是
        #             针对该失效机理的改动。三层仍按样本统一屏蔽，理由同上。
        #
        # 三个必要条件缺一不可：
        #   self.training               -> eval/val/predict 恒不激活，既有 best.pt
        #                                  推理行为逐位不变
        #   _mm_fusion_mask_enabled     -> 由 trainer 在 setup 后显式开启（默认无）
        #   _mm_fusion_mask_p > 0       -> 概率门控，0 表示关闭
        # 构建期的 stride 推断前向虽然处于 train 模式，但那时 trainer 尚未开启开关，
        # 因此不会被污染（见 DetectionModel.__init__ 的多头 stride 推断）。
        #
        # 注：本段必须自带完整初始化，不可依赖上面的 try 分支 —— Python 的 try
        # 不构成作用域，若赋值只发生在 try 内，except 后该名字仍未绑定。
        self._mm_fusion_mask_current = False
        self._mm_fusion_mask_keep = None
        # D2-P1 通路级 dropout：逐样本掩码（True=该样本强置 gate=1，fused 走纯 IR）。
        # 三个融合算子层共用同一份掩码（与融合颈屏蔽同理由：单层独立抽签会让未屏蔽层
        # 提供 RGB 捷径，退化通路得不到惩罚）。
        # 三个必要条件：self.training / _mm_gate_drop_p > 0 / 显式由 trainer 开启。
        _gd_p = float(getattr(self, "_mm_gate_drop_p", 0.0) or 0.0)
        self._mm_gate_drop_mask = None
        if self.training and _gd_p > 0.0 and torch.is_tensor(x):
            self._mm_gate_drop_mask = torch.rand(x.shape[0], device=x.device) < _gd_p
        # B3：本 batch 是否发生了「融合颈 RGB 半边屏蔽」。供 DualHeadFusionCriterion
        # 判断是否启用「fuse←ir 蒸馏」（只在 RGB 被移除的样本上蒸馏，才逼融合用 IR）。
        self._mm_fusion_drop_active = False
        # B4-b：逐样本屏蔽掩码（True=该样本保留 RGB；False=RGB 被屏蔽）。
        # 供 criterion 做「mask 条件下 fuse 头改用 IR 标签监督」的逐样本切换：
        # 被屏蔽样本的融合分支只能读 IR，此时用 RGB 标签监督它等于要求"无米之炊"，
        # 只会被优化成常量背景；改用 IR 标签才能真正建立"从 IR 解码目标"的压力。
        self._mm_fusion_drop_per_sample = None
        _fm_p = float(getattr(self, "_mm_fusion_mask_p", 0.0) or 0.0)
        if getattr(self, "_mm_fusion_mask_enabled", False) and self.training and _fm_p > 0.0:
            _fm_scope = str(getattr(self, "_mm_fusion_mask_scope", "batch") or "batch")
            if _fm_scope == "sample" and torch.is_tensor(x):
                # 逐样本抽签：同一 batch 内混有「屏蔽 / 未屏蔽」帧，融合头在同一次
                # forward 里就必须同时支持 RGB 与 IR，无法把 RGB 当捷径专精。
                # 逐 batch 口径（整批同屏蔽）在 p=0.1/0.2 下力度不足，是伪融合的成因。
                self._mm_fusion_mask_keep = torch.rand(x.shape[0], device=x.device) >= _fm_p
                self._mm_fusion_mask_current = True
                self._mm_fusion_drop_active = bool((~self._mm_fusion_mask_keep).any().item())
                # 逐样本：keep=True 表示保留 RGB（未屏蔽）；criterion 用 ~keep 判定"被屏蔽"
                self._mm_fusion_drop_per_sample = ~self._mm_fusion_mask_keep
            else:
                self._mm_fusion_mask_current = bool(torch.rand(1).item() < _fm_p)
                self._mm_fusion_drop_active = self._mm_fusion_mask_current
                if self._mm_fusion_drop_active and torch.is_tensor(x):
                    # batch 口径：整批同屏蔽，等价于"每个样本都被屏蔽"
                    self._mm_fusion_drop_per_sample = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
                elif torch.is_tensor(x):
                    self._mm_fusion_drop_per_sample = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        # 同步到检测头：criterion 持有的是 head（最后一层），不持有外层模型，
        # 这样 B3 蒸馏能读到屏蔽状态而又不在 criterion 里制造对外层模型的引用环
        # （引用环会让 trainer 的 EMA deepcopy 卡死）。
        _head = self.model[-1] if len(self.model) else None
        if _head is not None:
            try:
                _head._mm_fusion_drop_active = self._mm_fusion_drop_active
                _head._mm_fusion_drop_per_sample = self._mm_fusion_drop_per_sample
            except Exception:
                pass
        # ===== FUSION-NECK RGB-HALF MASKING END =====
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            # ===== MULTIMODAL EXTENSION START - 多模态层级路由处理 =====
            # Apply multimodal routing if enabled and module has MM attributes
            if mm_routing_enabled and mm_input_sources and mm_router:
                routed_x = mm_router.route_layer_input(x, m, mm_input_sources, profile)
                if routed_x is not None:
                    x = routed_x

            # Check for spatial reset requirement
            if mm_router and hasattr(m, '_mm_spatial_reset') and m._mm_spatial_reset:
                x = mm_router.reset_spatial_input(x, m, mm_input_sources, profile)
            # ===== MULTIMODAL EXTENSION END =====

            # ===== FUSION-NECK RGB-HALF MASKING START - 屏蔽执行（路线二）=====
            # 位置有意选在 m(x) 之前、模态路由之后：这里 x 是喂给 Concat 的
            # 输入列表 [rgb_half, ir_half]，与诊断脚本 forward_hook 的遮挡口径
            # 完全一致（同一位置、同一 torch.cat([zeros_like(a), b]) 语义）。
            # 只改写 x 不改模块，故屏蔽状态不残留到下一个 batch / 下一次推理。
            if self._mm_fusion_mask_current and getattr(m, "_mm_fusion_concat", False):
                x = apply_fusion_rgb_mask(
                    x,
                    getattr(self, "_mm_fusion_mask_fill", "zero"),
                    getattr(self, "_mm_fusion_mask_keep", None),
                )
            # ===== FUSION-NECK RGB-HALF MASKING END =====

            # ===== 通路级 DROPOUT（D2-P1）：逐样本强置融合算子 gate=1 =====
            # 与上方的「融合颈 RGB 半边屏蔽」是两种不同的干预：
            #   - 上方作用于融合节点**输入端**（x=[rgb_half, ir_half] 置零 rgb_half），
            #     但 rgb_proj(0)=bias≠0，RGB 分量仍留在 fused 里，gate 仍可依赖它 -> 被绕过；
            #   - 本段按逐样本掩码在算子内部把 gate 强置为 1，即 fused = ir_attended
            #     （base_mode="rgb" 时），直接消去 fused 中的 RGB 项，物理上无法绕过，
            #     且与探针 drop_rgb_term 在同一数学位置。
            # 只在 train 模式且 _mm_gate_drop_p > 0 时激活；eval/val/predict 恒不激活。
            if DeformableCrossModalFusion is not None and isinstance(m, DeformableCrossModalFusion):
                m._mm_gate_drop_mask = self._mm_gate_drop_mask if self.training else None
            # ===== 通路级 DROPOUT END =====

            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
                if isinstance(m, YOLOEDetect) and hasattr(self, "pe"):
                    m.fuse(self.pe.to(next(self.model.parameters()).device))
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()/Segment()/Pose()/OBB()
        heads = DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
        if isinstance(m, heads):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        prune_info = getattr(model, "prune_info", None)
        if isinstance(prune_info, dict) and prune_info.get("is_pruned"):
            # shape-intersected loading would silently drop most pruned weights
            raise RuntimeError(
                "Refusing to shape-intersect load a pruned checkpoint into a YAML-built skeleton "
                "(weights would be silently dropped). Use finetrain() for pruned models, or rebuild "
                "via ultralytics.nn.mm.pruning.spec.rebuild_pruned_model()."
            )
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        det_loss_vec, det_items = self.criterion(preds, batch)

        return det_loss_vec, det_items

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


def _extract_stride_features(out, head, end2end=False):
    """Normalize detection head outputs to feature-map lists for stride inference."""
    if isinstance(out, dict):
        if end2end and "one2many" in out:
            out = out["one2many"]
        if isinstance(out, dict) and "feats" in out:
            out = out["feats"]

    # Segment/Pose/OBB style training outputs package detection features in the first tuple item.
    if isinstance(out, tuple):
        out = out[0]

    if not isinstance(out, (list, tuple)):
        head_name = head.__class__.__name__
        raise TypeError(
            f"Stride inference expected list/tuple feature maps from {head_name}, "
            f"but got {type(out).__name__}"
        )
    return out


class DetectionModel(BaseModel):
    """
    YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass,
    augmented inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 传递多模态路由器（如果存在）
        if hasattr(self.model, 'multimodal_router'):
            self.multimodal_router = self.model.multimodal_router
        else:
            self.multimodal_router = None
        # Persist router for runtime ablation/filling so BaseModel forward can reuse it
        self.mm_router = self.multimodal_router if self.multimodal_router is not None else None

        # Build strides
        m = self.model[-1]  # Detect()/Segment()/Pose()/OBB()/...
        heads = DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
        if DUAL_HEAD_AVAILABLE and isinstance(m, (DualHeadFusionDetect, DualHeadFusionDetectV26)):
            # ===== 多头（双头+融合分支）：需为每个头分别推断 stride =====
            # 单头时代码只给 model[-1] 赋 stride；多头下三个头都是独立模块，
            # 若不逐个赋值，前两个头的 stride 会保持 torch.zeros(nl)，导致 anchor 退化。
            s = 256  # 2x min stride
            # 必须保持 training=True：Detect.forward 依据自身 training 标志决定返回
            # 「训练态特征图列表」还是「推理态 (y, feats) 元组」，stride 推断需要前者。
            self.model.train()
            for module in m._branch_modules.values():
                module.inplace = self.inplace
                module.training = True  # 确保各头返回特征图列表
            self.stride = torch.zeros(m.nl)
            try:
                out = self.forward(torch.zeros(1, ch, s, s))
                # 训练态下聚合头返回 {"rgb": [f...], "ir": [f...], "fuse": [f...]}
                # （B1 变体额外含 "fuse_ir"）。注意此处走的是 forward() 而非
                # _predict_once()，NoneProfiler 场景下 forward 只返回最后一层输出，
                # 故下面需兼容「非 dict」的情形。
                for name, module in m._branch_modules.items():
                    branch_out = out[name] if isinstance(out, dict) and name in out else None
                    # 统一解包成 one2many 特征图列表：
                    #   YOLO11 Detect    训练态 -> List[Tensor]；推理态 -> (y, List[Tensor])
                    #   YOLO26 v26Detect 训练态 -> {"one2many": List[Tensor], "one2one": ...}
                    #                    推理态 -> (y, {...})
                    # stride 只与空间尺寸有关，两个分支尺寸相同，取 one2many 即可。
                    if isinstance(branch_out, tuple):  # 推理态 (y, feats)
                        branch_out = branch_out[1]
                    if isinstance(branch_out, dict):
                        branch_out = branch_out.get("one2many")
                    if not isinstance(branch_out, (list, tuple)) or len(branch_out) == 0:
                        # 非 dict 输出（如多模态路由下 forward 被包裹）时，退回用
                        # 共享 stride 兜底，避免因无法解析而构建失败。
                        default_stride = torch.tensor([float(s / (s / (8 * (2 ** i)))) for i in range(m.nl)])
                        module.stride = default_stride
                        LOGGER.warning(
                            "多头 stride 推断：分支 '%s' 输出非预期类型(%s)，回退默认 stride %s",
                            name, type(branch_out).__name__, default_stride.tolist())
                        continue
                    module.stride = torch.tensor([s / x.shape[-2] for x in branch_out])
                # 以 RGB 分支 stride 作为模型整体 stride
                self.stride = m.rgb_detect.stride
            except RuntimeError as e:
                # 与单头保持一致的降级策略：构建期 GPU-only 算子无法在 CPU 上前向
                if "is_cuda" in str(e) or "to be true" in str(e):
                    default_stride = torch.tensor([float(8 * (2 ** i)) for i in range(m.nl)])
                    for module in m._branch_modules.values():
                        module.stride = default_stride.clone()
                    self.stride = default_stride
                    LOGGER.info("多头 stride 推断前向跳过(构建期 GPU-only 算子 CPU 限制)，回退默认 stride %s",
                                default_stride.tolist())
                else:
                    raise
            m.stride = self.stride
            self.model.train()
            m.bias_init()
        elif isinstance(m, heads):  # includes all Detect/Segment/Pose/OBB subclasses (e.g., LSCD variants)
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                out = self.forward(x)
                return _extract_stride_features(out, m, self.end2end)

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            try:
                _stride_out = _forward(torch.zeros(1, ch, s, s))
                # DetectAux：训练输出包含辅助分支（2*nl 个特征图），stride 只应基于主分支 nl 个特征图计算
                if hasattr(m, "dfl_aux") and isinstance(_stride_out, (list, tuple)) and hasattr(m, "nl"):
                    nl = int(getattr(m, "nl"))
                    if len(_stride_out) >= nl:
                        _stride_out = _stride_out[:nl]
                m.stride = torch.tensor([s / x.shape[-2] for x in _stride_out])  # forward
            except RuntimeError as e:
                # GPU-only 算子(如 mamba selective_scan_fn/causal_conv1d_fn)在构建期 CPU stride 推断前向失败，
                # 回退默认 stride；model.to(device) 后训练/推理用真实 GPU 数据，Detect 头会重算 stride
                if "is_cuda" in str(e) or "to be true" in str(e):
                    _nl = int(getattr(m, "nl", 3))
                    m.stride = torch.tensor([float(8 * (2 ** i)) for i in range(_nl)])
                    LOGGER.info("%s stride 推断前向跳过(构建期 GPU-only 算子 CPU 限制)，回退默认 stride %s",
                                type(m).__name__, m.stride.tolist())
                else:
                    raise
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            # 初始化检测头偏置（若实现提供）
            if hasattr(m, "bias_init") and callable(getattr(m, "bias_init")):
                m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        Clip YOLO augmented inference tails.

        Args:
            y (List[torch.Tensor]): List of detection tensors.

        Returns:
            (List[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        args = getattr(self, "args", None)
        is_end2end = getattr(self, "end2end", False)
        loss_cfg = resolve_detection_loss_profile(args, self)

        def _f(key, default):
            """从 args 取 float 超参，**不吞掉 0.0**。

            不能写成 ``float(getattr(args, key, default) or default)``：
            0.0 在 Python 里是 falsy，``0.0 or 1.0`` 会返回 1.0，
            于是「显式把某分支权重设为 0」会被静默改回默认值。
            """
            v = getattr(args, key, None)
            return default if v is None else float(v)

        # ===== 双头 + 融合分支：走多路检测 loss（含可选的 DFSC）=====
        # 该架构下 preds 为 {"rgb": ..., "ir": ..., "fuse": ...}，且 IR 头必须用
        # batch["bboxes_ir"] 监督，标准 v8DetectionLoss 无法处理，故单独分流。
        m_last = self.model[-1] if len(self.model) else None
        if DUAL_HEAD_AVAILABLE and isinstance(m_last, (DualHeadFusionDetect, DualHeadFusionDetectV26)):
            # YOLO26 版聚合头（DualHeadFusionDetectV26）内部持有 v26Detect，其分支输出是
            # {"one2many": ..., "one2one": ...} 而非 List[Tensor]。DualHeadFusionCriterion
            # 通过 `_one2many_of()` / `_one2one_of()` 统一解包，故两类聚合头共用同一损失实现。
            #
            # 端到端头必须同时监督 one2many 与 one2one：只算 one2many 会让 NMS-free 分支
            # 永远没有梯度，推理输出是未训练的随机结果。故此处按 is_end2end 透传
            # ProgLoss / STAL / stride 增强三项 YOLO26 原生配置。
            # YOLO11 聚合头（end2end=False）一律取 False/None，保证与改造前逐位一致。
            _end2end = bool(is_end2end)
            if _end2end:
                if loss_cfg["profile"] == "yolo26":
                    _prog = bool(loss_cfg["use_prog"])
                    _topk2 = loss_cfg["tal_topk2"]
                    _tal_stride = loss_cfg["tal_stride"]
                else:
                    _prog = bool(getattr(args, "loss_prog", True))
                    _topk2 = 1 if bool(getattr(args, "loss_stal", True)) else None
                    _tal_stride = [8, 16] if bool(getattr(args, "loss_stride_enhance", True)) else None
            else:
                _prog, _topk2, _tal_stride = False, None, None
            # 延迟导入：dual_loss 依赖 utils.loss，与 nn.tasks 构成导入环
            from ultralytics.nn.mm.dual_loss import DualHeadFusionCriterion as _DHFC

            return _DHFC(
                self,
                branch_weights={
                    "rgb": _f("mm_dual_w_rgb", 1.0),
                    "ir": _f("mm_dual_w_ir", 1.0),
                    "fuse": _f("mm_dual_w_fuse", 1.0),
                    # B1：融合头第二坐标系（IR）分支。默认 1.0；仅当模型挂载了
                    # fuse_ir 检测头（-b1 配置）时才会真正参与 loss 计算。
                    "fuse_ir": _f("mm_dual_w_fuse_ir", 1.0),
                },
                dfs_weight=_f("mm_dfs_weight", 0.0),
                dfs_mode=str(getattr(args, "mm_dfs_mode", "cls") or "cls"),
                dfs_temperature=_f("mm_dfs_temperature", 1.0),
                consist_weight=_f("mm_dual_consist_weight", 0.0),
                # B1-2：fuse / fuse_ir 双坐标系语义一致性。默认 0（关闭）；
                # 仅在模型挂载 fuse_ir 头时才有实际作用。
                fuse_ir_consist_weight=_f("mm_dual_w_fuse_ir_consist", 0.0),
                # B3：drop-RGB 条件下的 fuse←ir 蒸馏。默认 0（关闭）；
                # 仅当同时开启融合颈 RGB 屏蔽（mm_fusion_rgb_mask_p>0）时才有实际作用。
                fuse_ir_distill_weight=_f("mm_dual_w_fuse_ir_distill", 0.0),
                # P0：门控激活正则。惩罚 GatedReliabilityFusion 门控塌缩到中性 0.5。
                # 默认 0（关闭）；仅在 --config dual-fusion-gated 下有意义。
                gate_reg_weight=_f("mm_gate_reg_weight", 0.0),
                gate_reg_target=_f("mm_gate_reg_target", 0.15),
                # 门控锚定正则（G1）：把 alpha 拉向由注入位移标定的目标曲线。
                alpha_anchor_weight=_f("mm_alpha_anchor_weight", 0.0),
                alpha_min=_f("mm_alpha_min", 0.2),
                # 判别实验（2026-10-14）：fuse 头无条件改用 IR 坐标系标签监督。
                # 用于检验伪融合瓶颈是否在监督坐标系（详见 dual_loss._resolve_fuse_label_keys）。
                fuse_ir_label_always=bool(getattr(args, "mm_fuse_ir_label_always", False)),
                tal_topk=10,
                # 标签分配的定位相似度：默认 "iou"（保持既有行为），
                # "geoshape" 启用 COXNet 的 GeoShape 相似度（对微小目标的几像素位移更鲁棒）。
                tal_metric=str(getattr(args, "loss_assign_metric", "iou") or "iou").lower(),
                # YOLO26 端到端专用：one2one 分支监督 + ProgLoss 渐进权重 + STAL/stride 增强
                end2end=_end2end,
                prog_loss=_prog,
                tal_topk2=_topk2,
                tal_stride=_tal_stride,
            )

        if loss_cfg["profile"] == "yolo26":
            if is_end2end:
                if loss_cfg["use_prog"]:
                    return E2ELoss(
                        self,
                        loss_fn=v8DetectionLoss,
                        tal_stride=loss_cfg["tal_stride"],
                        tal_metric=loss_cfg["tal_metric"],
                    )
                return E2EDetectLoss(
                    self,
                    tal_topk2=loss_cfg["tal_topk2"],
                    tal_stride=loss_cfg["tal_stride"],
                    tal_metric=loss_cfg["tal_metric"],
                )
            return v8DetectionLoss(
                self,
                tal_topk=10,
                tal_topk2=loss_cfg["tal_topk2"],
                tal_stride=loss_cfg["tal_stride"],
                tal_metric=loss_cfg["tal_metric"],
            )

        return (
            E2EDetectLoss(self, tal_metric=loss_cfg["tal_metric"])
            if is_end2end
            else v8DetectionLoss(self, tal_metric=loss_cfg["tal_metric"])
        )

    def distill_forward(self, batch):
        """Forward pass that returns raw predictions alongside the standard loss.

        This method is only called during distillation training.  It reuses the
        existing ``loss()`` logic so the detection loss computation stays in sync
        with the non-distillation path.

        The returned *preds* are the full detection-head output package used by
        **output-level distillation** (teacher-level, not layer-index-level).
        For feature-level distillation, intermediate features are collected
        separately via hooks registered by the ``DistillRuntime``.

        Args:
            batch (dict): Training batch dict (must contain ``'img'`` etc.).

        Returns:
            tuple: ``(loss, loss_items, preds)`` where *preds* are the raw
                detection-head outputs before loss computation.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"])
        det_loss_vec, det_items = self.criterion(preds, batch)
        return det_loss_vec, det_items, preds


class OBBModel(DetectionModel):
    """
    YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized
    loss computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo11n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """
    YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized
    loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class SemSegModel(BaseModel):
    """YOLOMM semantic segmentation model that returns dense logits.

    Inherits BaseModel directly, not DetectionModel, because:
    - SemSegHead is not in DETECT_CLASS / SEGMENT_CLASS / ... tuples,
      so DetectionModel's stride inference would not work for SemSegHead.
    - Manual router propagation + forward trial stride inference avoids
      inheriting anchors/NMS/end2end/DFL fields that SemSeg never uses.
    """

    def __init__(self, cfg="yolo11n-mm-mid-semseg.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        self.yaml["channels"] = ch
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        if self.yaml["nc"] > 255:
            raise ValueError(
                f"SemSeg nc={self.yaml['nc']} exceeds maximum 255. "
                f"uint8 PNG labels cannot represent class IDs > 255."
            )
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        self.multimodal_router = getattr(self.model, "multimodal_router", None)
        self.mm_router = self.multimodal_router if self.multimodal_router is not None else None

        s = 256
        self.model.eval()
        try:
            with torch.no_grad():
                y = self.forward(torch.zeros(1, ch, s, s))
        except Exception as e:
            raise RuntimeError(
                f"SemSegModel forward failed during stride inference. "
                f"Check YAML config and input channels (ch={ch}). "
                f"Ensure dataset_config with correct Xch is injected via trainer.get_model(). "
                f"Original error: {e}"
            ) from e
        if y.ndim != 4:
            raise ValueError(f"SemSegModel output must be [B,nc,H,W], got {tuple(y.shape)}")
        self.register_buffer("stride", torch.tensor([s / int(y.shape[-2])]), persistent=True)
        self.model.train()
        initialize_weights(self)
        if verbose:
            self.info()

    def init_criterion(self):
        return SemSegLoss(self)


class DepthModel(BaseModel):
    """YOLOMM dense depth model. Inherits BaseModel, not DetectionModel."""

    def __init__(self, cfg="yolo11n-mm-mid-depth.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if nc and nc != self.yaml.get("nc", 1):
            LOGGER.info(f"Overriding model.yaml nc={self.yaml.get('nc')} with nc={nc}")
            self.yaml["nc"] = nc
        self.yaml.setdefault("nc", 1)
        if not has_multimodal_config(self.yaml):
            raise ValueError(
                "YOLOMM DepthModel requires multimodal RGB/X routing in YAML. "
                "Official yolo26-depth weights/yaml are not supported."
            )
        # Model._new calls DepthModel(cfg_dict) with default ch=3; mid-fusion YAML needs fused RGB+X.
        if ch == 3:
            try:
                from ultralytics.nn.mm.utils import resolve_multimodal_channel_contract

                ch = int(
                    resolve_multimodal_channel_contract(
                        data=self.yaml.get("dataset_config"),
                        model_cfg=self.yaml,
                    ).fused_ch
                )
            except Exception:
                ch = 6
        self.yaml["channels"] = ch
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {0: "depth"}
        self.inplace = self.yaml.get("inplace", True)

        self.multimodal_router = getattr(self.model, "multimodal_router", None)
        self.mm_router = self.multimodal_router if self.multimodal_router is not None else None

        s = 256
        self.model.eval()
        try:
            with torch.no_grad():
                y = self.forward(torch.zeros(1, ch, s, s))
        except Exception as e:
            raise RuntimeError(
                f"DepthModel forward failed during stride inference. "
                f"Check YAML and input channels (ch={ch}). Original error: {e}"
            ) from e
        if isinstance(y, dict):
            y = y["depth"]
        if y.ndim != 4 or y.shape[1] != 1:
            raise ValueError(f"DepthModel output must be [B,1,H,W], got {tuple(y.shape)}")
        self.register_buffer("stride", torch.tensor([s / int(y.shape[-2])]), persistent=True)
        self.model.train()
        initialize_weights(self)
        if verbose:
            self.info()

    def init_criterion(self):
        return DepthLoss(self)


class PoseModel(DetectionModel):
    """
    YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized
    loss computation for keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """
        Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        from ultralytics.nn.modules.head import Pose26

        last = self.model[-1]
        if isinstance(last, Pose26):
            return PoseLoss26(self)
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """
    YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks,
    providing model initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo11n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """
        Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """
        Update a TorchVision classification model to class count 'n' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        
        # ===== MULTIMODAL EXTENSION START - 持久化MultiModalRouter =====
        self.mm_router = None
        self.mm_routing_enabled = False
        self.mm_input_sources = None
        
        if MULTIMODAL_AVAILABLE:
            try:
                from ultralytics.nn.mm import MultiModalRouter
                # Create persistent router with model configuration
                config_dict = getattr(self, 'yaml', None)
                self.mm_router = MultiModalRouter(config_dict, verbose=verbose)
                if verbose:
                    LOGGER.info("RTDETRDetectionModel: 持久化MultiModalRouter已创建")
            except Exception as e:
                if verbose:
                    LOGGER.warning(f"RTDETRDetectionModel: MultiModalRouter初始化失败: {e}")
        # ===== MULTIMODAL EXTENSION END =====

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(model=self)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions.

        Returns:
            loss_sum (torch.Tensor): Total loss value.
            loss_items (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def distill_forward(self, batch):
        """Forward pass that returns raw predictions alongside the standard loss.

        Reuses ``loss()`` logic to keep detection loss computation in sync.

        The returned *preds* are the full RT-DETR detection-head output package
        used by **output-level distillation** (teacher-level, not layer-index-level).
        For feature-level distillation, intermediate features are collected
        separately via hooks registered by the ``DistillRuntime``.

        Args:
            batch (dict): Training batch dict.

        Returns:
            tuple: ``(loss, loss_items, preds)`` where *preds* are the raw
                RT-DETR detection-head outputs.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        loss_scalar = sum(loss.values())
        loss_items = torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )
        return loss_scalar, loss_items, preds

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        # ===== MULTIMODAL EXTENSION START - 多模态路由初始化 =====
        mm_router = self.mm_router
        mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
        # ===== MULTIMODAL EXTENSION END =====
        
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
            # ===== MULTIMODAL EXTENSION START - 多模态层级路由处理 =====
            # Apply multimodal routing if enabled and module has MM attributes
            if mm_routing_enabled and mm_input_sources and mm_router:
                routed_x = mm_router.route_layer_input(x, m, mm_input_sources, profile)
                if routed_x is not None:
                    x = routed_x

            # Check for spatial reset requirement
            if mm_router and hasattr(m, '_mm_spatial_reset') and m._mm_spatial_reset:
                x = mm_router.reset_spatial_input(x, m, mm_input_sources, profile)
            # ===== MULTIMODAL EXTENSION END =====
            
            if profile:
                self._profile_one_layer(m, x, dt)
            # ===== 多输出主干（单模块多输出）支持：与 RTDETR-main 对齐 =====
            if hasattr(m, "backbone"):
                base_idx = len(y)
                feats = m(x)
                if not isinstance(feats, (list, tuple)):
                    raise TypeError(
                        f"backbone 模块 '{m.__class__.__name__}' 输出必须为 list/tuple，当前={type(feats).__name__}"
                    )
                feats = list(feats)
                if len(feats) > 5:
                    raise ValueError(
                        f"backbone 模块 '{m.__class__.__name__}' 输出层数过多：len={len(feats)}，期望<=5"
                    )
                for _ in range(5 - len(feats)):
                    feats.insert(0, None)
                for local_i, feat in enumerate(feats):
                    global_i = base_idx + local_i
                    y.append(feat if global_i in self.save else None)
                x = feats[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class RTDETRMMOBBModel(RTDETRDetectionModel):
    """RTDETRMM OBB model handling decoder angle outputs and 5D xywhr targets."""

    def __init__(self, cfg="rtdetr-r18-mm-obb.yaml", ch=6, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        from ultralytics.models.utils.loss import RTDETROBBLoss

        return RTDETROBBLoss(model=self)

    def distill_forward(self, batch):
        """RTDETRMM OBB does not support detect-style distillation."""
        raise NotImplementedError(
            "RTDETRMM OBB distill is intentionally disabled in this release. "
            "Disable distill or implement an OBB-aware distill adapter."
        )

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gtgroups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gtgroups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, dec_angles, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]

        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
            _dn_angles, dec_angles = torch.split(dec_angles, dn_meta["dn_num_split"], dim=2)

        loss = self.criterion(
            (dec_bboxes, dec_scores, dec_angles, enc_bboxes, enc_scores),
            targets,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox", "loss_angle"]],
            device=img.device,
        )


class WorldModel(DetectionModel):
    """
    YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based
    class specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x) or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """
    YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts,
    supporting both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings cooperated with reprta module.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """
        Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (List[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """
        Get fused vocabulary layer from the model.

        Args:
            names (list): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (List[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """
        Get class positional embeddings.

        Args:
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """
    YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts,
    providing specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """
    Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through
    model averaging or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """
        Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def _reject_official_depth_checkpoint(model):
    """Reject official YOLO26-Depth weights that have no YOLOMM RGB/X routing."""
    is_depth = isinstance(model, DepthModel) or getattr(model, "task", None) == "depth"
    if not is_depth:
        with contextlib.suppress(Exception):
            is_depth = any(isinstance(m, Depth) for m in model.modules())
    if not is_depth:
        yaml_cfg = getattr(model, "yaml", None)
        if isinstance(yaml_cfg, dict):
            with contextlib.suppress(Exception):
                is_depth = str(yaml_cfg["head"][-1][-2]).lower() == "depth"
    if not is_depth:
        pt = getattr(model, "pt_path", None)
        if pt is not None:
            is_depth = "-depth" in Path(pt).stem
    if not is_depth:
        return
    yaml_cfg = getattr(model, "yaml", None)
    if not isinstance(yaml_cfg, dict):
        yaml_cfg = {}
    router = getattr(model, "multimodal_router", None) or getattr(model, "mm_router", None)
    if has_multimodal_config(yaml_cfg) or getattr(router, "has_multimodal_config", False):
        return
    raise ValueError(
        "YOLOMM DepthModel requires multimodal RGB/X routing in YAML. "
        "Official yolo26-depth weights/yaml are not supported."
    )


def torch_safe_load(weight, safe_only=False):
    """
    Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def _compat_patch_legacy_sppf(model: nn.Module, weight_path: str | None = None) -> int:
    """
    Patch legacy pickled models that were saved before SPPF gained runtime attributes.

    Why needed:
        We sometimes load full pickled nn.Module objects from ckpt["model"]/ckpt["ema"].
        In that case, module __init__ is NOT re-run, so newly introduced attributes may be missing.
        Newer code may access these attributes (e.g., SPPF.forward uses self.n), causing AttributeError.

    What it does:
        - If an SPPF instance lacks `n`, infer it from channel dimensions when possible, else default to 3.
        - If an SPPF instance lacks `shortcut`, default it to False.

    Returns:
        int: Number of SPPF modules patched.
    """

    def _infer_sppf_n(m: SPPF) -> int | None:
        try:
            cv1_conv = getattr(getattr(m, "cv1", None), "conv", None)
            cv2_conv = getattr(getattr(m, "cv2", None), "conv", None)
            if cv1_conv is None or cv2_conv is None:
                return None
            hidden = int(getattr(cv1_conv, "out_channels", 0) or 0)
            in_ch = int(getattr(cv2_conv, "in_channels", 0) or 0)
            if hidden <= 0 or in_ch <= 0 or in_ch % hidden:
                return None
            n = in_ch // hidden - 1
            return int(n) if n >= 1 else None
        except Exception:
            return None

    patched = 0
    for m in model.modules():
        if not isinstance(m, SPPF):
            continue

        need_patch = False
        if not hasattr(m, "n"):
            inferred = _infer_sppf_n(m)
            # YOLO11/YOLOv5-style SPPF uses 3 pooling iterations by default
            m.n = inferred if inferred is not None else 3
            need_patch = True

        if not hasattr(m, "shortcut"):
            m.shortcut = False
            need_patch = True

        if need_patch:
            patched += 1

    if patched:
        wp = weight_path or getattr(model, "pt_path", None) or "unknown"
        LOGGER.debug(f"检测到旧权重缺少 SPPF 运行时字段，已自动补齐 (patched={patched}, weight={wp})")
    return patched


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """
    Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.

    Args:
        weights (str | List[str]): Model weights path(s).
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        (torch.nn.Module): Loaded model.
    """
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = getattr(model, "task", guess_model_task(model))
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Compatibility patching for legacy pickled weights
        _compat_patch_legacy_sppf(model, weight_path=w)
        metadata = ckpt.get(MM_METADATA_KEY) if isinstance(ckpt.get(MM_METADATA_KEY), dict) else {}
        model_metadata = getattr(model, "metadata", None)
        if not isinstance(model_metadata, dict):
            model_metadata = {}
        model_metadata.update(metadata)
        model.metadata = model_metadata
        sync_x_modality_name_to_model(model, metadata.get(MM_X_MODALITY_KEY, "unknown"))
        _reject_official_depth_checkpoint(model)

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """
    Load a single model weights.

    Args:
        weight (str): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    # Compatibility patching for legacy pickled weights
    _compat_patch_legacy_sppf(model, weight_path=weight)
    metadata = ckpt.get(MM_METADATA_KEY) if isinstance(ckpt.get(MM_METADATA_KEY), dict) else {}
    model_metadata = getattr(model, "metadata", None)
    if not isinstance(model_metadata, dict):
        model_metadata = {}
    model_metadata.update(metadata)
    model.metadata = model_metadata
    sync_x_modality_name_to_model(model, metadata.get(MM_X_MODALITY_KEY, "unknown"))
    _reject_official_depth_checkpoint(model)

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def _validate_and_fill_dea_args(args, c_left):
    """Validate DEA args strictly and auto-fill only the channel when为None.

    期望签名：DEA(channel, kernel_size, p_kernel=None, m_kernel=None, reduction=16)
    允许的最简形式：DEA([None, kernel_size]) 或 DEA([channel, kernel_size])。
    不做旧参数顺序的自动更正，遇到不合规直接报错。
    """
    a = list(args) if isinstance(args, (list, tuple)) else [args]

    # 最简形式 [channel/None, kernel_size]
    if len(a) == 2:
        ch, ks = a
        ch = c_left if ch is None else ch
        if not isinstance(ch, int) or not isinstance(ks, int) or ks <= 0:
            raise ValueError(
                f"DEA expects [channel(or None), kernel_size] as minimal form, got {args}"
            )
        return [ch, ks, None, None, 16]

    # 完整形式：补齐长度到5
    while len(a) < 5:
        a.append(None)
    ch, ks, pk, mk, rd = a[:5]
    ch = c_left if ch is None else ch
    # 严格校验类型与取值
    if not isinstance(ch, int) or ch <= 0:
        raise ValueError(f"DEA arg[0]=channel must be positive int, got {ch}")
    if not isinstance(ks, int) or ks <= 0:
        raise ValueError(f"DEA arg[1]=kernel_size must be positive int, got {ks}")
    if pk is not None and (not isinstance(pk, (list, tuple)) or len(pk) != 2):
        raise ValueError(f"DEA arg[2]=p_kernel must be 2-list/tuple or None, got {pk}")
    if mk is not None and (not isinstance(mk, (list, tuple)) or len(mk) != 2):
        raise ValueError(f"DEA arg[3]=m_kernel must be 2-list/tuple or None, got {mk}")
    if rd is None:
        rd = 16
    if not isinstance(rd, int) or rd <= 0:
        raise ValueError(f"DEA arg[4]=reduction must be positive int, got {rd}")

    return [ch, ks, pk, mk, rd]


def parse_model(d, ch, verbose=True, dataset_config=None):
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.
        dataset_config (dict, optional): Dataset configuration containing Xch and other multimodal info.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))

    # 多模态配置解析（仅处理已迁移的组件）
    multimodal_router = None
    if MULTIMODAL_AVAILABLE and 'multimodal' in d:
        try:
            # 解析多模态配置
            config_parser = MultiModalConfigParser()
            model_config = config_parser.parse_config(d)

            # 创建多模态路由器
            if model_config.get('has_multimodal_layers', False):
                multimodal_router = MultiModalRouter(model_config)
                if verbose:
                    LOGGER.info(f"{colorstr('multimodal:')} Router initialized with {len(model_config.get('input_layers', []))} input layers")
        except Exception as e:
            if verbose:
                LOGGER.warning(f"Failed to initialize multimodal router: {e}")
            multimodal_router = None
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    is_backbone = False  # 单模块多输出主干出现后，后续层索引需要整体偏移 4（与 RTDETR-main 对齐）
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            FourierConv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            # 支持 SPPF 及其变体
            *SPPF_CLASS,
            # C2PSA 系列在 C2PSA_CLASS 中统一管理
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            # C2f 变体（RTDETRMM mm-mid-c2f-*）：需作为 base_modules 才能正确注入 c1/c2（并支持 repeats -> n）
            C2f_CAMixer,
            C2f_Heat,
            C2f_FMB,
            C2f_MSMHSA_CGLU,
            C2f_MogaBlock,
            C2f_SHSA,
            C2f_SHSA_CGLU,
            C2f_HDRAB,
            C2f_RAB,
            C2f_FFCM,
            C2f_SMAFB,
            C2f_SMAFB_CGLU,
            C2f_AP,
            C2f_CSI,
            C2f_gConv,
            C2f_FCA,
            C2f_FDConv,
            C2f_FDT,
            C2f_FourierConv,
            C2f_GlobalFilter,
            C2f_LSBlock,
            C2f_Strip,
            C2f_StripCGLU,
            C2f_wConv,
            C2f_FasterFDConv,
            C2f_FasterSFSConv,
            C2f_Faster_KAN,
            C2f_FAT,
            C2f_SMPCGLU,
            C2f_DBlock,
            C2f_AdditiveBlock,
            C2f_AdditiveBlock_CGLU,
            C2f_IEL,
            C2f_DTAB,
            C2f_PFDConv,
            C2f_SFSConv,
            C2f_PSFSConv,
            C2f_EBlock,
            C2f_HFERB,
            C2f_JDPM,
            C2f_ETB,
            C2f_SFHF,
            C2f_MSM,
            C2f_ELGCA,
            C2f_ELGCA_CGLU,
            C2f_LEGM,
            C2f_LFEM,
            C2f_ESC,
            C2f_KAT,
            C2f_BiFocus,
            RepNCSPELAN4,
            RepNCSPELAND,  # ELAN followed by dictionary injection (YOLO-RD)
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            FSConv,
            MRCB,
            C2fCIB,
            A2C2f,
            ConvNormLayer,  # RTDETR module
            LoGStem,  # LoG Stem 模块 (4x 下采样)
            LoGStem2x,  # LoG Stem2x 模块 (2x 下采样，可替代 Conv)
            # 下采样模块 (Downsample)
            LAWDS,
            EdgeLAWDS,
            FreqLAWDS,
            HWD,
            RouterLAWDS,
            V7DownSampling,
            # 即插即用注意力模块 (Public)
            CoordAtt,
            DeformableLKA,
            EMA_Attention,
            LSKBlock,
            LSKBlock_SA,
            # 新一批 channel-changing 公共模块（c1→c2, base_modules 分支）
            AdaptiveRectangularConv,
            ConvUtr,
            FDSM,
            Poly_Kernel_Inception_Block,
            # 新一批融合模块（c1→c2, base_modules 分支）
            CCFF,
            Modulation_Fusion_Module,
        } | set(C3K2_CLASS) | set(C2PSA_CLASS) | set(NECK_CLASS) | set(ATTENTION_CLASS) | set(MAMBA_CLASS)  # 动态添加已迁移模块
    )
    # 防御性检查：确保 Block 类未意外进入 base_modules
    assert not (block_repeat_modules & base_modules), (
        f"Block 类不应同时出现在 block_repeat_modules 和 base_modules 中: "
        f"{block_repeat_modules & base_modules}"
    )
    # 上采样模块是通道保持（c2=ch[f]，YAML args 可为空），必须走专用分支，不能进 base_modules
    assert not (set(UPSAMPLE_CLASS) & base_modules), (
        f"上采样模块不应出现在 base_modules 中（空 args 会触发 args[0] IndexError）: "
        f"{set(UPSAMPLE_CLASS) & base_modules}"
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C2f_CAMixer,
            C2f_Heat,
            C2f_FMB,
            C2f_MSMHSA_CGLU,
            C2f_MogaBlock,
            C2f_SHSA,
            C2f_SHSA_CGLU,
            C2f_HDRAB,
            C2f_RAB,
            C2f_FFCM,
            C2f_SMAFB,
            C2f_SMAFB_CGLU,
            C2f_AP,
            C2f_CSI,
            C2f_gConv,
            C2f_FCA,
            C2f_FDConv,
            C2f_FDT,
            C2f_FourierConv,
            C2f_GlobalFilter,
            C2f_LSBlock,
            C2f_Strip,
            C2f_StripCGLU,
            C2f_wConv,
            C2f_FasterFDConv,
            C2f_FasterSFSConv,
            C2f_Faster_KAN,
            C2f_FAT,
            C2f_SMPCGLU,
            C2f_DBlock,
            C2f_AdditiveBlock,
            C2f_AdditiveBlock_CGLU,
            C2f_IEL,
            C2f_DTAB,
            C2f_PFDConv,
            C2f_SFSConv,
            C2f_PSFSConv,
            C2f_EBlock,
            C2f_HFERB,
            C2f_JDPM,
            C2f_ETB,
            C2f_SFHF,
            C2f_MSM,
            C2f_ELGCA,
            C2f_ELGCA_CGLU,
            C2f_LEGM,
            C2f_LFEM,
            C2f_ESC,
            C2f_KAT,
            C2f_BiFocus,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fCIB,
            A2C2f,
        } | set(C3K2_CLASS) | set(C2PSA_CLASS)  # 动态添加所有 C3k2 与 C2PSA 变体
    )
    # ===== MULTIMODAL EXTENSION START - 多模态路由器初始化 =====
    mm_router = None
    if MULTIMODAL_AVAILABLE:
        try:
            from ultralytics.nn.mm import MultiModalRouter
            # 构建配置字典，包含dataset_config
            config_dict = d.copy()
            if dataset_config:
                config_dict['dataset_config'] = dataset_config
            mm_router = MultiModalRouter(config_dict, verbose)
        except Exception as e:
            if verbose:
                LOGGER.warning(f"MultiModal router initialization failed: {e}")
    # ===== MULTIMODAL EXTENSION END =====

    # 通用解析子程序：两输入的跨模态注意力/融合模块（CTF/MHCA）
    def _parse_two_input_equal_attn(module, f, args, ch, i):
        if isinstance(f, int) or len(f) != 2:
            raise ValueError(f"{module.__name__} expects 2 inputs, got {f} at layer {i}")
        c_left, c_right = ch[f[0]], ch[f[1]]
        # Auto-fill first dim arg if missing/None
        if len(args) == 0:
            args.insert(0, c_left)
        else:
            if args[0] is None:
                args[0] = c_left
        # Module-specific defaults and output channel calc
        if module is MultiHeadCrossAttention:
            # Ensure num_heads exists
            if len(args) < 2:
                args.append(2)
            c2_val = c_left  # tuple outputs with per-branch channels = C
        elif module is CrossTransformerFusion:
            c2_val = c_left * 2  # concat two branches
        else:
            c2_val = c_left
        return c2_val, args

    # 针对 C3k2 变体的参数归一化，补齐缺失的关键形参以匹配上游实现
    def _normalize_c3k2_args(module, raw_args):
        # 保留原列表，避免外部引用被直接修改
        args = list(raw_args)
        c2_val = args[0] if args else None

        # 如果 C3k2 变体未成功导入，则抛出明确错误，避免静默降级
        if not C3K2_EXTRACTION_AVAILABLE:
            raise RuntimeError(
                "C3k2 extraction modules导入失败，无法解析C3k2层参数。原始异常: "
                f"{_C3K2_IMPORT_ERROR}"
            )

        # fmapsize 相关
        if module in (C3k2_DAttention, C3k2_Parc, C3k2_FocusedLinearAttention):
            if len(args) < 2 or isinstance(args[1], bool):
                args = [c2_val, (20, 20)] + args[2:]
            # 确保第三参为 shortcut
            if len(args) < 3 or not isinstance(args[2], bool):
                args = args[:2] + [True] + args[3:]

        # 聚合注意力：input_resolution & sr_ratio
        elif module is C3k2_AggregatedAtt:
            need_fill = len(args) < 3 or isinstance(args[1], bool) or isinstance(args[2], bool)
            if need_fill:
                # 按通道规模推导默认输入分辨率与步幅比
                input_res = 40 if c2_val is not None and c2_val <= 512 else 20
                sr_ratio = 2 if c2_val is not None and c2_val <= 512 else 1
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [c2_val, input_res, sr_ratio, shortcut]

        # SWC 位移卷积：kernel_size
        elif module is C3k2_SWC:
            if len(args) < 2 or isinstance(args[1], bool):
                ks = 11 if c2_val is not None and c2_val <= 256 else (9 if c2_val is not None and c2_val <= 512 else 7)
                shortcut = True
                # 若第二/第三参数本就是布尔，视作 shortcut
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                elif len(args) > 2 and isinstance(args[2], bool):
                    shortcut = args[2]
                args = [c2_val, ks, shortcut]

        # UniRep 大核：k
        elif module is C3k2_UniRepLKNetBlock:
            if len(args) < 2 or isinstance(args[1], bool):
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [c2_val, 7, shortcut]

        # iRMB 派生：深度卷积核
        elif module in (C3k2_iRMB_DRB, C3k2_iRMB_SWC):
            if len(args) < 2 or isinstance(args[1], bool):
                ks = 13 if c2_val is not None and c2_val <= 256 else (11 if c2_val is not None and c2_val <= 512 else 9)
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                elif len(args) > 2 and isinstance(args[2], bool):
                    shortcut = args[2]
                args = [c2_val, ks, shortcut]

        return args

    for i, layer_config in enumerate(d["backbone"] + d["head"]):  # from, number, module, args, [input_type]
        # ===== MULTIMODAL EXTENSION START - 层配置解析 =====
        # Parse layer configuration with optional 5th field for multimodal routing
        if mm_router:
            c1, mm_input_source, mm_attributes = mm_router.parse_layer_config(layer_config, i, ch, verbose)
            f, n, m, args = layer_config[:4]  # Extract standard 4 fields
        else:
            # Standard parsing for non-multimodal layers
            if len(layer_config) == 4:
                f, n, m, args = layer_config
            elif len(layer_config) == 5:
                f, n, m, args, _ = layer_config  # Ignore 5th field if no MM router
            else:
                raise ValueError(f"Invalid layer definition at index {i}: expected 4 or 5 elements, got {len(layer_config)}")
            c1 = None
            mm_input_source = None
            mm_attributes = {}
        # ===== MULTIMODAL EXTENSION END =====
        
        try:
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else globals()[m]
            )  # get module
        except KeyError as e:
            # 如果YAML中使用的模块找不到，抛出错误并终止程序
            raise ImportError(
                f"模块 '{m}' 在YAML配置的第 {i} 层中被使用，但未找到该模块。"
                f"请确保模块已正确导入。"
                f"如果这是来自extra_modules的模块，请检查它是否在__init__.py中正确导出。"
            ) from e
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        m_created = False
        m_instance = None
        layer_is_backbone = False
        # =====================================================================
        # 万物皆可融 Block 分支 — 支持通过 YAML dict 动态指定子模块
        # 必须在 base_modules 分支之前，否则 Block 类会走通用分支
        # =====================================================================
        if m in block_repeat_modules:
            # 多模态兼容：检查是否需要使用 router 计算的 c1
            c1_m = c1 if (mm_input_source is not None and c1 is not None) else ch[f]
            c2 = args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            n = n_
            # 解析 args 中的 dict 类型参数（YAML 中指定子模块）
            for i_a, a in enumerate(args):
                if isinstance(a, dict):
                    # 解析 module 名称为类对象
                    mod_name = a.get('module', 'Bottleneck')
                    if mod_name.startswith('nn.'):
                        mod_cls = getattr(torch.nn, mod_name[3:])
                    else:
                        mod_cls = globals().get(mod_name)
                        if mod_cls is None:
                            raise ValueError(
                                f"万物皆可融 Block: 找不到子模块 '{mod_name}'。"
                                f"请确保该模块已在 tasks.py 中导入。"
                            )
                    # 解析子模块参数
                    mod_params = a.get('param', {})
                    if mod_params:
                        from functools import partial as _partial
                        resolved_params = {}
                        for k, v in mod_params.items():
                            if isinstance(v, str):
                                with contextlib.suppress(ValueError):
                                    v = ast.literal_eval(v)
                            resolved_params[k] = v
                        mod_cls = _partial(mod_cls, **resolved_params)
                    # 解析 selfatt 参数
                    selfatt = a.get('selfatt', False)
                    # 将 dict 替换为 (module, selfatt) tuple
                    args[i_a] = {'module': mod_cls, 'selfatt': selfatt}
            args = [c1_m, c2, n, *args[1:]]
            # 处理 args 中的 dict 参数：展开为关键字参数
            kwargs = {}
            positional_args = []
            for a in args:
                if isinstance(a, dict):
                    kwargs.update(a)
                else:
                    positional_args.append(a)
            m_ = nn.Sequential(*(m(*positional_args, **kwargs) for _ in range(n))) if n > 1 else m(*positional_args, **kwargs)
        elif (ATTENTION_AVAILABLE and m in set(ATTENTION_CLASS)) \
                or (PUBLIC_ATTN_AVAILABLE and m in set(_PUBLIC_PRESERVE_CLASS)) \
                or (MAMBA_AVAILABLE and m in set(MAMBA_CLASS)):
            # 注意力 / Mamba / public channel-preserving 模块: c2=c1, args=[dim, *extra_params]
            c2 = ch[f]
            args = [c2, *args]
        elif m is ChannelTransformer:
            if isinstance(f, int) or len(f) not in {3, 4}:
                raise ValueError(f"{m.__name__} expects 3 or 4 inputs, got {f} at layer {i}")

            input_channels = [ch[x] for x in f]
            default_patch_sizes = [40, 20, 10, 5][:len(input_channels)]
            args = list(args)

            if len(args) == 0:
                args = [input_channels, 640, False, default_patch_sizes]
            else:
                args[0] = input_channels
                if len(args) < 2:
                    args.append(640)
                if len(args) < 3:
                    args.append(False)
                if len(args) < 4:
                    args.append(default_patch_sizes)
                elif args[3] is None:
                    args[3] = default_patch_sizes

            c2 = input_channels
            m_instance = m(*args)
            m_created = True
        elif m in base_modules:
            # 针对特定 C3k2 变体补齐缺省参数（需在装配 c1/c2 之前执行）
            args = _normalize_c3k2_args(m, args)
            # 确保 args 可变（支持后续 args[0] 写回）
            args = list(args)
            skip_c2_scale = False

            # ===== MULTIMODAL EXTENSION START - 多模态通道计算 =====
            # Use multimodal router computed c1 if available, otherwise use standard logic
            if mm_input_source and c1 is not None:
                # For multimodal layers, c1 is computed by router, c2 from args
                c1_actual = c1
                c2 = args[0]
                # c1 is already set by mm_router.parse_layer_config()
            else:
                c1, c2 = ch[f], args[0]
                c1_actual = c1
            # ===== MULTIMODAL EXTENSION END =====

            # HWD (Haar 小波下采样): 输出通道等于输入通道，不进行 width 缩放
            if DOWNSAMPLE_AVAILABLE and m is HWD:
                c2 = c1_actual
                args = [c2]
                skip_c2_scale = True

            # 统一缩放 c2 并写回 args[0]（确保后续 [c1, *args] 插入时使用缩放后的值）
            if not skip_c2_scale and c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args[0] = c2  # 写回缩放后的 c2，防止后续使用未缩放值

            # C2fAttn 特殊处理：在插入 c1 之前调整 embed channels 和 num heads
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            # SPPF 变体参数长度校验（硬错误，防止参数错位）
            if m is SPPF:
                if not (1 <= len(args) <= 4):
                    raise ValueError(
                        f"Layer {i} (SPPF): 参数长度非法，期望 args=[c2] 或 [c2,k] 或 [c2,k,n] 或 [c2,k,n,shortcut]，"
                        f"但收到 args={args}"
                    )
            elif SPPF_EXTRACTION_AVAILABLE and m is SPPF_LSKA:
                if not (1 <= len(args) <= 2):
                    raise ValueError(
                        f"Layer {i} (SPPF_LSKA): 参数长度非法，期望 args=[c2] 或 [c2,k]，"
                        f"但收到 args={args}（SPPF_LSKA 不支持 n/shortcut 参数）"
                    )

            # 统一插入 c1（此时 args[0] 已是缩放后的 c2）
            args = [c1, *args]

            # repeat_modules 处理
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1

            # C3K2_CLASS 特殊处理（M/L/X 尺寸）
            if m in C3K2_CLASS:  # for M/L/X sizes - 支持所有C3k2变体
                legacy = False
                if scale in "mlx":
                    args[3] = True

            # A2C2f 特殊处理（L/X 尺寸）
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))

            # C2fCIB 特殊处理
            if m is C2fCIB:
                legacy = False
        elif m in {
            AIFI,
            AIFI_LPE,
            AIFI_RepBN,
            AIFI_SEFN,
            AIFI_Mona,
            AIFI_DyT,
            AIFI_SEFFN,
            AIFI_EDFFN,
            TransformerEncoderLayer_LocalWindowAttention,
            TransformerEncoderLayer_DAttention,
            TransformerEncoderLayer_HiLo,
            TransformerEncoderLayer_EfficientAdditiveAttnetion,
            TransformerEncoderLayer_AdditiveTokenMixer,
            TransformerEncoderLayer_MSMHSA,
            TransformerEncoderLayer_DHSA,
            TransformerEncoderLayer_DPB,
            TransformerEncoderLayer_Pola,
            TransformerEncoderLayer_TSSA,
            TransformerEncoderLayer_ASSA,
            TransformerEncoderLayer_MSLA,
            TransformerEncoderLayer_Pola_SEFN,
            TransformerEncoderLayer_ASSA_SEFN,
            TransformerEncoderLayer_ASSA_SEFN_Mona,
            TransformerEncoderLayer_Pola_SEFN_Mona,
            TransformerEncoderLayer_ASSA_SEFN_Mona_DyT,
            TransformerEncoderLayer_Pola_SEFN_Mona_DyT,
            TransformerEncoderLayer_Pola_SEFFN_Mona_DyT,
            TransformerEncoderLayer_Pola_EDFFN_Mona_DyT,
        }:
            args = [ch[f], *args]
        elif m in {
            TimmBackbone,
            convnextv2_atto,
            convnextv2_femto,
            convnextv2_pico,
            convnextv2_nano,
            convnextv2_tiny,
            convnextv2_base,
            convnextv2_large,
            convnextv2_huge,
            repvit_m0_9,
            repvit_m1_0,
            repvit_m1_1,
            repvit_m1_5,
            repvit_m2_3,
            efficientformerv2_s0,
            efficientformerv2_s1,
            efficientformerv2_s2,
            efficientformerv2_l,
            EfficientViT_M0,
            EfficientViT_M1,
            EfficientViT_M2,
            EfficientViT_M3,
            EfficientViT_M4,
            EfficientViT_M5,
            SwinTransformer_Tiny,
        }:
            # 约束：单模块多输出主干必须出现在 backbone 首层，否则索引体系会产生歧义
            if i != 0:
                raise RuntimeError(
                    f"{m.__name__ if hasattr(m, '__name__') else m} 必须作为模型 backbone 的第 0 层（单模块多输出主干）。"
                    "请将模态输入投影Conv放在该模块内部（已内置），不要在其前面再插入额外层。"
                )
            if n != 1:
                raise ValueError(
                    f"{m.__name__ if hasattr(m, '__name__') else m} 不支持 repeats>1（单模块多输出主干）"
                )
            # 注入输入通道（优先使用 router 解析出的 c1，以支持 Dual=6ch 等多模态输入）
            in_ch = c1 if c1 is not None else ch[f]
            args = [in_ch, *args]
            m_instance = m(*args)
            m_created = True
            c2 = list(getattr(m_instance, "channel"))
            layer_is_backbone = True
        elif m in frozenset({HGStem, HGBlock}):
            c1 = c1 if (mm_input_source is not None and c1 is not None) else ch[f]
            cm, c2 = args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif DUAL_HEAD_AVAILABLE and m is MisalignGatedConcat:
            # 结构等价于 Concat：输出通道为两路之和，仅对 IR 半边乘逐样本标量 alpha
            c2 = sum(ch[x] for x in f)
            args = [[ch[x] for x in f], float(args[0]) if args else 0.2]
        # ===== GOLD-YOLO / 提取类池化-融合模块 =====
        elif SPPF_EXTRACTION_AVAILABLE and m in frozenset({SimFusion_4in, AdvPoolFusion}):
            # 纯拼接型融合，输出通道为所有输入之和
            c2 = sum(ch[x] for x in f)
        elif SPPF_EXTRACTION_AVAILABLE and m is SimFusion_3in:
            # SimFusion_3in([c0,c1,c2], out)
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[f_] for f_ in f], c2]
        elif SPPF_EXTRACTION_AVAILABLE and m is IFM:
            # IFM(c1, [o1,o2,...], embed_dim_p=96, fuse_block_num=3)
            c1 = ch[f]
            c2 = sum(args[0])
            args = [c1, *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is InjectionMultiSum_Auto_pool:
            # InjectionMultiSum_Auto_pool(inp, oup, global_inp, flag), f=[local, global]
            c1 = ch[f[0]]
            c2 = args[0]
            args = [c1, *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is PyramidPoolAgg:
            # PyramidPoolAgg(inc=sum(inputs), ouc=args[0], stride, pool_mode='torch')
            c2 = args[0]
            args = [sum(ch[x] for x in f), *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is PyramidPoolAgg_PCE:
            # 输出为各输入池化后拼接，通道为输入之和
            c2 = sum(ch[x] for x in f)
        elif SPPF_EXTRACTION_AVAILABLE and m is WaveletPool:
            # 小波下采样，通道*4
            c2 = ch[f] * 4
        # ===== 上采样模块分支 =====
        # 上采样模块: c2=ch[f]（通道数不变），args=[c2, *args]
        elif UPSAMPLE_AVAILABLE and m in set(UPSAMPLE_CLASS):
            # 多模态兼容：检查是否需要使用 router 计算的 c1
            if mm_input_source and c1 is not None:
                c2 = c1
            else:
                c2 = ch[f]
            args = [c2, *args]
        # DEA 专用分支：允许旧风格参数并归一化到新签名
        elif m is DEA:
            # Expect exactly two inputs; output channels follow the left branch
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            args = _validate_and_fill_dea_args(args, c_left)
            c2 = c_left
        elif m is MJRNet:
            # MJRNet: 双输入融合模块，输出通道等于输入通道
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            if c_left != c_right:
                raise ValueError(f"{m.__name__} expects equal input channels, got {c_left} vs {c_right} at layer {i}")
            # args[0] 是 in_channels（用实际值替换），args[1] 是 reduction
            reduction = args[1] if len(args) > 1 else 16
            args = [c_left, reduction]
            c2 = c_left
        elif m is RFEM:
            # RFEM: 单输入感受野扩展模块，输出通道等于输入通道
            c1 = ch[f]
            # args 可能包含 num_scales, kernel_sizes, dilations，但 in_channels 必须从上层获取
            args = [c1] + list(args[1:]) if len(args) > 1 else [c1]
            c2 = c1
        elif m is DeformableInteractiveAttention:
            # DeformableInteractiveAttention: 无通道参数（stride=1, distortionmode=False）
            # c2 等于输入通道，args 不注入通道数
            c2 = ch[f]
            args = list(args)
        elif m is Shift_Module:
            # Shift_Module: 无通道参数（shift_size=1）
            # c2 等于输入通道，args 不注入通道数
            c2 = ch[f]
            args = list(args)
        elif m is InverseConv2DLayer:
            # InverseConv2DLayer: 强制 out_channels == in_channels（逐通道卷积）
            # YAML args: [kernel_size, scale?, padding?, ...] — 不包含通道数
            c1 = ch[f]
            c2 = c1
            args = [c1, c1, *args]
        elif m is DRFD:
            # DRFD: 下采样残差融合模块
            # YAML: [in_channels, out_channels]，但 parse_model 需要 [c1, c2] 格式
            c1 = ch[f]
            c2 = args[0] if args else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2]
        elif m is Cut:
            # Cut: 空间到深度下采样模块
            # YAML: [in_channels, out_channels]，但 parse_model 需要 [c1, c2] 格式
            c1 = ch[f]
            c2 = args[0] if args else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2]
        elif m is MSIA:
            # MSIA: 双输入迭代聚合模块，输出通道等于输入通道
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            if c_left != c_right:
                raise ValueError(f"{m.__name__} expects equal input channels, got {c_left} vs {c_right} at layer {i}")
            # args[0] 是 in_channels（用实际值替换），args[1] 是 reduction
            reduction = args[1] if len(args) > 1 else 16
            args = [c_left, reduction]
            c2 = c_left
        elif m is RFF:
            # RFF: 残差特征融合模块 (低级特征 + 高级特征)
            # 输入: [低级特征索引, 高级特征索引]，输出通道等于低级特征通道
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs (low_feat, high_feat), got {f} at layer {i}")
            c_low, c_high = ch[f[0]], ch[f[1]]
            # args: [low_channels, high_channels, groups]
            groups = args[0] if len(args) > 0 else 4
            args = [c_low, c_high, groups]
            c2 = c_low  # 输出对齐到低级特征通道
        elif m is MCFGatedFusion:
            if isinstance(f, int) or len(f) < 2:
                raise ValueError(f"{m.__name__} expects >=2 inputs, got {f} at layer {i}")
            mode = args[0] if len(args) > 0 else "add"
            k = args[1] if len(args) > 1 else 1
            c_out = args[2] if len(args) > 2 and args[2] else None
            main_idx = args[3] if len(args) > 3 else 0
            aux_idx = args[4] if len(args) > 4 else 1
            zero_init = args[5] if len(args) > 5 else True
            use_bn = args[6] if len(args) > 6 else False
            act = args[7] if len(args) > 7 else True

            c_main = ch[f[main_idx]]
            c_aux = ch[f[aux_idx]]
            c_out = c_out or c_main
            if c_out != nc:
                c_out = make_divisible(min(c_out, max_channels) * width, 8)
            args = [c_main, c_aux, c_out, mode, k, 1, None, 1, main_idx, aux_idx, zero_init, use_bn, act]
            c2 = c_main
        elif ReliabilityAwareFusion is not None and m is ReliabilityAwareFusion:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            ouc = args[0] if args and args[0] is not None else c_left
            if ouc != nc:
                ouc = make_divisible(min(ouc, max_channels) * width, 8)
            max_offset = args[1] if len(args) > 1 else 0.08
            args = [[c_left, c_right], ouc, max_offset]
            c2 = ouc
        elif GatedReliabilityFusion is not None and m is GatedReliabilityFusion:
            # 显式门控融合（改动 A）：签名同 ReliabilityAwareFusion，额外支持 base / gate 两个可选参数
            #   YAML 形如: [[rgb_idx, ir_idx], 1, GatedReliabilityFusion, [ouc, max_offset, base, gate]]
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            ouc = args[0] if args and args[0] is not None else c_left
            if ouc != nc:
                ouc = make_divisible(min(ouc, max_channels) * width, 8)
            max_offset = args[1] if len(args) > 1 and args[1] is not None else 0.08
            g_base = args[2] if len(args) > 2 and args[2] is not None else "rgb"
            g_gate = args[3] if len(args) > 3 and args[3] is not None else "channel"
            args = [[c_left, c_right], ouc, max_offset, g_base, g_gate]
            c2 = ouc
        elif DeformableCrossModalFusion is not None and m is DeformableCrossModalFusion:
            # 改动 C4：K 点可变形交叉注意力融合。前三个参数与 GatedReliabilityFusion 对齐，
            # 保证诊断脚本零改动；额外支持 num_points(K) / attn。
            #   YAML 形如: [[rgb_idx, ir_idx], 1, DeformableCrossModalFusion, [ouc, max_offset, num_points, base, attn]]
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            ouc = args[0] if args and args[0] is not None else c_left
            if ouc != nc:
                ouc = make_divisible(min(ouc, max_channels) * width, 8)
            max_offset = args[1] if len(args) > 1 and args[1] is not None else 0.08
            num_points = args[2] if len(args) > 2 and args[2] is not None else 4
            d_base = args[3] if len(args) > 3 and args[3] is not None else "rgb"
            d_attn = args[4] if len(args) > 4 and args[4] is not None else "channel"
            args = [[c_left, c_right], ouc, max_offset, num_points, d_base, d_attn]
            c2 = ouc
        elif ChannelPartitionFusion is not None and m is ChannelPartitionFusion:
            # 改动 D3：通道级硬性模态分工融合。前四个参数与 GatedReliabilityFusion 对齐
            # （inc, ouc, max_offset, base），额外支持 split（三段通道占比）与 num_points/attn。
            #   YAML 形如: [[rgb_idx, ir_idx], 1, ChannelPartitionFusion, [ouc, max_offset, base, split, num_points, attn]]
            #   split 为三元组，形如 [0.5, 0.25, 0.25]（RGB 段 / IR 段 / 交互段）。
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            ouc = args[0] if args and args[0] is not None else c_left
            if ouc != nc:
                ouc = make_divisible(min(ouc, max_channels) * width, 8)
            p_max_offset = args[1] if len(args) > 1 and args[1] is not None else 0.08
            p_base = args[2] if len(args) > 2 and args[2] is not None else "rgb"
            p_split = args[3] if len(args) > 3 and args[3] is not None else (0.5, 0.25, 0.25)
            p_num_points = args[4] if len(args) > 4 and args[4] is not None else 4
            p_attn = args[5] if len(args) > 5 and args[5] is not None else "channel"
            args = [[c_left, c_right], ouc, p_max_offset, p_base, p_split, p_num_points, p_attn]
            c2 = ouc
        # 跨模态双输入融合模块：CIDAF / CGAFusion / DAF / WDAF
        # 签名模式: Module(inc=[c_left, c_right], ouc, ...) 或 Module(in_dim=[c_left, c_right], out_dim, ...)
        elif m in frozenset({CIDAF, CGAFusion, DAF, DynamicAlignFusion, WDAF}):
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            # 自动推断输出通道：默认取 ouc = args[0]，若未指定则取 c_left
            ouc = args[0] if len(args) > 0 and args[0] is not None else c_left
            if ouc != nc:
                ouc = make_divisible(min(ouc, max_channels) * width, 8)
            # CGAFusion 额外支持 reduction 参数
            if m is CGAFusion:
                reduction = args[1] if len(args) > 1 else 8
                args = [[c_left, c_right], ouc, reduction]
            else:
                args = [[c_left, c_right], ouc]
            c2 = ouc
        # 其他融合模块（与多模态路由无关的纯特征融合），要求两路输入空间尺寸一致
        elif m in frozenset({FeatureFusion, FCM, FCMFeatureFusion, ConvMixFusion, ScalarGate, ChannelGate, CAM, SEFN, FusionConvMSAA, MSC, SpatialDependencyPerception, FDFEF, CMCE, FACMA, WCMF, CVIM, BIE, MAFM, FEFM, PagFM, GFM, LFGA, FFM, PIM, DPCF}):
            # Expect exactly two inputs; output channels follow the left branch
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            # Auto infer dim if not provided (None or missing)
            if len(args) == 0:
                args.insert(0, c_left)
            elif args[0] is None:
                args[0] = c_left
            c2 = c_left
        elif m is PST:
            # Pyramid Sparse Transformer expects (x, upper_feat) with upper_feat at 1/2 resolution.
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs (x, upper_feat), got {f} at layer {i}")
            c_left, c_up = ch[f[0]], ch[f[1]]
            if len(args) < 1:
                raise ValueError(f"{m.__name__} requires args [c2, mlp_ratio?, e?, k?], got {args} at layer {i}")
            c_out = args[0]
            mlp_ratio = args[1] if len(args) > 1 else 2.0
            e = args[2] if len(args) > 2 else 0.5
            k = args[3] if len(args) > 3 else 0
            # signature: PST(c1, c_up, c2, n=1, mlp_ratio=..., e=..., k=...)
            args = [c_left, c_up, c_out, n, mlp_ratio, e, k]
            c2 = c_out
            n = 1
        
        # IR侧轻量金字塔（简化版），输出多尺度tuple (P3/P4/P5)。放在此处可避免进入 NNexpend 区域。
        elif m is SpatialPriorModuleLite:
            # 期望单路输入（IR图像或其特征），返回3个尺度特征。
            if isinstance(f, (list, tuple)) and len(f) != 1:
                raise ValueError(f"{m.__name__} expects 1 input, got {f} at layer {i}")
            # 参数签名（强制显式传参，不做兼容性自动填充）：
            # SpatialPriorModuleLite(inplanes: int, embed_dims: (C8,C16,C32), in_chans: int, use_bn: bool=False/True)
            if len(args) < 3:
                raise ValueError(
                    f"{m.__name__} 需要显式参数: [inplanes, (C8,C16,C32), in_chans][, use_bn]，"
                    f"但第 {i} 层收到: {args}"
                )
            # 校验 embed_dims
            embed = args[1]
            if not isinstance(embed, (list, tuple)) or len(embed) != 3:
                raise ValueError(
                    f"{m.__name__} 的第2个参数应为3元组 (C8,C16,C32)，但收到: {embed}"
                )
            # 记录第一个输出通道到通道追踪器（后续可用 Index 选择具体尺度）
            try:
                c2 = int(embed[0])
            except Exception as e:
                raise ValueError(f"{m.__name__} 的 embed_dims[0] 需为整数，但收到: {embed[0]}") from e
        elif m in frozenset({CrossTransformerFusion, MultiHeadCrossAttention}):
            c2, args = _parse_two_input_equal_attn(m, f, args, ch, i)
        elif m is ConvFFN_GLU:
            # Standalone conv-ffn with GLU gate. By default assumes input is 2C and output is C.
            # If args not provided, set [in_channels=c_in, out_channels=c_in // 2]
            c_in = ch[f]
            if len(args) < 2:
                # [in_channels, out_channels]
                args = [c_in, max(c_in // 2, 1), *args]
            else:
                # fill None placeholders
                if args[0] is None:
                    args[0] = c_in
                if args[1] is None:
                    args[1] = max(c_in // 2, 1)
            c2 = args[1]
        elif m in SEMSEG_CLASS:
            if not isinstance(f, (list, tuple)) or len(f) < 1:
                raise ValueError(f"{m.__name__} expects at least 1 input [P3] or 2 inputs [P3,P4], got from={f}")
            args.append([ch[x] for x in f])
            c2 = args[0]  # SemSegHead outputs raw logits with nc channels.
        elif m in DEPTH_CLASS:
            if not isinstance(f, (list, tuple)) or len(f) < 3:
                raise ValueError(f"{m.__name__} expects 3 inputs [P3,P4,P5], got from={f}")
            args.append([ch[x] for x in f])  # Depth(c_mid, [cP3,cP4,cP5])
            c2 = 1
        elif m in (DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS):
            # 为检测/分割/姿态/旋转头注入各层输入通道列表
            args.append([ch[x] for x in f])
            # nc 覆盖：YAML 中的字面值（如 nc: 80）可能与外部传入的 nc 不一致
            # （例如 DetectionModel(yaml, nc=1) 会覆盖 self.yaml["nc"]，但不会改写层 args）。
            # 检测头的第 0 个参数就是 nc，必须与最终 nc 对齐，否则头会按 80 类构建。
            if len(args) > 0 and isinstance(args[0], (int, float)) and not isinstance(args[0], bool):
                if int(args[0]) != int(nc):
                    args[0] = nc
            # 多头架构（双头+融合分支）下，检测头可能出现在模型中间而非末尾。
            # 检测头输出的通道是 no = nc + reg_max*4（如 65），这与 neck 特征图通道
            # 语义不同，不能计入 ch[] 供下游 layer 复用，否则后续任何以该索引为输入的
            # 层都会拿到 65 这个错误通道数。此处沿用该头「首个输入特征图」的通道数作为
            # 占位，保证 ch[] 对下游依旧表示「可继续消费的特征图通道」。
            c2 = ch[f[0]] if isinstance(f, (list, tuple)) and len(f) > 0 else ch[f]
            # 分割头的通用通道缩放（与上游保持一致）
            if m in SEGMENT_CLASS:
                # 某些分割实现会使用第三个位置作为通道相关超参（如原生 Segment 的 npr/c4），保持与上游一致的缩放
                # 注意：LSCD 分割实际使用 args[3] 作为 hidc，后续专门处理
                if len(args) > 2 and isinstance(args[2], (int, float)):
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            # 统一 legacy 标记以维持兼容（若目标类未定义该属性不会产生副作用）
            for cls_ in (Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB):
                if m is cls_:
                    m.legacy = legacy
                    break

            # ===== LSCD 系列专项缩放：按上游做法对 hidc 等通道型参数进行宽度缩放 =====
            if LSCD_AVAILABLE:
                if m is Detect_LSCD and len(args) > 1 and isinstance(args[1], (int, float)):
                    # Detect_LSCD(nc, hidc, ch)
                    args[1] = make_divisible(min(args[1], max_channels) * width, 8)
                elif m is Segment_LSCD and len(args) > 3 and isinstance(args[3], (int, float)):
                    # Segment_LSCD(nc, nm, npr, hidc, ch) -> 缩放 hidc
                    args[3] = make_divisible(min(args[3], max_channels) * width, 8)
                elif m is Pose_LSCD and len(args) > 2 and isinstance(args[2], (int, float)):
                    # Pose_LSCD(nc, kpt_shape, hidc, ch) -> 缩放 hidc
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                elif m is OBB_LSCD and len(args) > 2 and isinstance(args[2], (int, float)):
                    # OBB_LSCD(nc, ne, hidc, ch) -> 缩放 hidc
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)

            # ===== LSPCD 系列专项参数注入：reg_max/end2end + 分割头 nm 缩放 =====
            # LSPCD 签名: (nc, reg_max=16, end2end=False, ch) 或带额外参数
            # YAML 中 args 通常只含 nc 等基本参数，通用头分支已追加 ch 列表，
            # 此处需在 ch 之前插入 reg_max 和 end2end。
            if LSPCD_AVAILABLE and m in (
                Detect_LSPCD, Segment_LSPCD, Segment26_LSPCD,
                OBB_LSPCD, OBB26_LSPCD, Pose_LSPCD, Pose26_LSPCD,
            ):
                # args 末尾已被通用分支追加了 [ch[x] for x in f]，取出并重新组装
                ch_list = args.pop()  # 移除末尾的 ch 列表
                # 追加 reg_max, end2end, ch
                args.extend([d.get('reg_max', 16), d.get('end2end', False), ch_list])
                # 分割头的 nm 通道缩放
                if m in (Segment_LSPCD, Segment26_LSPCD):
                    # Segment_LSPCD(nc, nm=32, npr=256, reg_max, end2end, ch)
                    if len(args) > 1 and isinstance(args[1], (int, float)):
                        args[1] = make_divisible(min(args[1], max_channels) * width, 8)

            # ===== YOLO11 检测头变体专项缩放/解析（无需编译）=====
            # 1) AFPN 系列：Detect_AFPN_*(nc, hidc, [block_type], ch)
            if m in (Detect_AFPN_P345, Detect_AFPN_P2345) and len(args) > 1 and isinstance(args[1], (int, float)):
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            elif m in (Detect_AFPN_P345_Custom, Detect_AFPN_P2345_Custom):
                # args: [nc, hidc, block_type(str|cls), ch]
                if len(args) > 1 and isinstance(args[1], (int, float)):
                    args[1] = make_divisible(min(args[1], max_channels) * width, 8)
                if len(args) > 2 and isinstance(args[2], str):
                    if args[2] not in globals():
                        raise ValueError(f"{m.__name__} 的 block_type='{args[2]}' 未注册到 tasks.py globals()，请先导入该类")
                    args[2] = globals()[args[2]]

            # 2) 共享卷积头家族：Detect_LSCSBD/LSDECD/RSCD/LSCD_LQE -> 缩放 hidc
            if m in (Detect_LSCSBD, Detect_LSDECD, Detect_RSCD, Detect_LSCD_LQE) and len(args) > 1 and isinstance(args[1], (int, float)):
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            elif m in (Segment_LSCSBD, Segment_LSDECD, Segment_RSCD, Segment_LSCD_LQE) and len(args) > 3 and isinstance(args[3], (int, float)):
                # Segment_*(nc, nm, npr, hidc, ch)
                args[3] = make_divisible(min(args[3], max_channels) * width, 8)
            elif m in (Pose_LSCSBD, Pose_LSDECD, Pose_RSCD, Pose_LSCD_LQE) and len(args) > 2 and isinstance(args[2], (int, float)):
                # Pose_*(nc, kpt_shape, hidc, ch)
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            elif m in (OBB_LSCSBD, OBB_LSDECD, OBB_RSCD, OBB_LSCD_LQE) and len(args) > 2 and isinstance(args[2], (int, float)):
                # OBB_*(nc, ne, hidc, ch)
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m in {RTDETRDecoder, RTDETROBB}:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif DUAL_HEAD_AVAILABLE and m in (DualHeadFusionDetect, DualHeadFusionDetectV26):
            # ===== 双头 + 融合分支聚合头 =====
            # 同一个分支同时承载 YOLO11 版（内部是 Detect）与 YOLO26 版（内部是 v26Detect），
            # 两者的构建逻辑完全一致：都只是把若干个已构建的检测头挂到聚合层上，
            # 差异体现在子头自身 forward 的返回形态，由聚合头与损失层分别处理。
            #
            # YAML 写法（B1 默认关闭）：
            #   [[rgb_head_idx, ir_head_idx, fuse_head_idx], 1, DualHeadFusionDetect, []]
            # YAML 写法（B1 启用，追加 IR 坐标系辅助头 fuse_ir_detect）：
            #   [[rgb_idx, ir_idx, fuse_idx, fuse_ir_idx], 1, DualHeadFusionDetect, []]
            #
            # 输入是已构建完成的 Detect 层（模型中先行注册），聚合头本身**零参数**，
            # 故不通过 __init__ 注册子模块，改为在实例上挂普通属性引用，避免 state_dict
            # 中出现重复键（同一 Detect 既属于 layers[i] 又属于聚合头）。
            #
            # B1 兼容性：3 输入（旧）与 4 输入（新）都受支持，且 3 输入路径行为不变。
            if not isinstance(f, (list, tuple)) or len(f) not in (3, 4):
                raise ValueError(
                    f"DualHeadFusionDetect 需要 3 个输入（RGB/IR/Fuse 三个检测头索引）"
                    f"或 4 个输入（追加 fuse_ir 辅助头），实得 f={f}"
                )
            head_modules = []
            for idx in f:
                mod = layers[idx]
                if not isinstance(mod, DETECT_CLASS):
                    raise ValueError(
                        f"DualHeadFusionDetect 的输入 {idx} 不是检测头（实为 {type(mod).__name__}）；"
                        f"请确认 YAML 中各输入分别指向 rgb/ir/fuse(/fuse_ir) 检测头层"
                    )
                head_modules.append(mod)
            has_fuse_ir = len(f) == 4

            # ===== FUSION-NECK RGB-HALF MASKING START - 融合颈定位（路线二）=====
            # 目的：为「训练时随机屏蔽融合颈 Concat 的 RGB 半边」提供结构标记。
            # 为什么在这里定位：MultiModalRouter 只记录 5 字段层的模态归属
            # （_mm_input_source），而融合颈 Concat 是 4 字段层，拿不到模态信息；
            # 但聚合头的输入索引 f 一定指向三个 Detect 头，其中 f[2] 必为融合头，
            # 融合头的输入即融合 neck 的 C3k2 输出，其上一层就是融合 Concat。
            # 该关系由 YAML 结构唯一确定，不依赖任何命名或模态推断。
            #
            # 判定链：agg.fuse_detect.f = [47, 49, 51]
            #        -> layers[47].f = -1 -> 前驱 46（融合节点）
            # 每个融合节点形如：
            #   N 系列（dual-fusion/-b1）: [[rgb_neck_idx, ir_neck_idx], 1, Concat, [1]]
            #   reliable 系列            : [[rgb_neck_idx, ir_neck_idx], 1, ReliabilityAwareFusion, [ouc, mo]]
            # 两者 f[0] 侧（索引 0）均为 RGB 半边，通道数由 ch[f[0]] 给出；
            # 屏蔽执行端 apply_fusion_rgb_mask 只作用于「输入列表的索引 0」，
            # 对 Concat 与 ReliabilityAwareFusion 语义完全一致，故二者都接受。
            #
            # 多重校验（任一不满足即放弃该分支，绝不猜测）：
            #   1) 前驱层必须是 Concat 或 ReliabilityAwareFusion，且 f 恰为二元
            #   2) 两路输入通道数必须相等（同一 stage 融合的必然特征）
            # 未通过校验时仅告警，不抛异常 —— 保证既有 YAML 构建行为不变。
            #
            # 注意：此处 layers 只含已构建层（下标 0..i-1），当前层尚未 append
            # （见本函数末的 layers.append(m_)），故索引上界必须用 len(layers)，
            # 写成 i 或 i+1 会越界。
            fusion_concat_idx = []
            if len(head_modules) >= 3 and isinstance(getattr(head_modules[2], "f", None), (list, tuple)):
                for _i in range(len(layers)):
                    layers[_i]._mm_fusion_concat = False
                for _src_idx in head_modules[2].f:
                    if not isinstance(_src_idx, int) or not 0 <= _src_idx < len(layers):
                        continue

                    def _is_fusion_node(_m) -> bool:
                        """是否为候选融合节点（Concat / 四个双光谱融合算子之一）。"""
                        if isinstance(_m, (Concat, MisalignGatedConcat)):
                            return True
                        return any(
                            _cls is not None and isinstance(_m, _cls)
                            for _cls in (
                                ReliabilityAwareFusion,
                                GatedReliabilityFusion,
                                DeformableCrossModalFusion,
                                ChannelPartitionFusion,
                            )
                        )

                    # 从融合头输入向前回溯，找到真正的融合节点。
                    #
                    # 为什么需要循环而不是只回退一层：
                    #   deform/part3h 的融合节点**直接**就是 fuse 头输入的上一层
                    #   （46/48/50）-> C3k2(47/49/51) -> fuse 头，回退一层即可命中。
                    #   但 M0(concat) 的融合节点是「Concat + 1x1 Conv 压回原通道」两层结构：
                    #   46 Concat -> 47 Conv(1x1) -> 48 C3k2 -> fuse 头。
                    #   此时回退一层只落到 47(Conv)，判定失败 -> RGB 屏蔽与 B3 蒸馏全部静默失效。
                    # 因此沿单输入链继续回溯（最多 4 跳），遇到融合节点即停止。
                    _pre_idx = _src_idx
                    _cand = None
                    for _hop in range(4):
                        _layer = layers[_pre_idx] if 0 <= _pre_idx < len(layers) else None
                        if _layer is None:
                            break
                        _pf = getattr(_layer, "f", None)
                        if isinstance(_pf, int):
                            _next = (_pre_idx - 1) if _pf == -1 else _pf
                        elif isinstance(_pf, (list, tuple)) and len(_pf) == 1:
                            _next = _pf[0]
                        else:
                            _next = None  # 多输入层：无法沿单链继续，停止回溯
                        if not isinstance(_next, int) or not 0 <= _next < len(layers):
                            break
                        _try = layers[_next]
                        if _is_fusion_node(_try):
                            _cand = _try
                            _pre_idx = _next
                            break
                        _pre_idx = _next
                    if _cand is None:
                        _cand = layers[_pre_idx] if 0 <= _pre_idx < len(layers) else None
                    if _cand is None:
                        continue
                    _is_concat = isinstance(_cand, (Concat, MisalignGatedConcat))
                    _is_reliable = ReliabilityAwareFusion is not None and isinstance(
                        _cand, ReliabilityAwareFusion
                    )
                    _is_gated = GatedReliabilityFusion is not None and isinstance(
                        _cand, GatedReliabilityFusion
                    )
                    _is_deform = DeformableCrossModalFusion is not None and isinstance(
                        _cand, DeformableCrossModalFusion
                    )
                    _is_partition = ChannelPartitionFusion is not None and isinstance(
                        _cand, ChannelPartitionFusion
                    )
                    if not (_is_concat or _is_reliable or _is_gated or _is_deform or _is_partition):
                        continue
                    _cand_f = getattr(_cand, "f", None)
                    if not isinstance(_cand_f, (list, tuple)) or len(_cand_f) != 2:
                        continue
                    if not all(isinstance(_x, int) and 0 <= _x < len(ch) for _x in _cand_f):
                        continue
                    _ch0, _ch1 = ch[_cand_f[0]], ch[_cand_f[1]]
                    if not (isinstance(_ch0, int) and isinstance(_ch1, int) and _ch0 == _ch1 and _ch0 > 0):
                        continue
                    _cand._mm_fusion_concat = True
                    _cand._mm_fusion_concat_rgb_ch = _ch0
                    fusion_concat_idx.append(_pre_idx)
            if fusion_concat_idx:
                _idx_sorted = sorted(fusion_concat_idx)
                _type_str = ",".join(
                    f"{_k}:{type(layers[_k]).__name__}" for _k in _idx_sorted
                )
                LOGGER.info(
                    f"融合颈 RGB 屏蔽：识别到融合节点层 {_idx_sorted}"
                    f"（{_type_str}；RGB 半边通道数 "
                    f"{[int(ch[layers[_k].f[0]]) for _k in _idx_sorted]}）"
                )
            else:
                LOGGER.warning(
                    "融合颈 RGB 屏蔽：未能从 DualHeadFusionDetect 反推出融合节点层"
                    "（Concat / ReliabilityAwareFusion / GatedReliabilityFusion / DeformableCrossModalFusion），"
                    "mm_fusion_rgb_mask_p 将不会生效（请检查 YAML 中融合头/融合 neck 的连接）"
                )
            # ===== FUSION-NECK RGB-HALF MASKING END =====
            agg = m.__new__(m)  # 跳过 __init__，避免注册子模块
            nn.Module.__init__(agg)
            # branch_order 恒为推理三分支（含 B1 时也不变），保证推理路径与旧权重兼容
            agg.branch_order = ("rgb", "ir", "fuse")
            agg.rgb_detect, agg.ir_detect, agg.fuse_detect = head_modules[:3]
            agg.fuse_ir_detect = head_modules[3] if has_fuse_ir else None
            agg._branch_modules = dict(zip(agg.branch_order, head_modules[:3]))
            if has_fuse_ir:
                # 追加到训练分支（不影响 branch_order / 推理路径）
                agg._branch_modules["fuse_ir"] = agg.fuse_ir_detect
            agg.nl = int(getattr(head_modules[0], "nl", 3))
            agg.nc = int(getattr(head_modules[0], "nc", -1))
            agg.no = int(getattr(head_modules[0], "no", agg.nc + 64))
            agg.reg_max = int(getattr(head_modules[0], "reg_max", 16))
            agg.stride = torch.zeros(agg.nl)
            args = []
            m_instance = agg
            m_created = True
            # 聚合层不产生新的特征图；c2 沿用首个头的输入通道，避免污染 ch[]
            c2 = ch[f[0]]
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is DConv:
            # RD: channel-preserving injection block, supports (c1, alpha=0.8, atoms=512)
            # Flexible YAML args: [] or [atoms] or [alpha] or [atoms, alpha]
            c1 = ch[f]
            c2 = ch[f]
            alpha = 0.8
            atoms = 512
            if len(args) == 1:
                if isinstance(args[0], (int, torch.Tensor)):
                    atoms = int(args[0])
                else:
                    alpha = float(args[0])
            elif len(args) >= 2:
                # assume [atoms, alpha]
                atoms = int(args[0])
                alpha = float(args[1])
            args = [c1, alpha, atoms, *args[2:]]
        elif m is TorchVision:
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        elif m is Index:
            # Select an element from a tuple/list output (e.g., from FCM)
            source_channels = ch[f]
            slot = int(args[0]) if args else 0
            if isinstance(source_channels, (list, tuple)):
                if not 0 <= slot < len(source_channels):
                    raise IndexError(
                        f"Index layer {i} selects slot {slot}, but producer {f} only exposes {len(source_channels)} outputs"
                    )
                c2 = source_channels[slot]
            else:
                # Single-width multi-output producers (e.g. FCM) store a scalar width because all slots share it.
                c2 = source_channels
            # keep args as-is: [index]
        elif m is Blocks:
            # 特殊处理Blocks模块：[ch_out, block_type, block_nums, stage_num, act, variant]
            block_type = globals()[args[1]]  # 将字符串转换为类
            c1, c2 = ch[f], args[0] * block_type.expansion
            args = [c1, args[0], block_type, *args[2:]]
        
        # === SOEP Neck模块处理 ===
        elif m is CSPOmniKernel:
            # CSPOmniKernel: 只需要dim参数，从输入通道自动推断
            c2 = ch[f]
            args = [c2]
        elif m is SPDConv:
            # SPDConv: 空间到深度卷积，需要 inc 和 ouc，可选 dimension（默认1）
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            dim_arg = args[1] if len(args) > 1 else 1
            args = [c1, c2, dim_arg]
        elif m is MFM:
            # MFM: 多尺度特征调制，需要inc列表和dim参数
            # args应该是[dim, reduction]，inc从f自动推断
            if len(args) < 1:
                raise ValueError(f"MFM requires at least 1 arg (dim), got {args}")
            inc = [ch[x] for x in f]
            dim = args[0]
            c2 = dim
            # args保持为[inc, dim, reduction(可选)]
            if len(args) >= 2:
                args = [inc, dim, args[1]]
            else:
                args = [inc, dim]
        # === C3k2 变体（需要参数重排） ===
        elif C3K2_EXTRACTION_AVAILABLE and m in (C3k2_DAttention,):
            # 兼容 YAML 写法 [c2, c3k?(bool), e?(float)]
            # 目标签名: (c1, c2, n=1, fmapsize=None, c3k=False, e=0.5, g=1, shortcut=True)
            # 注意：parse_model 已用外层 n 重复模块，内部 n 保持默认1
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c3k_flag = False
            e_val = 0.5
            if len(args) > 1 and isinstance(args[1], bool):
                c3k_flag = args[1]
            if len(args) > 2 and isinstance(args[2], (int, float)):
                e_val = float(args[2])
            # 不显式传入 fmapsize（保持为 None，内部自适应），不传 g/shortcut（使用默认）
            args = [c1, c2, 1, None, c3k_flag, e_val]
        elif m is SNI:
            # SNI: 软最近邻插值，上采样层，通道数保持不变，仅接收 up_f 参数
            # 保持 YAML 中的 args（如 [2]）不变，不注入通道参数
            c1, c2 = ch[f], ch[f]
            # args 原样传递
        elif m is GSConvE:
            # GSConvE: 卷积式增强模块，需要显式的输出通道数
            c1, c2 = ch[f], args[0] if len(args) > 0 else ch[f]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is WFU:
            # WFU: 加权特征融合上采样，双输入 [x_big, x_small]，输出通道 = x_big 通道
            c1 = [ch[x] for x in f]
            c2 = c1[0]
            args = [c1]

        # NNexpend [disabled by default]
        # Extra modules 扩展模块处理（已默认禁用，保留占位注释）

        # elif m is PST:
        #     c1,c_up,c2 = ch[f[0]],ch[f[1]],args[0]
        #     c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     args = [c1,c_up,c2, *args[1:]]
        #     args.insert(3,n)
        #     n = 1
        # elif m is C3k2_KW:  # C3k2核仓库模块 - 来自extra_modules
        #     c1, c2 = ch[f], args[0]
        #     if c2 != nc:
        #         c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     # 初始化核仓库管理器
        #     if 'warehouse_manager' not in locals():
        #         from ultralytics.nn.extra_modules.kernel_warehouse import get_warehouse_manager
        #         warehouse_manager = get_warehouse_manager()
        #     args = [c1, c2, *args[1:]]
        #     # 在特定位置插入层名称和仓库管理器
        #     args.insert(2, f'layer{i}')
        #     args.insert(2, warehouse_manager)
        #     if n > 1:
        #         args.insert(4, n)  # 由于插入了参数，调整位置
        #         n = 1
        # elif m in {C3k2_DySnakeConv, C3k2_OREPA, C3k2_REPVGGOREPA,
        #           C3k2_RFAConv, C3k2_RFCBAMConv, C3k2_RFCAConv,
        #           C3k2_VSS, C3k2_wConv}:  # 其他C3k2变体 - 来自extra_modules
        #     c1, c2 = ch[f], args[0]
        #     if c2 != nc:
        #         c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     # 标准C3k2参数处理
        #     args = [c1, c2, *args[1:]]
        #     # 特定变体的特殊处理
        #     if m is C3k2_DySnakeConv:
        #         # DySnakeConv可能需要特殊的通道处理
        #         # 但对于C3k2变体，保持标准处理
        #         pass
        #     # 为C3k2块添加n参数
        #     if n > 1:
        #         args.insert(2, n)
        #         n = 1
        #     # M/L/X尺寸的特殊处理
        #     if scale in "mlx":
        #         # C3k2_wConv有特殊的参数处理
        #         if m is C3k2_wConv:
        #             if len(args) > 0 and isinstance(args[-1], bool):
        #                 args[-1] = True
        #             elif len(args) > 1:
        #                 args[-2] = True
        #         else:
        #             # 其他C3k2变体使用标准的args[3]处理
        #             if len(args) > 3:
        #                 args[3] = True

        # 标准卷积模块处理 - 来自extra_modules（默认禁用）
        # elif m in {RFAConv, RFCBAMConv, RFCAConv,  # RFA系列卷积
        #           VSSBlock_YOLO, XSSBlock,  # Mamba VSS/XSS块
        #           CSP_FreqSpatial,  # 频空间CSP模块
        #           FeaturePyramidSharedConv,  # 特征金字塔共享卷积
        #           DSConv_YOLO13, wConv2d}:  # YOLO13系列卷积
        #     c1, c2 = ch[f], args[0]
        #     if c2 != nc:
        #         c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     args = [c1, c2, *args[1:]]
        #     # 只有XSSBlock和CSP_FreqSpatial支持重复参数
        #     if m in {XSSBlock, CSP_FreqSpatial} and n > 1:
        #         args.insert(2, n)
        #         n = 1

        # 注意力机制模块处理 - 来自extra_modules（默认禁用）
        # elif m in {EMA, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw,
        #           TripletAttention, CoordAtt,
        #         #   CBAM,
        #           BAMBlock, LSKBlock, ScConv, LAWDS, EMSConv, EMSConvP,
        #           SEAttention, CPCA, Partial_conv3, FocalModulation, EfficientAttention, MPCA, deformable_LKA,
        #           EffectiveSEModule, LSKA, SegNext_Attention, DAttention, MLCA, TransNeXt_AggregatedAttention,
        #           FocusedLinearAttention, LocalWindowAttention, ChannelAttention_HSFPN, ELA_HSFPN, CA_HSFPN, CAA_HSFPN,
        #           DySample, CARAFE, CAA, ELA, CAFM, AFGCAttention, EUCB, EfficientChannelAttention,
        #           ContrastDrivenFeatureAggregation, FSA, AttentiveLayer, EUCB_SC}:  # 所有注意力机制模块
        #     c2 = ch[f]
        #     args = [c2, *args]

        # NNexpend（默认禁用）

        # elif m in {HAFB}:
        #     if args[0] == 'head_channel':
        #         args[0] = d[args[0]]
        #     c1 = [ch[x] for x in f]
        #     c2 = make_divisible(min(args[0], max_channels) * width, 8)
        #     args = [c1, c2, *args[1:]]
        else:
            c2 = ch[f] if isinstance(f, int) else args[0]

        # ===== 单模块多输出主干：c2 为 list 时，将该模块标记为 backbone 并启用索引偏移 =====
        layer_is_multi_output = isinstance(c2, (list, tuple))
        if layer_is_multi_output:
            if layer_is_backbone:
                is_backbone = True
            if not m_created:
                if n != 1:
                    raise ValueError(f"{m.__name__ if hasattr(m, '__name__') else m}: 多输出主干不支持 repeats>1")
                m_instance = m(*args)
            m_ = m_instance
            if layer_is_backbone:
                # 供 forward/predict 路径识别
                m_.backbone = True
        else:
            # 支持在分支内自行构建实例（如 DualHeadFusionDetect 需持有其他层引用，
            # 不能用 m(*args) 常规构造）。m_created 为 True 时直接复用 m_instance。
            if m_created:
                m_ = m_instance
            else:
                m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = (i + 4 if is_backbone else i), f, t  # attach index, 'from' index, type

        # ===== MULTIMODAL EXTENSION START - 多模态属性设置 =====
        # Set multimodal attributes on the module if this layer has multimodal routing
        if mm_attributes and mm_router:
            mm_router.set_module_attributes(m_, mm_attributes)
        # ===== MULTIMODAL EXTENSION END =====
        if verbose:
            # Format args for display - convert class objects to simple names
            display_args = []
            for arg in args:
                if isinstance(arg, type):
                    display_args.append(arg.__name__)
                else:
                    display_args.append(arg)
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(display_args):<30}")  # print
        save_base = (i + 4) if is_backbone else i
        save.extend(x % save_base for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        if i == 0:
            ch = []
        if layer_is_multi_output:
            if layer_is_backbone:
                ch.extend(c2)
                # 对齐 RTDETR-main：补齐到 5 路（0-P1/2, 1-P2/4, 2-P3/8, 3-P4/16, 4-P5/32）
                for _ in range(5 - len(ch)):
                    ch.insert(0, 0)
            else:
                ch.append(list(c2))
        else:
            ch.append(c2)

    model = torch.nn.Sequential(*layers)
    
    # 将multimodal_router附加到模型上，以便后续访问
    if mm_router:
        model.multimodal_router = mm_router

    return model, sorted(save)


def yaml_model_load(path):
    """
    Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)  # noqa
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'semseg', 'depth', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "rtdetrobb" in m:
            return "obb"
        if "rtdetrdecoder" in m:
            return "detect"
        if m == "depth":
            return "depth"
        if "detect" in m:
            return "detect"
        if "semseg" in m:
            return "semseg"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, RTDETROBB):
                return "obb"
            if isinstance(m, SemSegHead):
                return "semseg"
            if isinstance(m, Depth):
                return "depth"
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, POSE_CLASS):
                return "pose"
            elif isinstance(m, OBB_CLASS):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-semseg" in model.stem or "semseg" in model.parts:
            return "semseg"
        if "-depth" in model.stem:
            return "depth"
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'semseg', 'depth', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
