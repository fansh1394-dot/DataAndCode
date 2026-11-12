# -*- coding: utf-8 -*-
"""完整训练配置（服务器用）。

与 trainMM.py（冒烟测试版，epochs=2）的区别：
    - epochs / close_mosaic 等改为完整训练值
    - 增加路线 A / B / C 的权重初始化开关
    - 增加训练前自检调用（可选）
    - 训练结束后自动提示跑配准验收

用法：
    # 路线 A：随机初始化（默认）
    python train_full.py --route A

    # 路线 B：用单模态 COCO 权重初始化骨干
    python train_full.py --route B --pretrained /root/autodl-tmp/yolo11n.pt

    # 路线 C：用已有本项目多模态权重续训/微调
    python train_full.py --route C --pretrained /root/code/.../best.pt

    # 先做训练前自检，再训练
    python train_full.py --route A --preflight

    # 覆盖批次/轮数（按显存调整）
    python train_full.py --route A --epochs 200 --batch 32

    # ---- 双头 + 融合分支（2026-09-22 新增，对应消融 A0 / B6）----
    # A0-2：双头 + DFSC（推荐）
    python train_full.py --route A --config dual-fusion --label-from dual \
        --mm-dfs-weight 0.5 --name abl_A0-2_dual_head_dfs

    # A0-3：双头但关闭 DFSC
    python train_full.py --route A --config dual-fusion --label-from dual \
        --mm-dfs-weight 0.0 --name abl_A0-3_dual_head_nodfs

    # ---- 红外单模态（消融 A2 / 泛化 G2-G4）----
    # 必须 --label-from ir，否则会继承 RGB-IR 之间 3.62pp 的系统性偏移
    python train_full.py --route A --config baseline --single-modal x \
        --label-from ir --name single_ir

    # 仅预览标签映射改写，不实际写盘
    python train_full.py --route A --config dual-fusion \
        --label-from dual --label-from-dry-run

注意：本脚本不改动 trainMM.py，两者可并存。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# 默认配置（按需修改这里，或用命令行覆盖）
# --------------------------------------------------------------------------- #

# 预置的消融配置别名 -> 模型 YAML 相对路径。
# 这些配置共享同一套"双路骨干 + P4/P5 中期融合"骨架，
# 差异仅在融合层之后插入的模块，构成标准消融对照。
CONFIG_ALIASES = {
    # 纯基线：融合后用 C2PSA，无任何特色模块 —— 论文的对照基准
    "baseline": "ultralytics/cfg/models/mm/yolo11n-mm-mid.yaml",
    # 改进版 A：融合后插入 Poly_Kernel_Inception_Block
    "pki": "ultralytics/cfg/models/mm/yolo11n-mm-mid-pki.yaml",
    # 改进版 B：融合后插入 C3k2_PKIModule（当前项目默认）
    "c3k2-pki": "ultralytics/cfg/models/mm/detect/yolo11/extraction/yolo11n-mm-c3k2_pkimodule.yaml",
    # 小目标专项：含 P2 高分辨率检测头
    "mid-p2": "ultralytics/cfg/models/mm/yolo11n-mm-mid-p2.yaml",
    # 融合策略对照
    "early": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-early.yaml",
    "asymmetric": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-asymmetric.yaml",
    "late-ref": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-late-ref.yaml",
    # 双头 + 融合分支（2026-09-22 新增）：三个独立检测头，
    # rgb 头 ← RGB 标签；ir 头 ← IR 标签(labels_ir)；fuse 头 ← RGB 标签。
    # 需配合双标签数据集（prepare_mm_dataset.py --label-from dual）使用。
    "dual-fusion": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-dual-fusion.yaml",
    # B1：融合头双坐标系监督。在 dual-fusion 基础上，为融合分支增设第二个检测头
    # fuse_ir（复用融合 neck 特征 47/49/51，← IR 坐标系标签 labels_ir/），
    # 使融合特征必须同时能解码出 RGB 系与 IR 系两组框。
    # 与 dual-fusion 的差异仅两行（多一个 Detect、聚合层 4 输入），属同一骨架的
    # 监督策略变体，可直接与 dual-fusion 对照。
    "dual-fusion-b1": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-dual-fusion-b1.yaml",
    # 非对齐核心方法：局部偏移 + 可靠性门控，沿用 B1 双坐标系监督。
    "dual-fusion-reliable": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-dual-fusion-reliable.yaml",
    # 改动 A：把可靠融合的凸组合算子换成显式门控残差（base + g*(ir-rgb)），
    # 打破「融合不可能超过单模态」的凸组合上限。其余与 reliable 版完全一致。
    "dual-fusion-gated": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-dual-fusion-gated.yaml",
    # 改动 C4：K 点可变形交叉注意力融合。每位置预测 K 个采样点 + 内容注意力聚合，
    # 突破「单一稠密偏移 + 双线性重采样」在强视差下的多假设失配与小目标细节损失。
    "dual-fusion-deform": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-dual-fusion-deform.yaml",
    # 改动 D3：通道级硬性模态分工融合。把输出通道切分为 [RGB 专属 | IR 专属 | 交互]
    # 三段，其中 IR 专属段无 gate，IR 特征直接进输出，结构上无法被训练关闭。
    # 直接针对消融诊断暴露的「fuse 头对 IR 半边零响应」根因。
    "dual-fusion-partition": "ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-dual-fusion-partition.yaml",

    # ---- YOLO26 侧（v26 前缀）：骨干换 YOLO26 + 头换 v26Detect(end2end) ----
    # 与 YOLO11 同名配置逐层同构，只换代际；结果目录请配 --project YOLO26/ResTest。
    "v26-single-baseline": "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-single-baseline.yaml",
    "v26-dual-fusion-deform": "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml",
}

#: 使用「双头 + 融合分支」聚合层的配置别名集合（含 B1 变体）。
DUAL_HEAD_CONFIGS = (
    "dual-fusion", "dual-fusion-b1", "dual-fusion-reliable", "dual-fusion-gated",
    "dual-fusion-deform", "dual-fusion-partition", "v26-dual-fusion-deform",
    # YOLO26 下同样使用 DualHeadFusionDetectV26 聚合层的真实配置文件。
    # 原先未收录：这些配置的历史训练都没有传 --mm-dual-* 参数，校验分支从未执行到，
    # 缺口未被发现；一旦传入屏蔽/蒸馏参数就会被误判为"单头融合"而拒绝。
    "yolo26n-mm-dual-fusion-concat",
    "yolo26n-mm-dual-fusion-align3h",
    "yolo26n-mm-dual-fusion-part3h",
)

#: 含显式门控（GatedReliabilityFusion / DeformableCrossModalFusion / ChannelPartitionFusion）
#  的配置，这些配置下 L_gate 门控激活正则才有意义。
GATED_CONFIGS = ("dual-fusion-gated", "dual-fusion-deform", "dual-fusion-partition", "v26-dual-fusion-deform")

DEFAULTS = {
    # 模型结构：**故意不设默认值**。
    # 原默认是 CONFIG_ALIASES["baseline"]（YOLO11 配置），一旦漏写 --config，
    # 一次本该跑 YOLO26 的实验会完整跑成 YOLO11 单头基线并写进 ResTest/，
    # 全程零告警。现在强制要求显式 --config（见 main 里的解析分支）。
    "model_yaml": None,
    "scale": "n",

    # 数据（服务器路径）—— 原值 /root/autodl-tmp/uav_data/data.yaml 是旧机的过时路径
    "data": "/root/rivermind-data/uav_data_v2/data.yaml",

    # 完整训练超参（RTX 4090 24G 调优，详见下方说明）
    "epochs": 150,          # 冒烟测试是 2；单类别 2.8 万训练对，150 足够收敛
    "batch": 32,            # 4090 24G：batch 32 实测占用约 14-17G；显存紧可降到 16
    "imgsz": 640,           # 【不要降低】目标偏小(中位 37px)，降到 512 会损失精度
    "workers": 16,          # 4090 机器 CPU 通常充足；不够降到 8
    # 缓存模式：none（默认）
    #   ram  : 解码后图像全放内存。实测需 train 26GB + val 10.6GB ≈ 37GB，
    #          本项目规模必然 OOM，禁用。
    #   disk : 缓存落盘，内存占用可忽略，但需约 37GB 磁盘空间。
    #          仅当数据盘剩余 > 60GB 时才建议使用。
    #   none : 不缓存，靠 workers 并行读取。在 NVMe 高速盘上（实测 1.6GB/s）
    #          速度损失有限（约 5-15%），且不占内存与磁盘，最稳妥。
    "cache": "none",
    "close_mosaic": 15,     # 最后 15 轮关闭 mosaic，利于收敛（冒烟测试是 0）
    "patience": 50,         # 早停耐心值
    "save_period": 10,      # 每 10 轮存一次检查点
    "amp": True,            # 混合精度；若 loss 出现 nan 改为 False
    "plots": True,          # 训练期绘制 batch/标签图；显存紧张时可关（不影响指标）

    # 输出
    "project": "ResTest",
    "name": "yolomm-c3k2-pkimodule-full",
}

ROUTE_HELP = {
    "A": "随机初始化（不加载预训练权重）",
    "B": "单模态 COCO 权重初始化骨干（需 --pretrained 指向单模态 .pt）",
    "C": "已有本项目多模态权重续训 / 微调（需 --pretrained 指向多模态 .pt）",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(
        description="完整训练配置（服务器用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="路线说明：\n" + "\n".join(f"  {k}: {v}" for k, v in ROUTE_HELP.items()))
    ap.add_argument("--route", choices=["A", "B", "C"], default="A",
                    help="权重初始化路线，默认 A")
    ap.add_argument("--pretrained", default=None,
                    help="预训练权重路径（路线 B / C 必填）")
    ap.add_argument("--build-nc", type=int, default=None, dest="build_nc",
                    help="路线 C 换类别数时显式指定构建用 nc（默认沿用 ckpt 的 nc；"
                         "形状不匹配的检测头会被跳过并重新初始化）")
    ap.add_argument("--config", default=None,
                    help="模型配置别名或 YAML 路径。可用别名：" +
                         " | ".join(CONFIG_ALIASES.keys()) +
                         "（默认 baseline）")
    ap.add_argument("--data", default=None, help="data.yaml 路径")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None,
                    help="初始学习率覆盖（微调建议 0.001~0.003；不传则沿用默认 0.01）")
    ap.add_argument("--patience", type=int, default=None,
                    help="早停耐心值（默认沿用 CFG 的 50）。150 轮实验建议设 25~30，"
                         "使过拟合能在退步后及时终止，同时给足波动容忍度")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--box", type=float, default=None,
                    help="回归损失增益（default.yaml 默认 7.5）。提高可加强定位，"
                         "直接作用于 mAP50-95")
    ap.add_argument("--dfl", type=float, default=None,
                    help="DFL 损失增益（default.yaml 默认 1.5）。提高可加强边界回归")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子（默认 0；正式实验建议 0/1/2 三次独立训练）")
    ap.add_argument("--cache", default=None,
                    help="缓存模式：none | disk | ram（默认 none。ram 需约 37G 内存会 OOM；"
                         "disk 需约 37G 磁盘空间，仅在剩余 >60G 时用）")
    ap.add_argument("--plots", choices=["true", "false", "True", "False"], default=None,
                    help="训练期绘图开关（默认 true）。关闭后不生成 batch/标签图，"
                         "可减少主机侧 batch 持有与 matplotlib 开销；不影响任何指标")
    ap.add_argument("--device", default="0", help="GPU 编号，如 0 或 0,1")
    ap.add_argument("--amp", choices=["true", "false", "True", "False"], default=None,
                    help="混合精度开关（默认沿用 CFG 的 true）。设为 false 关闭 AMP、"
                         "以 fp32 全精度训练，显存约翻倍但与历史 fp32 实验严格可比。"
                         "论文主表需与旧实验同条件对比时使用。")
    ap.add_argument("--single-modal", default=None,
                    choices=["rgb", "thermal", "depth", "x", "fuse"],
                    help="单模态消融：构建期注入单模态结构。"
                         "可选 rgb | thermal | depth | x | fuse（x 即红外）。"
                         "不设则为双模态融合。")
    ap.add_argument("--modality", default=None,
                    help="模态消融（运行时）：如 rgb。注意与 --single-modal 不同。")
    ap.add_argument("--project", default=None,
                    help="输出根目录（相对项目根，如 YOLO26/ResTest）。默认沿用 CFG 的 ResTest，"
                         "YOLO11 实验不受影响")
    ap.add_argument("--name", default=None, help="输出目录名")
    ap.add_argument("--resume", action="store_true", help="从上次中断处续训")
    ap.add_argument("--preflight", action="store_true",
                    help="训练前先跑 tools/preflight_train.py 自检")
    ap.add_argument("--force", action="store_true",
                    help="自检不通过时仍然继续训练")
    ap.add_argument("--safe-augment", action="store_true",
                    help="重建实验的稳定模式：关闭 mosaic/mixup/cutmix/copy-paste，先验证非对齐监督契约")

    # ------------------------------------------------------------------ #
    # 双标签数据集（2026-09-22 新增）
    # ------------------------------------------------------------------ #
    grp_label = ap.add_argument_group(
        "双标签数据集",
        "控制 labels/labels_ir 两套标签的来源。单模态红外训练（A2/G2-G4）"
        "必须用 IR 标签，否则会继承 RGB-IR 之间的系统性偏移。")
    grp_label.add_argument("--label-from", default=None,
                           choices=["vis", "ir", "dual", "keep"],
                           help="标签来源：vis=仅可见光标签(默认行为，沿用 data.yaml)；"
                                "ir=主标签目录指向 labels_ir（红外单模态必用）；"
                                "dual=双标签模式(labels/ + labels_ir/)；"
                                "keep=不修改 data.yaml（默认）。"
                                "会按需改写 data.yaml 的 labels 映射。")
    grp_label.add_argument("--label-from-dry-run", action="store_true",
                           help="只打印将要写入的 labels 映射，不实际改写 data.yaml")

    # ------------------------------------------------------------------ #
    # 双头 + 融合分支（2026-09-22 新增，仅 --config dual-fusion 时生效）
    # ------------------------------------------------------------------ #
    grp_dual = ap.add_argument_group(
        "双头 + 融合分支",
        "仅在 --config dual-fusion / dual-fusion-b1 下生效。检测头各自监督："
        "rgb 头 ← RGB 标签；ir 头 ← IR 标签；fuse 头 ← RGB 标签；"
        "fuse_ir 头 ← IR 标签（仅 B1 变体存在）。")
    grp_dual.add_argument("--mm-dual-w-rgb", type=float, default=None,
                          help="RGB 头检测损失权重（默认 1.0）")
    grp_dual.add_argument("--mm-dual-w-ir", type=float, default=None,
                          help="IR 头检测损失权重（默认 1.0）。"
                               "若 IR 标签质量较差可适当降低")
    grp_dual.add_argument("--mm-dual-w-fuse", type=float, default=None,
                          help="融合头检测损失权重（默认 1.0）。"
                               "论文主指标来自此头，通常不建议降低")
    # ---- B1（融合头双坐标系监督）新增 ----
    grp_dual.add_argument("--mm-dual-w-fuse-ir", type=float, default=None,
                          help="B1：融合分支 IR 坐标系头（fuse_ir）损失权重（默认 1.0）。"
                               "仅在 --config dual-fusion-b1 下存在该头，其它配置下该参数"
                               "即使传入也不产生任何计算。若 IR 标签质量较差可适当降低")
    grp_dual.add_argument("--mm-dual-w-fuse-ir-consist", type=float, default=None,
                          help="B1：fuse 与 fuse_ir 两头的分类响应一致性权重"
                               "（默认 0.0=关闭）。作用于同一组融合特征的两个坐标系"
                               "输出，用于抑制同一特征给出矛盾语义")
    grp_dual.add_argument("--mm-dual-w-fuse-ir-distill", type=float, default=None,
                          help="B3：drop-RGB 条件下的 fuse←ir 蒸馏权重（默认 0.0=关闭，"
                               "推荐 1.0）。仅在融合颈 RGB 半边被屏蔽（--mm-fusion-rgb-mask>0）"
                               "的 batch 上生效，强制融合分支的分类响应向 ir 分支对齐，"
                               "把「用 IR 也能检测」变成显式优化目标。需配合屏蔽一起用，"
                               "单独给出不产生任何计算。")
    grp_dual.add_argument("--mm-dfs-weight", type=float, default=None,
                          help="DFSC 目标级跨模态语义一致性损失总权重"
                               "（默认 0.0=关闭，推荐 0.5）")
    grp_dual.add_argument("--mm-dfs-mode", default=None, choices=["cls", "feat"],
                          help="DFSC 采样位置：cls=分类分支(cv3)输出（默认，语义更集中）；"
                               "feat=neck 特征图（更底层，开销更大）")
    grp_dual.add_argument("--mm-dfs-temperature", type=float, default=None,
                          help="DFSC 余弦相似度温度（默认 1.0，越大约束越严格）")
    grp_dual.add_argument("--mm-dual-consist-weight", type=float, default=None,
                          help="跨头分类响应一致性权重（默认 0.0=关闭）")
    # ---- P0：门控激活正则（仅 dual-fusion-gated 有意义）----
    grp_dual.add_argument("--mm-alpha-anchor-weight", type=float, default=None,
                          dest="mm_alpha_anchor_weight",
                          help="错位感知门控的锚定正则权重（0 关闭；默认读 hyp）")
    grp_dual.add_argument("--mm-alpha-min", type=float, default=None,
                          dest="mm_alpha_min",
                          help="门控下界 alpha_min（默认 0.2）")
    grp_dual.add_argument("--mm-gate-reg-weight", type=float, default=None,
                          dest="mm_gate_reg_weight",
                          help="门控激活正则 L_gate 权重（默认 0.0=关闭，推荐 0.05~0.2）。"
                               "惩罚 GatedReliabilityFusion 门控 std 低于目标值，"
                               "阻止门控塌缩到中性 0.5；仅在 --config dual-fusion-gated 下生效。")
    grp_dual.add_argument("--mm-gate-reg-target", type=float, default=None,
                          dest="mm_gate_reg_target",
                          help="门控 std 目标值（默认 0.15）。std 低于该值才产生梯度。")
    # ---- B4-b：伪融合根因修复（mask 条件下 fuse 头改用 IR 标签）----
    grp_dual.add_argument("--mm-fuse-ir-label-on-mask", default=None,
                          dest="mm_fuse_ir_label_on_mask",
                          choices=["true", "false", "True", "False"],
                          help="伪融合根因修复开关（默认 true=启用）。开启后，被融合颈 RGB "
                               "半边屏蔽的样本，其 fuse 头改用 IR 坐标系标签监督，把"
                               "「必须从 IR 解码目标」变成显式优化目标。仅在同时开启 "
                               "--mm-fusion-rgb-mask>0 时有样本命中，其余情况逐位不变。")
    # ---- 模态 dropout（B1-2 用：让模型见过「单模态失效」样本）----
    grp_dual.add_argument("--mm-modal-dropout", type=float, default=None, dest="mm_modal_dropout_p",
                          help="模态随机遮挡概率（默认 0.0=关闭，推荐 0.3）。"
                               "训练时随机把某一模态的局部区域置零，制造「单模态局部失效」样本。"
                               "困难子集全部是单模态失效帧，而默认训练集不含此类样本 —— "
                               "不开这项，模型在 A 组上没有任何泛化依据。")
    # ---- 裁切放大（E19：训练期提高目标网格占位，推理仍 640）----
    grp_dual.add_argument("--mm-crop-zoom", type=float, default=None,
                          dest="mm_crop_zoom_p", metavar="P",
                          help="裁切放大概率（默认 0.0=关闭，推荐 0.6）。"
                               "以 GT 为中心随机裁切原图 0.6~0.85 边长后 letterbox 到 imgsz，"
                               "等效放大 1.2~1.7 倍；不改变推理输入尺寸。")
    grp_dual.add_argument("--mm-crop-zoom-side", type=float, nargs=2, default=None,
                          dest="mm_crop_zoom_side", metavar=("MIN", "MAX"),
                          help="裁切边长占原图比例范围（默认 0.6 0.85）")
    # ---- 目标尺度重分布（DIV-B：制造大目标训练样本）----
    grp_dual.add_argument("--mm-target-repaste", type=float, default=None,
                          dest="mm_target_repaste_p", metavar="P",
                          help="目标重粘贴概率（默认 0.0=关闭，推荐 0.5）。"
                               "把已有 GT 裁出、放大 1.5~4 倍后贴回原图，"
                               "用于补足 >=64px 大目标（v2 中仅占 1.2~2.3%%）。"
                               "RGB 与 IR 使用同一坐标与缩放，不破坏跨模态对应。")
    grp_dual.add_argument("--mm-target-repaste-scale", type=float, nargs=2, default=None,
                          metavar=("MIN", "MAX"), help="放大倍数范围（默认 1.5 4.0）")
    grp_dual.add_argument("--mm-target-repaste-ctx", type=float, default=None,
                          help="源框上下文余量占框尺寸的比例（默认 0.2）")
    grp_dual.add_argument("--mm-target-repaste-feather", type=int, default=None,
                          help="粘贴边缘羽化半径像素（默认 3，0=硬粘贴）")
    grp_dual.add_argument("--mm-target-repaste-drop-src", type=float, default=None,
                          dest="mm_target_repaste_drop_src_p",
                          help="粘贴后丢弃原目标的概率（默认 0.0=保留）")
    # ---- 异步几何扰动（双光谱错位鲁棒：把容忍度从标定精度搬到数据分布）----
    grp_dual.add_argument("--mm-async-geo-p", type=float, default=None, dest="mm_async_geo_p",
                          metavar="P",
                          help="单侧模态平移/旋转概率（默认 0.0=关闭，建议 0.5）。"
                               "依据错位退化曲线：1px@640 已 -2.2%%，4px -15.5%%，8px 融合增益转负。"
                               "注意：data.yaml 必须 align: false，否则数据集初始化直接报错。")
    grp_dual.add_argument("--mm-async-geo-translate-px", type=int, nargs=2, default=None,
                          metavar=("MIN", "MAX"),
                          help="平移像素范围（letterbox 后的 imgsz 空间，默认 1 2）")
    grp_dual.add_argument("--mm-async-geo-rotate-deg", type=float, nargs=2, default=None,
                          metavar=("MIN", "MAX"), help="旋转角度范围（度，默认 -0.5 0.5）")
    grp_dual.add_argument("--mm-async-geo-target", default=None, choices=["x", "rgb", "random"],
                          help="被扰动的模态：x=IR（默认）、rgb、random")
    grp_dual.add_argument("--mm-async-geo-border", type=int, default=None,
                          help="warp 边界填充值（默认 114）")
    # ---- 背景移植（DIV-C：把目标放进纹理背景）----
    grp_dual.add_argument("--mm-bg-swap", type=float, default=None,
                          dest="mm_bg_swap_p", metavar="P",
                          help="背景移植概率（默认 0.0=关闭，推荐 0.3）。"
                               "用另一序列的无目标块替换整幅背景，原 GT 前景回贴原位置；"
                               "RGB 与 IR 取同一序列同一坐标的背景块（C2 变体）。")
    grp_dual.add_argument("--mm-bg-bank", default=None,
                          help="背景库目录（由 scripts/build_bg_bank.py 生成）")
    grp_dual.add_argument("--mm-bg-ctx", type=float, default=None,
                          help="前景回贴时保留的上下文余量比例（默认 0.2）")
    grp_dual.add_argument("--mm-bg-feather", type=int, default=None,
                          help="前景回贴边缘羽化半径像素（默认 8）")
    # ---- 路线二：融合颈 RGB 半边随机屏蔽（打破「伪融合」退化）----
    # 诊断结论：在融合颈 Concat 处把 RGB 半边置零后，fuse 头损失直接归零
    # （0.05 -> 0.00），rgb/ir 头完全不变 —— 融合头实际只读 RGB 半边，
    # IR 半边被优化成近零通路，所谓「融合增益」并不来自红外。
    # 本组参数在训练时按概率切断融合分支的 RGB 支路，强迫 IR 通路吃到梯度。
    grp_dual.add_argument("--mm-fusion-rgb-mask", type=float, default=None,
                          dest="mm_fusion_rgb_mask_p", metavar="P",
                          help="融合颈 RGB 半边屏蔽概率（默认 0.0=关闭，推荐 0.3）。"
                               "仅训练生效；eval/验证/推理恒不激活，既有 best.pt 推理行为不变。"
                               "与 --mm-modal-dropout 的区别：后者会让整个 RGB backbone 塌陷、"
                               "rgb 头一起坏掉，本项只切断融合分支的一条支路，rgb 头通路保持完整作为对照。")
    grp_dual.add_argument("--mm-fusion-rgb-mask-epoch-start", type=int, default=None,
                          dest="mm_fusion_rgb_mask_epoch_start", metavar="N",
                          help="从第 N 个 epoch 起才启用屏蔽（默认 0=立即启用）。"
                               "设定 N>0 时前 N 个 epoch 保持原始数据分布，让模型先收敛出基线表示，"
                               "再施加通路约束，避免早期梯度冲突。")
    grp_dual.add_argument("--mm-fusion-rgb-mask-fill", default=None,
                          choices=["zero", "channel_mean"],
                          dest="mm_fusion_rgb_mask_fill",
                          help="RGB 半边的填充方式（默认 zero）。"
                               "zero=置零，与诊断脚本同口径，实验结论可直接对比；"
                               "channel_mean=替换为通道空间均值，可避免置零改变后续 C3k2 中 "
                               "BatchNorm 的输入均值偏置。")
    # ---- D2-P1：通路级 dropout（逐样本强置融合算子 gate=1）----
    # D0 判别实验证实：fuse 头的训练 cls_loss 能与 fuse_ir 头完全持平（0.388 vs 0.390），
    # 但伪融合探针下 fuse=0.0007 而 fuse_ir=0.6485 —— 损失相等、通路依赖相反。说明
    # 融合特征里 RGB/IR 并存、损失对「用哪条通路」无约束，fuse 头总能走 RGB 捷径。
    # 本项在融合算子输出表达式 fused=(1-gate)*rgb+gate*ir_attended 上逐样本强置
    # gate=1（fused=ir_attended），直接消去 RGB 项，物理上无法绕过；与探针
    # drop_rgb_term 同数学位置，训练扰动与评测判据构成同构因果链。
    grp_dual.add_argument("--mm-gate-drop-p", type=float, default=None,
                          dest="mm_gate_drop_p", metavar="P",
                          help="通路级 dropout 概率 (0,1]，逐样本强置融合算子 gate=1 使 "
                               "fused 走纯 IR（默认 0.0=关闭）。仅支持 --config dual-fusion-deform；"
                               "仅训练生效，eval/验证/推理与既有 best.pt 推理行为完全不变。"
                               "与 --mm-fusion-rgb-mask 的区别：后者在融合节点输入端置零 rgb_half，"
                               "但 rgb_proj(0)=bias≠0 使 RGB 分量仍留在 fused 中、屏蔽被绕过；"
                               "本项在表达式层面消去，无法绕过。")
    # ---- D3-P2：强制融合节点输出全部来自 IR（判别实验）----
    # 既有六条修复路线全部无效后，本项绕开所有间接手段，直接把融合算子输出整体
    # 替换为 IR 特征（fused = ir_attended），且**训练与验证均生效**。
    # 与「置零 RGB 段」的关键差异：这里用 IR 特征**原位填充**，幅值与语义保持 IR
    # 的自然分布，下游 3x3 卷积无从绕过（D3-P1 的通道硬划分正是被下游卷积抹平的）。
    # 用于回答：给 fuse 头一个「只能用 IR」的输入，它究竟能不能学会？
    #   能（mAP >> 地板）-> 瓶颈在训练信号分配；
    #   不能（≈0.001）  -> IR 特征对 fuse 头根本不可用。
    grp_dual.add_argument("--mm-force-ir-fusion", action="store_true",
                          dest="mm_force_ir_fusion",
                          help="D3-P2 判别实验：强制融合算子输出全部通道只含 IR 特征"
                               "（fused=ir_attended，原位填充而非置零）。训练与验证均生效。"
                               "仅支持 --config dual-fusion-partition（默认关闭）。")
    grp_dual.add_argument("--mm-fusion-rgb-mask-scope", default=None,
                          choices=["batch", "sample"],
                          dest="mm_fusion_rgb_mask_scope",
                          help="屏蔽的抽样粒度（默认 batch，仅在 --mm-fusion-rgb-mask > 0 时有效）。"
                               "batch=每个 forward 掷一次硬币、整批（三层融合颈）同屏蔽；"
                               "sample=逐样本掷签，同一 batch 内混有屏蔽/未屏蔽帧。"
                               "逐样本迫使融合头在同一次 forward 内同时支持 RGB 与 IR，"
                               "从而消除逐 batch 口径下的 RGB 捷径（诊断显示 p=0.1/0.2 逐 batch 均无效）。")
    # ---- 判别实验：监督坐标系（2026-10-14）----
    grp_dual.add_argument("--mm-fuse-ir-label-always", default=None,
                          dest="mm_fuse_ir_label_always",
                          choices=["true", "false", "True", "False"],
                          help="判别实验：fuse 头无条件改用 IR 坐标系标签监督（默认 false）。"
                               "与 --mm-fuse-ir-label-on-mask 的区别：后者只在样本被 RGB 屏蔽时切换，"
                               "本项对所有样本恒定切换。用于检验伪融合瓶颈是否在监督坐标系。")
    grp_dual.add_argument("--mm-freeze-for-fusion-probe", action="store_true",
                          dest="mm_freeze_for_fusion_probe",
                          help="判别实验：冻结 RGB/IR 骨干与 rgb/ir 检测头，"
                               "只训练融合颈（含融合算子）与 fuse 头。配合 "
                               "--mm-fuse-ir-label-always true 做监督坐标系判别。"
                               "仅支持双头架构（--config dual-fusion-*）。")
    grp_dual.add_argument("--mm-freeze-probe-scope", default=None,
                          choices=["neck_head", "head"],
                          dest="mm_freeze_probe_scope",
                          help="冻结探针的解冻范围（默认 neck_head，仅在 "
                               "--mm-freeze-for-fusion-probe 下有效）。"
                               "neck_head=解冻融合算子+融合颈+fuse 头；"
                               "head=只解冻 fuse 头，特征完全冻结，用于把"
                               "「特征不可用」与「检测头学不动」彻底分开。")
    # ---- 标签分配相似度（COXNet GeoShape 移植）----
    grp_dual.add_argument("--loss-assign-metric", default=None,
                          choices=["iou", "geoshape"],
                          dest="loss_assign_metric",
                          help="标签分配的定位相似度（默认 iou，保持 stock CIoU 行为）。"
                               "geoshape=启用 COXNet 的 GeoShape 相似度，把 IoU、归一化中心距离"
                               "与长宽比差异三个线索做凸组合，降低经典 IoU 对微小目标几像素"
                               "位移的过度敏感。对所有检测头（rgb/ir/fuse/fuse_ir）统一生效。")
    # ---- 回归损失类型（极小目标定位专项，2026-11-04 新增）----
    # 与 --loss-assign-metric 不同：后者改的是「标签分配给哪个 anchor」，
    # 本参数改的是「已分配目标之后用哪种相似度算回归损失」。
    grp_dual.add_argument("--loss-box", default=None,
                          choices=["iou", "giou", "diou", "ciou", "siou", "eiou",
                                   "wiou", "alphaiou", "nwd", "mpdiou", "ciou_nwd"],
                          dest="loss_box",
                          help="检测框回归损失类型（默认沿用 default.yaml 的 ciou）。"
                               "nwd=Normalized Wasserstein Distance，把框建模为高斯分布、"
                               "以 Wasserstein 距离度量，对微小目标的几像素位移不敏感；"
                               "wiou=Wise-IoU 动态非单调聚焦；"
                               "ciou_nwd=CIoU 主体 + λ=0.05 的 NWD 辅助项（混合回归）。"
                               "对所有检测头统一生效。")
    # ---- 跨分辨率特征蒸馏（尺度蒸馏，2026-11-05 新增）----
    grp_dual.add_argument("--distill", nargs=2, default=None,
                          metavar=("YAML", "MODE"),
                          help="蒸馏配置：[yaml 路径, 模式]。模式 feature/output/both。"
                               "配合教师 YAML 的 teachers[*].imgsz 可做跨分辨率尺度蒸馏"
                               "（教师按 imgsz 前向，特征由 F.interpolate 对齐到学生）。")
    # ---- 几何增强（双模态同步）----
    # 原生 RandomPerspective 用同一仿射矩阵同时变换 6 通道图与 instances_ir，
    # 已验证 RGB/IR 框逐框坐标差为 0，不会破坏 align:true 前提。
    # 默认 None = 不覆盖 ultralytics 默认值（degrees/shear/perspective 均为 0），
    # 保证历史实验口径不变。
    ap.add_argument("--degrees", type=float, default=None,
                    help="随机旋转角度范围 ±deg（默认不设置，沿用 ultralytics 默认 0）")
    ap.add_argument("--shear", type=float, default=None,
                    help="随机剪切角度范围 ±deg（默认不设置，沿用 ultralytics 默认 0）")
    ap.add_argument("--perspective", type=float, default=None,
                    help="随机透视强度 0~0.001（默认不设置，沿用 ultralytics 默认 0）")
    return ap.parse_args()


def run_preflight(data: str) -> bool:
    """调用训练前自检脚本。

    Args:
        data: data.yaml 路径。

    Returns:
        True 表示自检通过。
    """
    import subprocess
    script = PROJ / "tools" / "preflight_train.py"
    if not script.exists():
        print("[警告] 未找到 tools/preflight_train.py，跳过自检")
        return True

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    r = subprocess.run([sys.executable, str(script), "--data", data],
                       cwd=str(PROJ), env=env)
    return r.returncode == 0


def check_route(route: str, pretrained: str | None) -> None:
    """校验路线与权重参数是否匹配。

    Args:
        route: 路线字母。
        pretrained: 权重路径。

    Raises:
        SystemExit: 参数不合法时退出。
    """
    if route in ("B", "C") and not pretrained:
        print(f"[错误] 路线 {route} 必须提供 --pretrained 权重路径。")
        print(f"       路线说明：{ROUTE_HELP[route]}")
        raise SystemExit(2)
    if pretrained and not Path(pretrained).exists():
        print(f"[错误] 权重文件不存在: {pretrained}")
        raise SystemExit(2)
    if route == "B":
        print("[提示] 路线 B 的前提是单模态权重与多模态模型参数名能对上。")
        print("       若 load 时 Transferred 占比很低，建议改用路线 A。")
        print("       可用 python tools/preflight_train.py --test-weight <路径> 实测。")


# --------------------------------------------------------------------------- #
# 标签来源控制（2026-09-22 新增）
# --------------------------------------------------------------------------- #

#: 各标签模式要写入 data.yaml 的 labels 映射
LABEL_MODE_MAP = {
    # 仅可见光：主标签目录 = labels/，次标签同为 labels/（不启用双标签）
    "vis": {"rgb": "labels", "ir": "labels"},
    # 仅红外：主标签目录 = labels_ir/（红外单模态必用，绕开 RGB-IR 系统性偏移）
    "ir": {"rgb": "labels_ir", "ir": "labels_ir"},
    # 双标签：rgb 头用 visible.json 标签，ir 头用 infrared.json 标签
    "dual": {"rgb": "labels", "ir": "labels_ir"},
}


def _dump_yaml_fallback(data: dict, path: Path) -> None:
    """无 PyYAML 时的极简 YAML 写出（仅覆盖本脚本用到的顶层结构）。"""
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        elif isinstance(v, (list, tuple)):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_label_from(data_path: str, mode: str | None, dry_run: bool = False) -> bool:
    """按 ``--label-from`` 改写 data.yaml 的 ``labels`` 映射。

    为什么需要这一步
    ----------------
    数据层通过 data.yaml 的 ``labels`` 映射决定从哪个目录读标签：

    - ``labels: {rgb: labels, ir: labels_ir}`` -> 启用双标签（dual_labels_configured=True）
    - 无 ``labels`` 映射 -> 退回单标签，**IR 头将得不到监督**（静默降级，不报错）

    红外单模态实验（消融 A2、泛化 G2–G4）必须把主标签目录指向 ``labels_ir``，
    否则会继承 RGB 与 IR 之间 3.62 pp 的系统性偏移，使对照失去意义。

    Args:
        data_path: data.yaml 路径。
        mode: ``vis`` / ``ir`` / ``dual``；None 或 ``keep`` 表示不改写。
        dry_run: True 时只打印，不写盘。

    Returns:
        True 表示成功（含"未做修改"的情况）。
    """
    if not mode or mode == "keep":
        return True

    mapping = LABEL_MODE_MAP.get(mode)
    if mapping is None:
        print(f"[错误] 未知的 --label-from 取值: {mode}")
        return False

    p = Path(data_path)
    if not p.exists():
        print(f"[错误] data.yaml 不存在，无法改写标签映射: {p}")
        return False

    try:
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except ImportError:
        yaml = None
        # 无 PyYAML：用极简解析（本脚本只关心顶层键）
        data = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            k, _, v = s.partition(":")
            if line.startswith((" ", "\t")):
                continue
            data[k.strip()] = v.strip().split(" #")[0].strip()
    except Exception as e:
        print(f"[错误] 解析 data.yaml 失败: {type(e).__name__}: {e}")
        return False

    old = data.get("labels")
    data["labels"] = dict(mapping)

    print(f"\n[标签来源] --label-from {mode}")
    print(f"           data.yaml : {p}")
    print(f"           labels 映射: {old}  ->  {data['labels']}")

    if dry_run:
        print("           (dry-run，未写盘)")
        return True

    # 备份一次原文件，便于回滚
    bak = p.with_suffix(".yaml.bak_label")
    if not bak.exists():
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"           已备份原文件 -> {bak.name}")

    try:
        if yaml is not None:
            p.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        else:
            _dump_yaml_fallback(data, p)
    except Exception as e:
        print(f"[错误] 写入 data.yaml 失败: {type(e).__name__}: {e}")
        return False

    # 与数据层保持一致的提示
    if mode == "dual":
        print("           提示：训练日志应出现 "
              "'IR labels: N images' 且 dual_labels_configured=True")
    elif mode == "ir":
        print("           提示：主标签目录已指向 labels_ir，"
              "适用于红外单模态（A2 / G2-G4）")
    return True


def main() -> int:
    """训练主流程。"""
    args = parse_args()
    cfg = dict(DEFAULTS)
    for k in ("data", "epochs", "batch", "imgsz", "workers", "name", "cache"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if getattr(args, "plots", None) is not None:
        cfg["plots"] = str(args.plots).lower() == "true"
    if getattr(args, "patience", None) is not None:
        cfg["patience"] = int(args.patience)

    # --amp 覆盖：字符串 -> bool。设 false 则关闭 AMP 以 fp32 训练（显存约翻倍，
    # 但与历史 fp32 实验严格可比）。未指定时沿用 CFG 默认（true）。
    if args.amp is not None:
        cfg["amp"] = str(args.amp).lower() == "true"

    # cache 归一化：none/False -> False；ram/disk -> 对应字符串
    cache_mode = cfg["cache"]
    if isinstance(cache_mode, str):
        cm = cache_mode.strip().lower()
        cfg["cache"] = False if cm in ("none", "false", "0", "") else cm

    # --config 解析：别名或直接给 YAML 路径
    # 「未传 --config」绝不能静默落到 YOLO11 默认配置（那会让一次本该跑 YOLO26 的
    # 实验完整跑成 YOLO11 单头基线、并写进 YOLO11 的 ResTest/），故直接报错。
    config_label = None
    if args.config:
        key = args.config.strip()
        if key in CONFIG_ALIASES:
            cfg["model_yaml"] = CONFIG_ALIASES[key]
            config_label = key
        else:
            cfg["model_yaml"] = key
            config_label = Path(key).stem
    else:
        print("[错误] 必须显式指定 --config（如 v26-dual-fusion-deform / v26-single-baseline）。")
        print("       为了杜绝「漏参数静默跑成另一代」的事故，本脚本不再使用默认模型配置。")
        return 2

    # ---- 双头 + 融合分支的参数校验（必须先于自动命名，因为它会补 args.label_from）----
    is_dual_head = config_label in DUAL_HEAD_CONFIGS
    #: 融合分支额外挂载 IR 坐标系检测头（fuse_ir，即四头）的配置。
    #: v26-dual-fusion-deform 的 YAML 同样挂 fuse_ir（第 4 头），必须计入，
    #: 否则 --mm-dual-w-fuse-ir 会被 627 行误拒、loss 项数提示也会错报成 9 项。
    is_b1 = config_label in (
        "dual-fusion-b1", "dual-fusion-reliable", "dual-fusion-gated",
        "v26-dual-fusion-deform",
    )
    dual_args_used = any([
        args.mm_dual_w_rgb is not None, args.mm_dual_w_ir is not None,
        args.mm_dual_w_fuse is not None, args.mm_dfs_weight is not None,
        args.mm_dfs_mode is not None, args.mm_dfs_temperature is not None,
        args.mm_dual_consist_weight is not None,
        args.mm_dual_w_fuse_ir is not None,
        args.mm_dual_w_fuse_ir_consist is not None,
        args.mm_dual_w_fuse_ir_distill is not None,
        args.mm_fusion_rgb_mask_p is not None,
        args.mm_fusion_rgb_mask_epoch_start is not None,
        args.mm_fusion_rgb_mask_fill is not None,
        args.mm_fusion_rgb_mask_scope is not None,
        args.mm_fuse_ir_label_always is not None,
        args.mm_freeze_for_fusion_probe,
    ])
    if dual_args_used and not is_dual_head:
        print(f"[错误] 指定了 --mm-dual-* / --mm-dfs-* 参数，"
              f"但 --config 不是 dual-fusion / dual-fusion-b1（当前：{config_label}）。")
        print("       这些参数仅对双头 + 融合分支架构有效。")
        return 2

    # B1 专属参数落到非 B1 配置上：结构里根本没有 fuse_ir 头，给了也不会生效，
    # 属于典型的"以为调了其实没调"，必须显式拦截而不是静默忽略。
    if (args.mm_dual_w_fuse_ir is not None or args.mm_dual_w_fuse_ir_consist is not None) and not is_b1:
        print(f"[错误] 指定了 B1 专属参数（--mm-dual-w-fuse-ir / "
              f"--mm-dual-w-fuse-ir-consist），但 --config 不是 dual-fusion-b1"
              f"（当前：{config_label}）。")
        print("       这些参数仅在融合分支挂载 fuse_ir 头（-b1 配置）时才有意义。")
        return 2

    if is_dual_head:
        # 双头架构必须配双标签数据集，否则 IR 头无监督（数据层会静默降级）
        if args.label_from is None:
            print("[提示] 检测到 dual-fusion 架构，建议显式指定 --label-from dual，")
            print("       以确保 data.yaml 含双标签映射（labels/ + labels_ir/）。")
            print("       将自动按 dual 处理。")
            args.label_from = "dual"
        elif args.label_from in ("vis", "ir"):
            print(f"[警告] dual-fusion 搭配 --label-from {args.label_from}：")
            print("       此时两套标签同源，IR 头与 RGB 头学到的是同一坐标系，")
            print("       双头架构的核心价值（红外侧独立监督）将丧失。")
            print("       如确为消融对照可忽略，否则请改用 --label-from dual。")

    # ---- 输出根目录：YOLO26 实验统一落到 YOLO26/<project>，与 YOLO11 的 ResTest 隔离 ----
    if args.project:
        cfg["project"] = args.project

    # ---- 代际强制隔离（硬断言，不做 warning）----
    # 两代共用同一份 ultralytics 与 train_full，唯一的隔离手段就是这里的断言。
    # 写成 warning 会重演「参数传了但静默失效」的事故（见 multimodal/train.py 的教训），
    # 故一律 return 2 中止。
    _is_v26_cfg = config_label.startswith("v26-") or "/yolo26/" in str(cfg["model_yaml"])
    _is_v26_out = str(cfg["project"]).startswith("YOLO26")
    if _is_v26_cfg != _is_v26_out:
        print(f"[错误] 代际与输出目录不匹配：config={config_label}（v26={_is_v26_cfg}），"
              f"project={cfg['project']}（v26={_is_v26_out}）")
        print("       YOLO26 配置必须配 --project YOLO26/ResTest；YOLO11 配置必须配 ResTest。")
        print("       否则两代结果会混进同一目录，续训还会跨代读错 ckpt。")
        return 2
    print(f"[代际] {'YOLO26' if _is_v26_cfg else 'YOLO11'}  config={config_label}  "
          f"yaml={cfg['model_yaml']}  输出={cfg['project']}")

    # ---- 未显式指定 name 时自动命名（放在参数校验之后，确保后缀反映最终设置）----
    if args.name is None:
        if args.config:
            cfg["name"] = f"yolomm-{config_label}"
        # 单模态消融时追加后缀，避免与双模态结果混淆
        if args.single_modal:
            cfg["name"] = f"{cfg['name']}-{args.single_modal}"
        # 标签来源与 DFSC 状态也进名字，避免不同设置互相覆盖
        if args.label_from and args.label_from != "keep":
            cfg["name"] = f"{cfg['name']}-lbl{args.label_from}"
        if args.mm_dfs_weight is not None:
            cfg["name"] = f"{cfg['name']}-dfs{args.mm_dfs_weight}"
        # B1 双坐标系监督状态进名字：fuse_ir 头权重非默认时标注，避免与
        # dual-fusion 及其它 B1 权重设置互相覆盖。
        if is_b1 and args.mm_dual_w_fuse_ir is not None:
            cfg["name"] = f"{cfg['name']}-fir{args.mm_dual_w_fuse_ir}"
        # 路线二状态进名字：屏蔽概率与启动 epoch 都标注，避免与基线/其它 p 互相覆盖
        if args.mm_fusion_rgb_mask_p is not None:
            cfg["name"] = f"{cfg['name']}-frm{args.mm_fusion_rgb_mask_p:g}"
            if args.mm_fusion_rgb_mask_epoch_start:
                cfg["name"] = f"{cfg['name']}-e{args.mm_fusion_rgb_mask_epoch_start}"
            if args.mm_fusion_rgb_mask_fill and args.mm_fusion_rgb_mask_fill != "zero":
                cfg["name"] = f"{cfg['name']}-{args.mm_fusion_rgb_mask_fill}"
            if args.mm_fusion_rgb_mask_scope == "sample":
                cfg["name"] = f"{cfg['name']}-frs"
        # B3 蒸馏状态进名字，避免与纯屏蔽臂（S5）互相覆盖
        if args.mm_dual_w_fuse_ir_distill is not None and args.mm_dual_w_fuse_ir_distill > 0:
            cfg["name"] = f"{cfg['name']}-dist{args.mm_dual_w_fuse_ir_distill:g}"
        # 判别实验状态进名字
        if args.mm_freeze_for_fusion_probe:
            cfg["name"] = f"{cfg['name']}-frzprobe"
            if args.mm_freeze_probe_scope == "head":
                cfg["name"] = f"{cfg['name']}-hdonly"
        if args.mm_fuse_ir_label_always is not None and str(args.mm_fuse_ir_label_always).lower() == "true":
            cfg["name"] = f"{cfg['name']}-ircoord"

    print("=" * 72)
    print(f"完整训练 ｜ 路线 {args.route}：{ROUTE_HELP[args.route]}")
    print("=" * 72)
    for k in ("model_yaml", "data", "epochs", "batch", "imgsz", "close_mosaic",
              "workers", "cache", "amp", "name"):
        print(f"  {k:<14} = {cfg[k]}")
    print(f"  {'pretrained':<14} = {args.pretrained or '(无，随机初始化)'}")
    print(f"  {'device':<14} = {args.device}")
    print(f"  {'architecture':<14} = "
          f"{('双头+融合分支' + ('（B1 双坐标系监督）' if is_b1 else '')) if is_dual_head else '单头融合'}")
    if is_dual_head:
        _frm_p = args.mm_fusion_rgb_mask_p or 0.0
        _frm_extra = ""
        if _frm_p:
            _frm_fill = args.mm_fusion_rgb_mask_fill or "zero"
            _frm_extra = (f"（启动 epoch {args.mm_fusion_rgb_mask_epoch_start or 0}，"
                          f"fill={_frm_fill}）")
        print(f"  {'fusion_rgb_mask':<14} = {_frm_p}{_frm_extra}")

    check_route(args.route, args.pretrained)

    # ---- 标签来源改写（在自检之前，因为自检要读 data.yaml）----
    if args.label_from and args.label_from != "keep":
        if not apply_label_from(cfg["data"], args.label_from, dry_run=args.label_from_dry_run):
            print("\n[中止] 标签来源改写失败。")
            return 1

    # ---- 训练前自检 ----
    if args.preflight:
        print("\n" + "=" * 72)
        print("训练前自检")
        print("=" * 72)
        ok = run_preflight(cfg["data"])
        if not ok and not args.force:
            print("\n[中止] 自检未通过。修复后重试，或加 --force 强制继续。")
            return 1
        if not ok:
            print("\n[警告] 自检未通过，但 --force 已指定，继续训练。")

    # ---- 延迟导入，避免自检阶段被环境问题卡住 ----
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(",")[0])
    sys.path.insert(0, str(PROJ))

    from ultralytics import YOLOMM

    # 路线 A：从 YAML 构建（可用 scale 缩放网络宽度/深度）
    # 路线 B/C：以 YAML 构建结构，再加载 .pt 权重。
    #   注意：框架规定 `scale` 只能在"从 YAML 构建"时使用；
    #   若模型带 ckpt（从 .pt 载入），再传 scale 会直接抛
    #     ValueError: model_scale/scale can only be used when building a model from YAML.
    #   因此路线 B/C 下必须省略 scale —— .pt 里的结构已定型，
    #   缩放会造成权重与结构不匹配，报错本身是合理的保护。
    use_scale = args.route == "A"

    model_yaml = str(PROJ / cfg["model_yaml"]) if not Path(cfg["model_yaml"]).is_absolute() else cfg["model_yaml"]
    if not Path(model_yaml).exists():
        print(f"[错误] 模型配置不存在: {model_yaml}")
        return 1

    # ---- 路线 B/C：以 ckpt 的 nc 构建 wrapper ----
    # YOLOMM(yaml) 默认用 yaml 的 nc（通常 80）构建，而本项目多模态 ckpt 是 nc=1。
    # nc 不一致时 DetectionModel.load 按名字+形状取交集，四个检测头的 cv3 分类分支
    # （每头 51 键，含 model.56 内别名共 408 键）会因形状不匹配被静默跳过，
    # 分类分支随机初始化（实测冻结探针实验 rgb/ir 头 mAP 塌到 ~0.004，
    # 日志表现为 "Transferred 1544/1952"）。故加载前先按 ckpt 的 nc 重写临时 yaml。
    import torch
    import tempfile

    build_yaml = model_yaml
    if args.route in ("B", "C") and args.pretrained:
        _ck = torch.load(args.pretrained, map_location="cpu")
        _ck_nc = int(getattr(_ck.get("model"), "nc", 0) or 0) if isinstance(_ck, dict) else 0
        _nc = int(args.build_nc or _ck_nc)
        if args.build_nc:
            print(f"[nc 覆盖] --build-nc={args.build_nc}（ckpt nc={_ck_nc}，"
                  f"检测头分类分支将重新初始化）")
        if _nc:
            import yaml as _pyyaml

            with open(model_yaml, "r", encoding="utf-8") as f:
                _ycfg = _pyyaml.safe_load(f)
            if int(_ycfg.get("nc", -1)) != _nc:
                _ycfg["nc"] = _nc
                _tmp = tempfile.NamedTemporaryFile(
                    "w", suffix=".yaml", delete=False, encoding="utf-8", dir=str(PROJ)
                )
                _pyyaml.safe_dump(_ycfg, _tmp, allow_unicode=True, sort_keys=False)
                _tmp.close()
                build_yaml = _tmp.name
                print(
                    f"[nc 对齐] ckpt nc={_nc} != yaml nc={_ycfg['nc']}，"
                    f"已按 ckpt nc 构建模型（否则四个检测头的 cv3 分类分支将无法加载权重）"
                )

    model = YOLOMM(build_yaml)

    # ---- 路线 B / C：加载预训练权重 ----
    if args.route in ("B", "C") and args.pretrained:
        print(f"\n[加载权重] {args.pretrained}")
        model.load(args.pretrained)

    # ---- 组装训练参数 ----
    # resume 必须传 ckpt 路径而不是 True：Model.train() 内部会把 True 改写成
    # self.ckpt_path，而本脚本用 YOLOMM(yaml) 构建模型时 ckpt_path 为 None，
    # 于是 args["resume"] 变成 None，check_resume 的 if resume 分支不进入，
    # trainer 静默从头训练（实测日志 resume=None、进度 1/150）。
    resume_arg = None
    if args.resume:
        resume_arg = str(PROJ / cfg["project"] / cfg["name"] / "weights" / "last.pt")
        if not Path(resume_arg).is_file():
            print(f"[错误] --resume 指定的 ckpt 不存在: {resume_arg}")
            return 3
        print(f"\n[续训] 从 {resume_arg} 恢复（epoch/优化器/EMA/调度器一并恢复）")

    train_kwargs = dict(
        data=cfg["data"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        close_mosaic=cfg["close_mosaic"],
        patience=cfg["patience"],
        save_period=cfg["save_period"],
        amp=cfg["amp"],
        device=args.device,
        cache=cfg["cache"],
        workers=cfg["workers"],
        plots=cfg["plots"],
        seed=args.seed,
        resume=resume_arg,
        exist_ok=True,
        project=str(PROJ / cfg["project"]),
        name=cfg["name"],
        # 以下为可选消融开关，完整训练保持默认关闭
        # modality='rgb',      # 模态消融（非必要不开启）
        # single_modal='rgb',  # 单模态训练（需对应 yaml）
        # loss_cls='bce',      # 分类损失: bce | focal | efl | qfl
        # loss_box='ciou',     # 回归损失: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou
        # loss_dfl=True,       # DFL 开关
        # afss_enabled=True,   # AFSS 训练采样机制
    )
    if getattr(args, "lr0", None) is not None:
        train_kwargs["lr0"] = float(args.lr0)

    # 损失权重透传（box/dfl）：仅显式传入时覆盖，保持历史实验口径不变
    _loss_hyp = {}
    for _k in ("box", "dfl"):
        _v = getattr(args, _k, None)
        if _v is not None:
            _loss_hyp[_k] = float(_v)
    if _loss_hyp:
        train_kwargs.update(_loss_hyp)
        print(f"\n[损失权重] {_loss_hyp}（单变量覆盖默认 box=7.5 / dfl=1.5）")
    # 回归损失类型（字符串型 hyp，不能进 _loss_hyp 的 float 循环）
    if args.loss_box:
        train_kwargs["loss_box"] = args.loss_box
        print(f"\n[回归损失] loss_box='{args.loss_box}'（rgb/ir/fuse/fuse_ir 统一生效）")
        if args.loss_box == "nwd":
            print("  → NWD：以高斯 Wasserstein 距离衡量框相似度，缓解微小目标"
                  "几像素位移导致的 IoU 剧烈跳变")
            print("  ⚠️ NWD 量纲与 CiOU 不同，box 权重 7.5 未重新标定，首轮沿用默认值")
    # 跨分辨率特征蒸馏：显式传入时才启用
    if args.distill:
        _d_yaml, _d_mode = str(args.distill[0]), str(args.distill[1]).lower()
        train_kwargs["distill"] = [_d_yaml, _d_mode]
        print(f"\n[蒸馏] yaml={_d_yaml} mode={_d_mode}")
    # 几何增强：仅当显式传入时才覆盖，避免改变历史实验口径
    _geo = {}
    for _k in ("degrees", "shear", "perspective"):
        _v = getattr(args, _k, None)
        if _v is not None:
            _geo[_k] = float(_v)
    if _geo:
        train_kwargs.update(_geo)
        print(f"\n[几何增强] {_geo}（双模态同步，已验证不破坏对齐）")
    if args.safe_augment:
        train_kwargs.update(mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0)
        print("\n[安全增强模式] mosaic/mixup/cutmix/copy-paste 全部关闭")
    if use_scale and not args.resume:
        train_kwargs["scale"] = cfg["scale"]
    elif args.resume:
        # 续训时结构完全取自 ckpt，check_resume 会显式拒绝 model_scale/scale，
        # 必须省略，否则抛 "model_scale/scale is not supported with resume=True"
        print(f"\n[续训] 已自动省略 scale='{cfg['scale']}'（架构取自 ckpt）")
    else:
        print(f"\n[提示] 路线 {args.route} 使用 .pt 权重，已自动省略 scale='{cfg['scale']}'")
        print("       原因：scale 仅在使用 YAML 构建模型时有效；.pt 的网络结构已定型。")

    # 模态消融参数（仅在显式指定时传入，避免影响常规训练）
    # 续训时架构完全取自 ckpt，再传 single_modal 会与路由结构冲突并抛
    # "single_modal='rgb' can only be used with a plain single-modal YAML"
    if args.single_modal and not args.resume:
        train_kwargs["single_modal"] = args.single_modal
        print(f"\n[模态消融] single_modal='{args.single_modal}'（构建期单模态结构）")
        if args.single_modal == "x":
            print("           注意：'x' 对应红外模态。")
    elif args.single_modal:
        print(f"\n[续训] 已省略 single_modal='{args.single_modal}'（结构取自 ckpt）")
    if args.modality:
        train_kwargs["modality"] = args.modality
        print(f"[模态消融] modality='{args.modality}'（运行时消融）")

    # ---- 双头 + 融合分支超参透传（仅显式指定时传入）----
    dual_kwargs = {
        "mm_dual_w_rgb": args.mm_dual_w_rgb,
        "mm_dual_w_ir": args.mm_dual_w_ir,
        "mm_dual_w_fuse": args.mm_dual_w_fuse,
        "mm_dfs_weight": args.mm_dfs_weight,
        "mm_dfs_mode": args.mm_dfs_mode,
        "mm_dfs_temperature": args.mm_dfs_temperature,
        "mm_dual_consist_weight": args.mm_dual_consist_weight,
        # B1：融合分支 IR 坐标系头（仅 -b1 配置存在该头）
        "mm_dual_w_fuse_ir": args.mm_dual_w_fuse_ir,
        "mm_dual_w_fuse_ir_consist": args.mm_dual_w_fuse_ir_consist,
        # B3：drop-RGB 条件下的 fuse←ir 蒸馏
        "mm_dual_w_fuse_ir_distill": args.mm_dual_w_fuse_ir_distill,
        # P0：门控激活正则（仅 dual-fusion-gated 有意义）
        "mm_alpha_anchor_weight": args.mm_alpha_anchor_weight,
        "mm_alpha_min": args.mm_alpha_min,
        "mm_gate_reg_weight": args.mm_gate_reg_weight,
        "mm_gate_reg_target": args.mm_gate_reg_target,
        # B4-b：伪融合根因修复（mask 条件下 fuse 头改用 IR 标签）
        "mm_fuse_ir_label_on_mask": (
            None if args.mm_fuse_ir_label_on_mask is None
            else str(args.mm_fuse_ir_label_on_mask).lower() == "true"
        ),
        # 路线二：融合颈 RGB 半边随机屏蔽
        "mm_fusion_rgb_mask_p": args.mm_fusion_rgb_mask_p,
        "mm_fusion_rgb_mask_epoch_start": args.mm_fusion_rgb_mask_epoch_start,
        "mm_fusion_rgb_mask_fill": args.mm_fusion_rgb_mask_fill,
        "mm_fusion_rgb_mask_scope": args.mm_fusion_rgb_mask_scope,
        # D2-P1：通路级 dropout（逐样本强置融合算子 gate=1）
        "mm_gate_drop_p": args.mm_gate_drop_p,
        # D3-P2：强制融合算子输出全部来自 IR（判别实验）
        "mm_force_ir_fusion": (True if args.mm_force_ir_fusion else None),
        # 判别实验：fuse 头无条件用 IR 坐标系标签 + 冻结骨干与单模态头
        "mm_fuse_ir_label_always": (
            None if args.mm_fuse_ir_label_always is None
            else str(args.mm_fuse_ir_label_always).lower() == "true"
        ),
        "mm_freeze_for_fusion_probe": (
            True if args.mm_freeze_for_fusion_probe else None
        ),
        "mm_freeze_probe_scope": args.mm_freeze_probe_scope,
        # 标签分配相似度：iou（默认）| geoshape（COXNet 移植）
        "loss_assign_metric": args.loss_assign_metric,
    }
    passed = {k: v for k, v in dual_kwargs.items() if v is not None}
    if passed:
        train_kwargs.update(passed)
        print("\n[双头 + 融合分支] 超参：")
        for k, v in passed.items():
            print(f"  {k:<24} = {v}")
        if args.mm_dfs_weight is not None:
            if args.mm_dfs_weight > 0:
                print("  → DFSC 已开启；训练日志中的 last_dfs 应 > 0 且逐步下降")
            else:
                print("  → DFSC 已显式关闭（mm_dfs_weight=0）")

    # ---- 裁切放大（E19）----
    if args.mm_crop_zoom_p is not None:
        if not 0.0 <= args.mm_crop_zoom_p <= 1.0:
            print(f"[错误] --mm-crop-zoom 应在 [0, 1] 内，实得 {args.mm_crop_zoom_p}")
            return 1
        train_kwargs["mm_crop_zoom_p"] = args.mm_crop_zoom_p
        if args.mm_crop_zoom_side is not None:
            lo, hi = float(args.mm_crop_zoom_side[0]), float(args.mm_crop_zoom_side[1])
            if not (0.3 <= lo < hi <= 1.0):
                print(f"[错误] --mm-crop-zoom-side 应满足 0.3 <= MIN < MAX <= 1.0，实得 {lo} {hi}")
                return 1
            train_kwargs["mm_crop_zoom_side"] = [lo, hi]
        if args.mm_crop_zoom_p > 0:
            side = train_kwargs.get("mm_crop_zoom_side", [0.6, 0.85])
            print(f"\n[裁切放大] p={args.mm_crop_zoom_p} side={side}")
            print("  作用：训练期以 GT 为中心裁切再放大，提高极小目标的网格占位；推理仍为原尺寸")
            print("  ⚠️ 改变了训练期尺度/上下文分布，需在论文中声明为数据增强")

    # ---- 目标尺度重分布（DIV-B）----
    if args.mm_target_repaste_p is not None:
        if not 0.0 <= args.mm_target_repaste_p <= 1.0:
            print(f"[错误] --mm-target-repaste 应在 [0, 1] 内，实得 {args.mm_target_repaste_p}")
            return 1
        train_kwargs["mm_target_repaste_p"] = args.mm_target_repaste_p
        if args.mm_target_repaste_scale is not None:
            train_kwargs["mm_target_repaste_scale"] = list(args.mm_target_repaste_scale)
        if args.mm_target_repaste_ctx is not None:
            train_kwargs["mm_target_repaste_ctx"] = args.mm_target_repaste_ctx
        if args.mm_target_repaste_feather is not None:
            train_kwargs["mm_target_repaste_feather"] = args.mm_target_repaste_feather
        if args.mm_target_repaste_drop_src_p is not None:
            train_kwargs["mm_target_repaste_drop_src_p"] = args.mm_target_repaste_drop_src_p
        if args.mm_target_repaste_p > 0:
            sc = train_kwargs.get("mm_target_repaste_scale", [1.5, 4.0])
            print(f"\n[目标尺度重分布] p={args.mm_target_repaste_p} scale={sc}")
            print("  作用：把已有目标放大后贴回，补足 >=64px 大目标（原占比仅 1.2~2.3%）")
            print("  ⚠️ 本项引入合成目标，改变了训练数据分布，需在论文中声明为数据增强")

    # ---- 背景移植（DIV-C）----
    if args.mm_bg_swap_p is not None:
        if not 0.0 <= args.mm_bg_swap_p <= 1.0:
            print(f"[错误] --mm-bg-swap 应在 [0, 1] 内，实得 {args.mm_bg_swap_p}")
            return 1
        train_kwargs["mm_bg_swap_p"] = args.mm_bg_swap_p
        if args.mm_bg_bank is not None:
            train_kwargs["mm_bg_swap_bank"] = args.mm_bg_bank
        if args.mm_bg_ctx is not None:
            train_kwargs["mm_bg_swap_ctx"] = args.mm_bg_ctx
        if args.mm_bg_feather is not None:
            train_kwargs["mm_bg_swap_feather"] = args.mm_bg_feather
        if args.mm_bg_swap_p > 0:
            import os as _os
            bank = train_kwargs.get("mm_bg_swap_bank", "")
            print(f"\n[背景移植] p={args.mm_bg_swap_p} bank={bank}")
            if not bank or not _os.path.isdir(bank):
                print("[错误] 背景库目录不存在，请先运行 scripts/build_bg_bank.py")
                return 1
            print("  作用：用另一序列的无目标背景替换整幅背景，原 GT 前景回贴原位置")
            print("  ⚠️ 本项改变了训练数据分布，需在论文中声明为数据增强")

    # ---- 模态 dropout（B1-2：制造「单模态失效」训练样本）----
    # 默认值为 0.0（全关），正是 B1-1 在困难子集 A 组失效的直接原因：
    # 训练集里根本没有单模态失效帧，测试时自然无从泛化。
    if args.mm_modal_dropout_p is not None:
        if not 0.0 <= args.mm_modal_dropout_p < 1.0:
            print(f"[错误] --mm-modal-dropout 应在 [0, 1) 内，实得 {args.mm_modal_dropout_p}")
            return 1
        train_kwargs["mm_modal_dropout_p"] = args.mm_modal_dropout_p
        if args.mm_modal_dropout_p > 0:
            print(f"\n[模态 dropout] p={args.mm_modal_dropout_p}（已开启）")
            print("  作用：训练时随机遮挡单模态局部区域，迫使模型学会在模态失效时依赖另一路")
            print("  ⚠️ 本项改变了训练数据分布，与 B1-1 / A0-2 不再严格可比，需单独记录")
        else:
            print("\n[模态 dropout] p=0.0（显式关闭，与 B1-1 一致）")
    elif is_dual_head:
        print("\n[模态 dropout] 未指定 --mm-modal-dropout，使用默认 0.0（关闭）")
        print("  提示：困难子集全是单模态失效帧，若判据（A 组 fuse > 0.1）未达标，")
        print("        下一步应加 --mm-modal-dropout 0.3 重跑")

    # ---- 异步几何扰动（错位鲁棒训练）----
    # 依据 test 集错位退化曲线（mdp150_M0_concat_s0）：
    #   平移 1px -2.2% / 2px -6.0% / 4px -15.5% / 6px -23.2% / 8px -29.5%（融合增益转负）
    #   旋转 0.5° -4.7% / 1° -12.7% / 2° -24.3%
    # 结论：对齐残差 1px 即掉出 2σ 判定带，纯靠标定无法兜住，故在训练期注入错位样本。
    if args.mm_async_geo_p is not None:
        if not 0.0 <= args.mm_async_geo_p <= 1.0:
            print(f"[错误] --mm-async-geo-p 应在 [0, 1] 内，实得 {args.mm_async_geo_p}")
            return 1
        if args.mm_async_geo_p > 0:
            _tr = list(args.mm_async_geo_translate_px or [1, 2])
            _rot = list(args.mm_async_geo_rotate_deg or [-0.5, 0.5])
            if _tr[0] < 0 or _tr[1] < _tr[0]:
                print(f"[错误] --mm-async-geo-translate-px 需满足 0<=MIN<=MAX，实得 {_tr}")
                return 1
            if _rot[1] < _rot[0]:
                print(f"[错误] --mm-async-geo-rotate-deg 需满足 MIN<=MAX，实得 {_rot}")
                return 1
            train_kwargs["mm_async_geo_p"] = args.mm_async_geo_p
            train_kwargs["mm_async_geo_translate_px"] = _tr
            train_kwargs["mm_async_geo_rotate_deg"] = _rot
            if args.mm_async_geo_target is not None:
                train_kwargs["mm_async_geo_target"] = args.mm_async_geo_target
            if args.mm_async_geo_border is not None:
                train_kwargs["mm_async_geo_border"] = args.mm_async_geo_border
            print("\n[异步几何扰动] 已开启（单侧模态平移/旋转，模拟传感器错位）")
            print(f"  p={args.mm_async_geo_p}  translate_px={_tr}  rotate_deg={_rot}  "
                  f"target={args.mm_async_geo_target or 'x'}")
            print("  作用：让模型见过 1~4px 错位样本，把对齐容忍度从标定精度搬到数据分布")
            print("  ⚠️ 该扰动**故意破坏**两路像素对齐，data.yaml 必须 align: false；")
            print("     与全部 align:true 的历史实验不可直接横向比较（仅 fuse/IR 路受影响）")
        else:
            print("\n[异步几何扰动] p=0.0（显式关闭，与历史基线一致）")

    # ---- 路线二：融合颈 RGB 半边随机屏蔽 ----
    # mm_* 参数不在 cfg/__init__.py 的 CFG_*_KEYS 白名单内，不受 get_cfg 的类型/
    # 范围校验保护，故必须在此手动校验，否则非法值会一路带到训练里才暴露。
    if args.mm_fusion_rgb_mask_p is not None:
        if not 0.0 <= args.mm_fusion_rgb_mask_p <= 1.0:
            print(f"[错误] --mm-fusion-rgb-mask 应在 [0, 1] 内，实得 {args.mm_fusion_rgb_mask_p}")
            return 1
        if args.mm_fusion_rgb_mask_epoch_start is not None and args.mm_fusion_rgb_mask_epoch_start < 0:
            print(f"[错误] --mm-fusion-rgb-mask-epoch-start 应 >= 0，"
                  f"实得 {args.mm_fusion_rgb_mask_epoch_start}")
            return 1
        if (args.mm_fusion_rgb_mask_epoch_start is not None
                and args.mm_fusion_rgb_mask_epoch_start >= cfg["epochs"]):
            print(f"[警告] --mm-fusion-rgb-mask-epoch-start="
                  f"{args.mm_fusion_rgb_mask_epoch_start} 不小于总 epochs="
                  f"{cfg['epochs']}，屏蔽将永远不会生效。")
        if not is_dual_head:
            print(f"[错误] 指定了路线二参数，但 --config 不是 dual-fusion / dual-fusion-b1"
                  f"（当前：{config_label}）。")
            print("       融合颈 Concat 仅存在于双头 + 融合分支架构中。")
            return 2
        if args.mm_fusion_rgb_mask_p > 0:
            print(f"\n[融合颈 RGB 屏蔽] p={args.mm_fusion_rgb_mask_p}（已开启）")
            print("  依据：诊断发现融合头存在「伪融合」—— 屏蔽融合颈 RGB 半边后 fuse 损失")
            print("        归零而 rgb/ir 头不变，说明融合分支的 IR 半边是近零通路。")
            print("  作用：训练时按概率切断融合分支的 RGB 支路，迫使 IR 通路获得梯度。")
            print("  ⚠️ 本项改变了训练时融合分支的输入分布，与 B1-1 / A0-2 不再严格可比，")
            print("     需单独记录实验名（已自动附加 -frm 后缀）。")
            print(f"  填充方式：{args.mm_fusion_rgb_mask_fill or 'zero'}；"
                  f"启动 epoch：{args.mm_fusion_rgb_mask_epoch_start or 0}；"
                  f"抽样粒度：{args.mm_fusion_rgb_mask_scope or 'batch'}")
            if (args.mm_fusion_rgb_mask_scope or "batch") == "sample":
                print("  抽样粒度 sample：逐样本掷签，同一 batch 内混有屏蔽 / 未屏蔽帧，")
                print("                    融合头在同一次 forward 内必须同时支持 RGB 与 IR。")
                print("  ⚠️ 与 frm 系列（逐 batch）不可直接比较，实验名已附加 -frs 后缀。")
            print("  判据：若 fuse 头 loss 从基线 ~0.05 升到 ~0.1 量级且 rgb 头 loss 不变，")
            print("        说明屏蔽生效；反之应检查日志中的「命中融合 Concat 层」告警。")
        else:
            print("\n[融合颈 RGB 屏蔽] p=0.0（显式关闭，与基线一致）")
    elif is_dual_head:
        print("\n[融合颈 RGB 屏蔽] 未指定 --mm-fusion-rgb-mask，使用默认 0.0（关闭）")

    if args.mm_fusion_rgb_mask_scope is not None and not args.mm_fusion_rgb_mask_p:
        print("\n[融合颈 RGB 屏蔽] 指定了 --mm-fusion-rgb-mask-scope，"
              "但 --mm-fusion-rgb-mask 缺失或为 0，抽样粒度不会生效。")

    # B3 蒸馏依赖于「屏蔽确实发生」：没有屏蔽就没有 RGB 缺失条件，蒸馏永不激活。
    if args.mm_dual_w_fuse_ir_distill and not (args.mm_fusion_rgb_mask_p or 0.0):
        print("\n[错误] 指定了 --mm-dual-w-fuse-ir-distill，但未开启融合颈 RGB 屏蔽"
              "（--mm-fusion-rgb-mask）。")
        print("       B3 蒸馏只在 RGB 半边被屏蔽的 batch 上生效，二者必须同时使用。")
        return 2

    # P0：门控激活正则只在存在门控层（GatedReliabilityFusion / DeformableCrossModalFusion）
    # 的配置下有意义（不阻断，仅提示）。
    if (args.mm_gate_reg_weight or 0.0) > 0 and config_label not in GATED_CONFIGS:
        print(f"\n[警告] 指定了 --mm-gate-reg-weight，但 --config={config_label} 不含 "
              f"门控融合层，该项将是 no-op（不产生任何计算）。")
        print(f"       门控激活正则仅对 --config {' / '.join(GATED_CONFIGS)} 生效。")
    if args.mm_gate_reg_target is not None and (args.mm_gate_reg_target or 0.0) <= 0:
        print(f"\n[错误] --mm-gate-reg-target 应 > 0，实得 {args.mm_gate_reg_target}")
        return 2

    # ---- 判别实验：监督坐标系（fuse 头无条件 IR 标签 + 冻结骨干）----
    if args.mm_fuse_ir_label_always is not None and str(args.mm_fuse_ir_label_always).lower() == "true":
        if not is_b1:
            print("\n[警告] --mm-fuse-ir-label-always true 时，fuse 头改用 IR 坐标系标签监督；")
            print("       当前配置无 fuse_ir 头，融合分支将完全落在 IR 坐标系下。")
        print("\n[判别实验] fuse 头改用 IR 坐标系标签（无条件）：")
        print("  动机：伪融合已排除监督侧(S9_fix)与算子侧(C4)两类解释，")
        print("        剩余可疑环节为 fuse 头的监督坐标系 —— 它以 RGB 系标签监督，")
        print("        跨模态偏移使 IR 特征无法落准，梯度持续惩罚 IR 通路。")
        print("  验收：掩码探针 drop_rgb_term_scaled 下 fuse mAP 若显著回升（目标>=0.15），")
        print("        则瓶颈在监督坐标系；若仍塌到 ~0.01，则瓶颈在融合算子/梯度注入。")
    if args.mm_freeze_for_fusion_probe:
        print("\n[判别实验] 冻结骨干与 rgb/ir 头，仅训练融合颈（含融合算子）+ fuse 头。")
        print("  被冻结：RGB/IR 骨干、rgb 头、ir 头、fuse_ir 头（若存在）。")
        print("  训练量大幅下降，epochs 建议 20~30 即可观测结论。")
        print("  ⚠️ 本模式改变了可训练参数集合，与全量训练不可比，仅作机制判别。")

    if is_dual_head:
        if is_b1:
            print("\n[双头 + 融合分支 · B1] 四头权重均为 1.0，DFSC 关闭")
            print("  说明：第 4 头 fuse_ir 复用融合 neck 特征、以 IR 坐标系标签监督，")
            print("        使融合特征在两个坐标系下都必须独立成立。")
            print("  ⚠️ loss_items 由 9 项（3 分支 × 3）扩为 12 项（4 分支 × 3），")
            print("     下游按固定长度解析 val loss 列名 / results.csv 的代码需同步。")
        else:
            print("\n[双头 + 融合分支] 三头权重均为 1.0，DFSC 关闭")
            print("  提示：建议至少跑一次 --mm-dfs-weight 0.5 以评估 DFSC 贡献")

    # ---- 启动训练 ----
    print("\n" + "=" * 72)
    print("开始训练")
    print("=" * 72)
    model.train(**train_kwargs)

    out_dir = PROJ / cfg["project"] / cfg["name"]
    print("\n" + "=" * 72)
    print("训练完成")
    print("=" * 72)
    print(f"输出目录: {out_dir}")
    print(f"  最优权重: {out_dir / 'weights' / 'best.pt'}")
    print(f"  最后权重: {out_dir / 'weights' / 'last.pt'}")
    print(f"  指标曲线: {out_dir / 'results.csv'}")

    if is_dual_head:
        _n_branch = 4 if is_b1 else 3
        print(f"\n分模态指标（双头架构，一次推理 {_n_branch} 套 mAP）：")
        print("  metrics/rgb/metrics/mAP50-95(B)   ← rgb 头 vs RGB 标签")
        print("  metrics/ir/metrics/mAP50-95(B)    ← ir  头 vs IR  标签")
        print("  metrics/fuse/metrics/mAP50-95(B)  ← fuse 头 vs RGB 标签（论文主指标）")
        if is_b1:
            print("  metrics/fuse_ir/metrics/mAP50-95(B) ← fuse_ir 头 vs IR 标签（B1 新增）")
        print("  metrics/fusion_gain_over_rgb      ← 融合相对 RGB 的增益（应 > 0）")
        print("  metrics/fusion_gain_over_ir       ← 融合相对 IR  的增益（应 > 0）")
        print("\n  ⚠️ 报数注意：ir 头用 IR 坐标系标签（infrared.json），")
        print("     与 rgb/fuse 头的 RGB 坐标系标签不同源，三路 mAP 不可跨模态直接比较。")
        print("\n  自检要点（从训练日志确认）：")
        _branches = "[rgb, ir, fuse, fuse_ir]" if is_b1 else "[rgb, ir, fuse]"
        print(f"    - 出现 'DualHeadDetectionLoss: nc=.. branches={_branches}'")
        if is_b1:
            print("    - loss_items/loss_names 长度应为 12（4 分支 × 3），不是 9")
            print("    - fuse_ir 头必须拿到 bboxes_ir 监督；若其 loss 恒为 0，")
            print("      说明 data.yaml 缺 labels_ir 映射或标签目录为空")
        print("    - 出现 'IR labels: N images, ... dir=labels_ir'（双标签生效）")
        print("    - 若 dual_labels_configured=False，说明 data.yaml 缺 labels 映射，")
        print("      请用 --label-from dual 重新启动。")
    else:
        print("\n下一步（可选）：验证新模型下的跨模态配准是否依然达标")
        print("  1) 把 scripts/calib/run_fixed_predict.py 里的 WEIGHTS 改成上面的 best.pt")
        print("  2) python scripts/calib/run_fixed_predict.py --reg --mode affine --tag _full")
        print("  3) python scripts/calib/verify_align.py")
        print("\n  若框住率不达标，说明标定量里的模型偏置已变，需用新模型重新标定。")

    print("\n若要复现本次实验，等价命令：")
    _repro = ["python train_full.py", f"--route {args.route}"]
    if args.config:
        _repro.append(f"--config {args.config}")
    if args.pretrained:
        _repro.append(f"--pretrained {args.pretrained}")
    if args.single_modal:
        _repro.append(f"--single-modal {args.single_modal}")
    if args.modality:
        _repro.append(f"--modality {args.modality}")
    if args.label_from and args.label_from != "keep":
        _repro.append(f"--label-from {args.label_from}")
    for _k, _v in passed.items():
        _repro.append(f"--{_k.replace('_', '-')} {_v}")
    if args.mm_freeze_for_fusion_probe:
        _repro.append("--mm-freeze-for-fusion-probe")
        if args.mm_freeze_probe_scope:
            _repro.append(f"--mm-freeze-probe-scope {args.mm_freeze_probe_scope}")
    _repro.append(f"--name {cfg['name']}")
    print("  " + " ".join(_repro))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
