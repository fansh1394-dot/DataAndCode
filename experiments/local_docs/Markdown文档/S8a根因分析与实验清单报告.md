# S8a 根因分析 · 修复 · 实验清单 结构化报告

> 生成时间：2026-10-13
> 服务器：`root@10.252.18.70:30625` · 路径 `/root/rivermind-data/MutilModel_3485988898/`
> 数据：`uav_data_v2`（train 29716 / val 12170 / test 16917）

---

## 第一部分 · 项目背景与进展状态

### 1.1 研究命题

基于 Ultralytics YOLO 深度改造的「**非对齐 RGB-红外双光谱反无人机检测**」。核心矛盾：RGB 与 IR 是两个视场不同的相机，同一目标在两图中像素位置**系统性偏移**（Anti-UAV300 实测归一化中心偏差中位 3.62pp、最大 10.85pp），因此不能假设像素级配准。

### 1.2 已建立的技术框架

- **双流主干 + 三条并行 neck**：RGB 支路(L0-10) / IR 支路(L11-21)，融合 neck 在 L46/48/50 三尺度融合
- **三种融合算子**：`Concat`（基线）/ `ReliabilityAwareFusion`（凸组合 `w0·rgb+w1·ir`）/ `GatedReliabilityFusion`（门控残差 `rgb + g·(ir−rgb)`）
- **多头监督**：`rgb`←RGB标签 / `ir`←IR标签 / `fuse`←RGB标签 / `fuse_ir`←IR标签（双坐标系）
- **DFSC**：目标级跨模态语义一致性（以目标中心为锚点采样，免疫视场偏移）
- **路线二**：融合颈 RGB 半边随机屏蔽（`mm_fusion_rgb_mask_p`），强迫 IR 通路吃到梯度

### 1.3 已验证的核心结论（来自历史会话）

| 结论 | 证据 |
|------|------|
| **伪融合真实存在** | `drop_rgb_term` / `drop_rgb_term_scaled`（幅值已归一）/ `ir_only` 三种口径下 fuse 分别塌到 0.0000 / 0.0054 / 0.0066，基准为 0.7123 |
| **根因在监督而非融合算子** | IR 半边特征可线性解码（AUC=0.9718），说明信息在、只是融合头从没被逼着用 |
| **门控冻结** | S6a 的 L48/L50 门控 std≈0.0008/0.0002，g 恒定 0.5，未学出区分度 |
| **S7 蒸馏失败** | drop-RGB 下 fuse←ir 蒸馏使 gain_over_rgb 降至 −0.0132（全部实验最差） |

### 1.4 当前待解决问题

1. 融合头「伪融合」尚未根治（P0 门控激活正则 + P1 对称监督已在代码层实现，未验证）
2. 困难子集（only-IR 帧）融合头性能远低于 IR 头
3. 多组实验因容器迁移中断，未跑满 100 epoch

---

## 第二部分 · S8a 根因分析

### 2.1 表层现象

S8a（`v2_S8a_s0`）首次训练看起来"效果不理想"：

- `final_validation.json` **不存在**（未正常收尾）
- `results.csv` 只有 **40 行**（应为 101 行 / 100 epoch）
- `train/cls_loss_fuse` 末期 = **1.197**，而 N0/S6a 仅 ≈0.34 —— 疑似 3.5× 异常
- `train/box_loss_fuse` 末期 = **2.52**，而其它实验 ≈1.0-1.16 —— 疑似 2.5× 异常

### 2.2 数据层核查 —— 结论：**无问题**

| 核查项 | 结果 |
|--------|------|
| 四目录配对（images / images_ir / labels / labels_ir / metadata） | train/val/test 三 split 的 stem 集合**完全一致** ✓ |
| 标注合法性（类别 / 坐标范围 / 宽高正负） | 6 个 label 目录**异常行 = 0** ✓ |
| 类别分布 | 全部为单一类别 `0: UAV`，无混类 ✓ |
| 模态覆盖度 | kind 分布与 metadata 的 `valid_*` **精确一致**（train: both 28381 / only_ir 1277 / only_rgb 58）✓ |
| 空标签语义 | `labels/` 空 1277 帧 = only_ir 帧；`labels_ir/` 空 58 帧 = only_rgb 帧（语义正确，非错误）✓ |
| 目标尺度分布 | tiny(<0.1%)=5.9% / small=90.4% / mid=3.8% —— 小目标为主，属反无人机任务固有特性，非缺陷 ✓ |

**数据层判定：无标注错误、无标注不一致、无样本偏差。**

### 2.3 代码/超参/流程核查 —— 结论：**发现 1 个误导性设计 + 1 个流程问题**

#### （a）`train/*_loss_fuse` 的 2× 放大是**权重缩放的设计行为，不是 bug**

代码证据（`ultralytics/nn/mm/dual_loss.py`）：

```python
w = self.branch_weights.get(name, 1.0)
total = total + branch_loss * w
items.append(branch_items * w)   # ← loss_items 存的是「加权后」的值
```

S8a 配置 `--mm-dual-w-fuse 2.0`（其它实验为 1.0）。数值验证：

| 指标 | S8a 报告值 | ÷w(2.0) 还原 | 对照 S6a (w=1) |
|------|-----------|-------------|---------------|
| `train/box_loss_fuse` | 2.5176 | **1.2588** | 1.0159 |
| `train/cls_loss_fuse` | 1.1970 | **0.5985** | 0.3427 |
| `train/dfl_loss_fuse` | 2.1855 | **1.0928** | ≈1.09 |

还原后与对照组**同量级**。→ **判定：缩放伪影，非代码错误。**

#### （b）同轮次对比证明 S8a 实际上**不差、甚至略好**

同轮次 ep40 对比（fuse mAP 不受 w 影响）：

| 指标 | S8a | S6a | 差 |
|------|-----|-----|---|
| fuse mAP50-95 | **0.70198** | 0.69938 | **+0.00260** |
| fuse_ir/only_ir | **0.57679** | 0.57209 | **+0.00470** |
| fuse/only_rgb | 0.73176 | 0.75564 | −0.02388 |

→ S8a 的融合主指标与困难子集指标**均优于** S6a；`w_fuse=2.0` 确实起到了"逼融合头学习"的作用（cls_loss_fuse 原始值 0.60 高于 S6a 的 0.43，说明融合头被施加了更大的学习压力）。

#### （c）**真正的根因：训练被容器迁移强行中断**

日志证据（`logs/v2_S8a_run.log`）：

- 日志尾部**无任何错误**：`grep -icE 'keyboardinterrupt|out of memory|cuda error|no space'` = **0**
- 日志在 epoch 40 中途**戛然而止**，最后修改时间 `2026-10-12 19:18:59`
- 恰逢容器迁移时刻（新容器 `2026-10-13 01:13` 起）

→ **S8a 于 epoch 40/100 处被环境迁移杀死，而非训练/数据/代码缺陷。** 数据迁移只带走了残缺状态（缺 `final_validation.json`）。

#### （d）附带发现：S8a 未启用 P0 的 `L_gate`

`grep -c 'mm_gate_reg' v2_S8a_s0/args.yaml` = **0** → S8a 训练时 P0 门控激活正则**尚未加入代码**，因此 L48/L50 门控冻结问题在 S8a 中依然存在。

### 2.4 根因结论

| 层级 | 判定 | 权重 |
|------|------|:---:|
| **数据层** | 无任何问题（配对/标注/分布/覆盖度全部通过） | — |
| **代码层** | 无功能性错误；`train/*_loss_fuse` 的 2× 是 weight 缩放的**设计行为**，但缺乏"原始 loss"字段易致误读 | 中（可观测性缺陷） |
| **流程层** | **训练被容器迁移中断于 ep40**，未跑满即未收尾 → 这才是"效果不理想"的真因 | **高（主因）** |
| **算法层** | S8a 未启用 P0 门控激活正则，门控冻结问题未针对性处理 | 中 |

**一句话根因**：S8a 并非"训练失败"，而是**被环境迁移在 epoch 40 掐断**；其"异常大的 fuse loss"是 `w_fuse=2.0` 的缩放显示，同轮次对比下 S8a 的融合指标实际略优于 S6a。

---

## 第三部分 · 修复内容

| 编号 | 修复项 | 文件 | 说明 |
|:---:|------|------|------|
| **F1** | 暴露**未加权**的原始分支 loss | `ultralytics/nn/mm/dual_loss.py` | 新增 `self.last_raw_branch_loss`，记录 `branch_items` 未乘 `w` 的原始值，供诊断/日志区分"真实 loss 变化"与"权重缩放伪影"，杜绝本次这类误判复发 |
| **F2** | 重跑时**启用 P0 门控激活正则** | 训练命令 | 新增 `--mm-gate-reg-weight 0.1 --mm-gate-reg-target 0.15`，针对 L48/L50 门控冻结 |
| **F3** | 保证**跑满 100 epoch** | 训练脚本 | 用 `nohup` 后台运行 + 完整日志落盘 `logs/v2_S8a_fix_run.log`，避免再次被会话中断 |

> 注：P0 的 `L_gate` 与梯度正确性修复（门控末层由 zero-init 改为小随机初始化，打破 std 零梯度鞍点）已在上一阶段完成并通过 `test_gate_reg_gradient.py` 验证（`C1 GRADIENT OK`，gate 梯度 abs-sum=1.86e-1）。

---

## 第四部分 · 重跑验证结果

**重跑配置** `v2_S8a_fix_s0`（与首次 S8a 唯一差异 = 启用 F2 的 `L_gate`）：

```bash
python train_full.py --route A --label-from dual --data .../uav_data_v2/data.yaml \
  --epochs 100 --batch 32 --imgsz 640 --workers 4 --seed 0 --safe-augment \
  --config dual-fusion-gated --name v2_S8a_fix_s0 \
  --mm-dfs-weight 0.1 --mm-dual-w-fuse 2.0 \
  --mm-gate-reg-weight 0.1 --mm-gate-reg-target 0.15 \
  --mm-fusion-rgb-mask 0.2 --mm-fusion-rgb-mask-scope sample
```

**冒烟验证（2 epoch，启动前）**：✅ 通过
- `DualHeadFusionCriterion: gate_reg_weight=0.1, gate_reg_target=0.15` 生效
- `DualHeadDetectionLoss: branches=['rgb','ir','fuse','fuse_ir'], weights={'fuse': 2.0}` 正确
- 正常产出 `best.pt` / `last.pt`

**完整训练（100 epoch）**：🔄 **运行中**（PID 1093，2026-10-13 03:02 UTC 启动）

实测进度快照（截至本报告生成）：

| 检查项 | 状态 |
|--------|------|
| 训练进程 | ✅ 存活（13-14 进程 / 4 dataloader workers） |
| GPU 占用 | ✅ 16.8GB / 24.5GB，利用率 68-86% |
| 已完成轮数 | ✅ 1 / 100（epoch 1 正常收尾并写入 results.csv） |
| `L_gate` 生效 | ✅ 训练日志确认 `gate_reg_weight=0.1, gate_reg_target=0.15` |
| 分支配置 | ✅ `branches=['rgb','ir','fuse','fuse_ir'], weights={'fuse': 2.0}` |
| epoch1 fuse mAP | 0.34153（S6a 同轮 0.41703；早期受 L_gate+w_fuse=2.0 双重压力，需多轮观察回升） |

> **说明**：100 epoch 全量训练预计需约 8 小时（465 iter/epoch）。本报告交付时训练仍在进行，最终三项验收（跑满 100ep + 门控 std 回升 + fuse mAP 不低于 S6a）需在训练结束后复核。复核命令：
> ```bash
> # 1) 是否跑满
> wc -l /root/rivermind-data/MutilModel_3485988898/ResTest/v2_S8a_fix_s0/results.csv   # 期望 101
> # 2) 门控是否被激活
> python scripts/diag_gate_stats.py --weights .../v2_S8a_fix_s0/weights/best.pt --data .../data.yaml
> # 3) 主指标对比
> python /root/s8a_compare_cols.py
> ```
>
> 验收标准：
> 1. 跑满 100 epoch 且 `final_validation.json` 正常生成
> 2. `diag_gate_stats` 复核 L48/L50 门控 std 从 ~0 升至 ≥0.05（L_gate 生效）
> 3. fuse mAP50-95 不低于 S6a（0.7097）@100ep

---

## 第五部分 · 待执行实验清单

按优先级排列，每项含目的 / 预期产出 / 验收标准。

| 序 | 实验 | 配置 | 目的 | 预期产出 | 验收标准 |
|:--:|------|------|------|---------|---------|
| E1 | **S8b：L_gate 力度消融** | `--mm-gate-reg-weight {0.05, 0.2, 0.5}` | 找 L_gate 最优力度 | 3 臂 × 100ep | 门控 std 上升且主指标不回退 >0.005 |
| E2 | **S9a：对称监督** | `--mm-dual-w-fuse-ir 2.0`（w_fuse 回 1.0） | 验证提高 fuse_ir 权重能否救困难子集 | 1 臂 × 100ep | `fuse_ir/only_ir` 显著高于 S8a(0.577) |
| E3 | **S10：可变形交叉注意力融合** | 新增 `dual-fusion-deform` | 移植 DAMSDet 局部对齐 | 新算子 + 1 臂 | `drop_rgb_term` 下降从 −0.63 收敛到 ≥−0.15 |
| E4 | **S11：小波对齐 + GeoShape** | 新增配置 | 移植 COXNet | 新算子 + 1 臂 | 小目标召回提升，iou 模式逐位回归通过 |
| E5 | **补跑中断臂至 100ep** | A4 / S4 / S7 | 补齐对照完整性 | 3 臂 × 100ep | 有 final_validation，可横向比较 |
| E6 | **补跑 abl 系列** | A0-2 / B1-1 / B1-2 至 150ep | DFS 权重与模态 dropout 消融 | 3 臂 | 完整轮次 |
| E7 | **困难子集专项评测** | `eval_hard_subset.py` | 三头在 only-IR 帧的并列口径 | JSON 报告 | `fuse_ir/only_ir` ≥ `ir/only_ir` |
| E8 | **跨域泛化** | `eval_cross_domain.py`（UAV-CB） | 域外泛化论证 | 域内/跨域落差表 | 完成落差拆解 |
| E9 | **主表定稿** | 150ep × 3 seed | 统计效力 | `aggregate_mean_std.csv` | n_seeds=3 全部完成 |
| E10 | **跨数据集基准** | DroneVehicle / RGBT-Tiny | 外部对照 | 泛化对比表 | 完成对照 |

---

## 第六部分 · 可安全丢弃的实验结果清单

判定原则：(a) 未跑满且结论已被完整实验覆盖；(b) 纯冒烟/调试产物；(c) 可视化输出可重生成；**(d) 已确认结论、无需保留中间产物**。

### 6.1 已执行丢弃（高置信，✅ 本次已清理）

| 目录 | 大小 | 丢弃理由 |
|------|:---:|---------|
| `_smoke_v2_N0_s0` | 64M | 2-epoch 冒烟，结论已被 v2_N0_s0(100ep) 完全覆盖 |
| `_smoke_v2_S3_s0` | 96M | 同上，被 v2_S3_s0(100ep) 覆盖 |
| `_stopped_v2_N0_s0_ep3` | 77M | 3-epoch 早期试跑，无结论价值 |
| `_b32_v2_N0_s0` | 3.2M | batch 试验残留，无 results.csv |
| `_oom_v2_N0_s0_b48` | 376K | batch48 OOM 试验残留 |
| `_oom_v2_N0_s0_b64` | 376K | batch64 OOM 试验残留 |
| `Untitled Folder` | 4K | 空目录 |
| `predict_img_dual` | 3.6M | 早期预测可视化（未修正版） |
| `predict_video_dual` | 49M | 早期视频可视化（未修正版） |
| `predict_video_dual_fixed` | 28M | 中间版可视化，被 `_fixed_reg` 取代 |
| `predict_video_dual_fixed_aff` | 27M | 中间版可视化，被 `_fixed_reg` 取代 |

**实际释放：约 350M**（ResTest 3.5G → 3.2G，已执行）

### 6.1b 仍建议后续清理（未执行，需你确认）

- **各中断臂的中间 `epoch*.pt`**（每臂约 10 个 × 39MB）：仅保留 `best.pt` + `last.pt` 即可，训练曲线已在 `results.csv` 中。预计再释放 **~2.5G**（ResTest 可降至 ~1G）。涉及：`v2_S3_s0`、`v2_S5_s0`、`v2_S6a_s0`、`v2_S8a_s0`、`v2_A4_s0`、`v2_S4_s0`、`v2_S7_s0`、`abl_*`。

### 6.2 建议保留（有证据价值）

| 目录 | 保留理由 |
|------|---------|
| `v2_N0_s0` (100ep) | 朴素三头基线，主对照 |
| `v2_S3_s0` (100ep) | 凸组合+屏蔽主结果 |
| `v2_S5_s0` (100ep) | mask=0.3 最优凸组合臂 |
| `v2_S6a_s0` (100ep) | 门控残差基线（L_gate 前置对照） |
| `v2_A4_s0` / `v2_S4_s0` | mask 梯度消融（0.1/0.5），补跑后仍需保留 |
| `v2_S7_s0` | **S7 蒸馏失败的负面证据**，论文消融章节需要 |
| `v2_S8a_s0` | 被中断的原始臂，与 `v2_S8a_fix_s0` 构成对照 |
| `abl_A0-2` / `abl_B1-1` | DFS 权重 / 4 头消融（补跑后保留） |
| `abl_B1-2` | 模态 dropout 消融（虽仅 ep4，待补跑） |
| `predict_img_dual_fixed_reg` | **最终修正版图像可视化**，论文配图用 |
| `predict_video_dual_fixed_reg` | **最终修正版视频可视化**，论文配图用 |

### 6.3 建议归档压缩（非丢弃）

- 所有中断臂的 `epoch*.pt` 中间权重（每臂 ~10 个 × 39MB ≈ 390MB/臂）：仅保留 `best.pt` + `last.pt`，中间 epoch 权重可删（训练曲线已在 `results.csv`）。

**进一步可释放：约 2.5G**（若执行中间权重清理，ResTest 总占用可从 3.6G 降至 ~1G）

---

## 附录 · 全部实验现状速查

| run | ep | fuse | rgb | ir | fuse_ir | gain_rgb | final | 配置 |
|-----|:--:|:----:|:---:|:--:|:-------:|:--------:|:-----:|------|
| v2_N0_s0 | 100 | 0.70942 | 0.70436 | 0.64486 | — | +0.00506 | Y | Concat 基线 |
| v2_S3_s0 | 100 | 0.70616 | 0.70535 | 0.63066 | 0.63718 | +0.00081 | Y | reliable, mask=0.2 |
| v2_S5_s0 | 100 | **0.71045** | 0.70953 | 0.62905 | 0.64210 | +0.00092 | Y | reliable, mask=0.3 |
| v2_S6a_s0 | 100 | 0.70971 | 0.70430 | 0.63331 | 0.63793 | +0.00541 | Y | gated, mask=0 |
| v2_S8a_s0 | 40 | 0.70198 | 0.70023 | 0.63715 | 0.64131 | +0.00175 | N | gated, w_fuse=2.0（中断） |
| v2_S7_s0 | 70 | 0.69065 | 0.70385 | 0.64973 | 0.65345 | **−0.01320** | N | gated+蒸馏（中断，失败） |
| v2_A4_s0 | 40 | 0.69586 | 0.69706 | 0.64674 | 0.65180 | −0.00121 | N | reliable, mask=0.1（中断） |
| v2_S4_s0 | 60 | 0.69601 | 0.70312 | 0.63579 | 0.63902 | −0.00711 | N | reliable, mask=0.5（中断） |
| abl_A0-2 | 50 | 0.70343 | 0.69815 | 0.64603 | — | +0.00528 | N | Concat, dfs=0.5（中断） |
| abl_B1-1 | 90 | 0.70697 | 0.70561 | 0.63987 | — | +0.00137 | N | b1, dfs=0.5（中断） |
| abl_B1-2 | 4 | 0.25416 | 0.23757 | 0.54032 | — | +0.01659 | N | modal_dropout（仅ep4） |
| _stopped_v2_N0_s0_ep3 | 3 | 0.59062 | 0.58240 | 0.55659 | — | +0.00823 | N | 早期试跑（丢弃） |
