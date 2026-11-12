# 配准工作归档

> 本目录保存"双光谱跨模态配准"专题的全部工作痕迹，便于后期回溯与重训后复用。
> 归档时间：2026-09-19

---

## 一、当前生效方案（主线）

**结论：使用"模型框中心 → 红外热斑"自动标定，框住率 98.0%，已验收达标。**

| 项目 | 值 |
|---|---|
| 标定文件 | `D:\uav_data\_vis\calib.npz` |
| 标定来源 | 模型推理输出（RGB 框中心）→ 红外热斑质心 |
| 样本数 | 98 帧视频抽帧 |
| 框住率 | **98.0%**（原始未修 35.4%） |
| 中心偏移 | **(-0.44, -1.74) pp**（目标 ≤ 2 pp） |
| 泛化性 | 5 折交叉验证同为 98.0% |

### 三层修复对应关系

| 层级 | 问题 | 修复位置 |
|---|---|---|
| 根因 | IR 画幅被拉伸 + 坐标系错配 | `ultralytics/data/multimodal/utils.py`、`inference_dataset.py`、`engine/multimodal/results.py` |
| 残差 | 两镜头视场不完全一致（含尺度差） | `ultralytics/engine/multimodal/registration.py` + 标定量 |
| 验证 | — | `scripts/calib/verify_align.py` |

详细改动清单见 `MutilModel_3485988898/docs/CHANGELOG_配准修复.md`。

### 关键脚本（主线，保留在项目内）

| 脚本 | 作用 |
|---|---|
| `scripts/calib/run_fixed_predict.py` | 重跑推理生成结果目录；`--reg --mode affine` 启用配准 |
| `scripts/calib/verify_align.py` | 量化验收（框住率 / 偏移 / 框尺寸四方案对比） |
| `train_full.py` | 完整训练配置（`--route A/B/C`） |
| `tools/preflight_train.py` | 训练前自检（权重盘点 + 数据配对校验） |

---

## 二、为什么放弃人工标点（重要背景）

曾尝试用**人工标注对应点**做"纯几何标定"，以摆脱标定量对模型的依赖。**结论：这条路在本数据集上不可行。**

### 失败原因（已量化验证）

| 验证项 | 结果 | 说明 |
|---|---|---|
| 图像清晰度 | 拉普拉斯方差中位 798（RGB）/ 565（IR），阈值 100 | **不是模糊问题**，排除"图太糊"的猜想 |
| RGB 亮度分布 | 11601 对验证图中，**3803 对亮度 < 20（几乎纯黑）**，占比约 35% | 大量夜间/过暗图，人眼无法定位目标 |
| 自动提取热斑 | 交叉验证残差 **130 px** | "最亮红外区"与"最暗可见光区"常不是同一目标，自动法更差 |
| 人工标注质量 | 内点残差中位 **13.55 px**（模型标定仅 1.87 px） | 人眼在几十像素目标上无法可靠定位 |

**根本矛盾**：人工标点要求"同一物理点在两模态都能被眼睛精确定位"。而本数据集的目标（无人机）在两模态中往往仅占几十像素，且红外为模糊热斑。**这不是标注技巧问题，是数据特性决定的。**

### 归档的失败产物

| 文件 | 内容 |
|---|---|
| `annotation_attempt/pairs.backup48_20260918_220345.json` | 48 对人工标注原始数据（91 点） |
| `annotation_attempt/pairs.backup48_20260918_220345.csv` | 同上，点坐标表格 |
| `annotation_attempt/pairs.backup48_20260918_220345.meta.txt` | 元信息与质检结论 |

**用途**：仅作历史记录与对照，**不作为标定输入**。

---

## 三、重训模型后如何重新标定（自动，无需人工）

模型重训后，标定量中的"模型偏置"会变，**必须重跑一次自动标定**。

### 步骤（全自动，约 3 分钟）

```powershell
cd 'C:\Users\不如喝茶去\Desktop\双光谱毕设\MutilModel_3485988898'

# 1) 把 run_fixed_predict.py 里的 WEIGHTS 改成新训练产出的 best.pt
#    路径形如 ResTest/yolomm-c3k2-pkimodule-full/weights/best.pt

# 2) 推理（不启用配准，此时框在 RGB 画幅）
python scripts\calib\run_fixed_predict.py

# 3) 用新推理结果重新拟合标定 → 覆盖 calib.npz
#    （拟合会读取新的框位置，自动更新仿射系数）

# 4) 验收，看框住率是否仍 ≥ 95%
python scripts\calib\run_fixed_predict.py --reg --mode affine --tag _new
python scripts\calib\verify_align.py
```

### 注意

- 拟合脚本会**自动备份**旧 `calib.npz`（形如 `calib.bak_<时间戳>.npz`），可随时回退
- 若新模型框住率显著低于 98%，先查模型本身精度（mAP），再查配准
- **切莫在模型大幅变化后沿用旧标定**，会引入系统性偏移

---

## 四、诊断脚本（本目录 diagnostics/）

| 文件 | 作用 |
|---|---|
| `t_reg_unit_test.py` | `registration.py` 的 12 项单元自检（映射正确性、安全降级、越界裁剪等） |
| `eval_alternative_methods.py` | 替代方案可行性评估（自动热斑提取 vs 人工标点，量化对比） |
| `probe_offset_result.txt` | 早期两模态偏移探针输出 |

运行方式（需在项目根目录）：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\anaconda\envs\pytorch-2.11.0-cpu\python.exe' '配准工作归档\diagnostics\t_reg_unit_test.py'
```

> 注：`t_reg_unit_test.py` 里的路径为归档前的位置，若需运行请把 `sys.path` 指向
> `MutilModel_3485988898`，并把 `calib.npz` 路径确认为当前实际路径。

---

## 五、后续可选的改进方向

1. **重训后自动标定**：见第三节，已固化为脚本流程
2. **若将来需要真正的"纯几何标定"**：建议在**近处放置明显发热的标定物**（热水杯、电烙铁等），
   使其在两模态中占数百像素，此时人工标点精度足够。这是标准做法——用专门标定物，
   而非拿作业目标凑合。
3. **端侧部署**：沿用同一对相机时，`calib.npz` 可直接复用；更换相机/载荷维修后需重标。

---

## 六、一句话总结

**主线用模型自动标定（98.0%，已验证）；人工标定因数据特性不可行，已归档留档；重训模型后按第三节自动重标。**
