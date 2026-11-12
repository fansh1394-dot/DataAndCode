# 双光谱双头融合模型 · Kaggle 部署运行方案

**目标**：在 Kaggle（GPU T4×2 / P100）上跑通 `--config dual-fusion` 的双光谱双头融合训练，
产出可下载的权重与三路 mAP 指标。

**当前状态**：训练代码崩溃问题已修复并通过本地 CPU 验证，但 **Kaggle 上的代码包是修复前上传的**，
因此方案的第一步就是把修复送上去。

---

## 零、先想清楚：走哪条上传路线

代码进 Kaggle 有两条路，**推荐路线 A**：

| | 路线 A：重新上传代码 Dataset（推荐） | 路线 B：用补丁 cell |
|---|---|---|
| 做法 | 本地 robocopy 后重新 `kaggle datasets create` | 保留旧代码包，Notebook 里跑补丁 cell |
| 优点 | 干净、可复现、后续 session 直接跑 | 省一次上传（约 166 MB） |
| 缺点 | 需重新上传一次 | 每次新建 session 都要跑补丁 cell，忘记就崩 |
| 适用 | **正式训练** | 临时验证 / 不想等上传 |

> 两条路都已验证可用。下面以路线 A 为主线，路线 B 见文末「附录」。

---

## 一、准备两个 Kaggle Dataset

### 1.1 代码仓库（约 166 MB）

**关键**：必须上传**已修复**的代码。修复涉及两个文件：
`ultralytics/nn/mm/dual_val.py` 与 `ultralytics/models/yolo/multimodal/train.py`。

```powershell
# 用仓库自带的脚本一步到位（它会 robocopy + 建 dataset + 上传）
cd C:\Users\不如喝茶去\Desktop\双光谱毕设\scripts
powershell -ExecutionPolicy Bypass -File .\upload_to_kaggle.ps1 -Check   # 先检查环境与登录
powershell -ExecutionPolicy Bypass -File .\upload_to_kaggle.ps1         # 正式上传代码
```

脚本要点（已内置，无需手改）：

- `robocopy` 排除 `ResTest logs __pycache__ .git .idea .vs` 与 `*.pyc *.log`，避免把历史训练输出传上去。
- 用 `kaggle datasets create -p . --dir-mode zip` 上传；Notebook 侧会自动解压（模板 cell 4 已处理）。
- `logs/` 目录已在排除列表内 —— 该目录无需存在，缺失不影响上传。

若想手动执行，等价命令：

```powershell
$src = "C:\Users\不如喝茶去\Desktop\双光谱毕设\MutilModel_3485988898"
$dst = "D:\kaggle_upload\mutilmodel-code\MutilModel_3485988898"
robocopy $src $dst /E /XD ResTest logs __pycache__ .git .idea /XF *.pyc

cd D:\kaggle_upload\mutilmodel-code
kaggle datasets init -p .
# 编辑 dataset-metadata.json，把 id 改成 <你的Kaggle用户名>/mutilmodel-code
kaggle datasets create -p . --dir-mode zip
```

### 1.2 数据集（4.7 GB / 22 万文件）

数据已上传为 slug `uav-data`（整目录挂载模式）。**必须确认双标签在位**：

```
uav_data/
├── images/{train,val,test}/        可见光图
├── images_ir/{train,val,test}/     红外图（与 images 严格 1:1）
├── labels/{train,val,test}/        RGB 标签（visible.json）
├── labels_ir/{train,val,test}/     IR  标签（infrared.json）
└── data.yaml                       含 labels: {rgb: labels, ir: labels_ir}
```

> **这是双头架构的命门**：缺 `labels_ir` 时数据层会**静默降级**为单标签，
> IR 头拿不到监督却不报错，训练能跑完但结果无意义。
> 模板第 6 个 cell 会断言 `labels.ir == labels_ir`，不通过就停下。
> 如缺失，回本地跑 `scripts/gen_labels_ir.py` 修复后重传。

---

## 二、Notebook 设置

| 项 | 取值 | 说明 |
|---|---|---|
| Accelerator | **GPU T4 x2** 或 **GPU P100** | 单卡 `device=0`；双卡 `device=0,1` 见第八节 |
| Internet | 关闭 | 不需要联网 |
| Persistence | 关闭 | 产物靠 Output 保存 |

> 不吃满双卡也能跑，默认单卡更省心。是否上双卡、怎么配参数，见第八节。

---

## 三、执行顺序（逐个 cell 运行）

模板：`scripts/kaggle_train_dual_fusion.ipynb`
（由 `scripts/build_kaggle_notebook.py` 生成，改模板请改生成器）

| # | Cell | 作用 | 通过判据 |
|---|---|---|---|
| 0 | 说明 | 前置准备与用法 | — |
| 1 | **配置区** | 只需改这一段 | `RUN_NAME` / `EPOCHS` / `MODE` |
| 2 | 环境自检 | GPU、显存、磁盘、补依赖 | 打印 GPU 型号与显存 |
| 3 | 定位 Dataset | 模糊匹配挂载路径 | `CODE_MOUNT` / `DATA_MOUNT` 非空 |
| 4 | 准备代码 | 复制到本地盘 + 强制 import 仓库 fork | `ultralytics 路径` 指向本地副本 |
| 5 | 准备数据 | 写可写的 `data.yaml` 副本 | 打印改写后的配置 |
| 6 | **数据自检** | 双标签映射、四路 1:1、标签非空 | 断言全过 |
| 7 | 续训准备 | `MODE=resume/extend` 才需要 | `fresh` 时跳过 |
| 8 | 训练前自检 | `tools/preflight_train.py`（可选） | 非 0 仅提示、不阻断 |
| 9 | **启动训练** | 调 `train_full.py`，带 11h 守护 | 日志出现双头已启用标记 |
| 10 | 结果汇总 | 读 `results.csv` 出三路 mAP 与增益 | 打印分模态指标 |
| 11 | 打包 | 产物搬到 `/kaggle/working/kaggle_out` | 输出目录与体积清单 |

### 配置区关键项

```python
CODE_SLUG = "mutilmodel-code"      # 代码 Dataset 的 slug
DATA_SLUG = "uav-data"             # 数据 Dataset 的 slug
RUN_NAME  = "abl_A0-2_dual_head_dfs"
EPOCHS    = 60                     # 本次 session 轮数上限，留足余量
BATCH     = None                   # None=按显存自动；OOM 就手填 8
DFS_WEIGHT = 0.5                   # DFSC 权重；0.0=关闭（消融 A0-3）
MODE      = "fresh"                # fresh | resume | extend
MAX_HOURS = 11.0                   # 软保护，避免被 12h 硬杀
```

### 训练命令（cell 9 自动拼装）

```
python train_full.py --route A --epochs <EPOCHS> \
    --config dual-fusion --label-from keep \
    --data <data.yaml 副本> --imgsz 640 --batch <BATCH> \
    --workers <WORKERS> --device 0 --cache none \
    --name <RUN_NAME> --mm-dfs-weight <DFS_WEIGHT>
```

用 `--label-from keep` 而非 `dual`：映射已由 cell 5 写好，让脚本再改一次可能覆盖 Kaggle 路径配置。

---

## 四、跑完怎么确认「真的对了」

**这三条是双头是否真正生效的证据**，日志里必须都能看到：

```
[OK] DualHeadDetectionLoss: nc=1, branches=['rgb', 'ir', 'fuse'], ...
[OK] IR labels: N images, ... dir=labels_ir          # 双标签生效
[OK] 双头 + 融合分支：启用分模态评测器，loss_names=[...9 项...]
```

以及验证阶段的三路指标与融合增益：

```
DualHead 分模态 mAP50-95(B): rgb=.. ir=.. fuse=..
DualHead 融合增益(vs RGB) = +0.0xxxx  ✔ 融合更优
```

模板 cell 9 末尾会自动扫日志做探针（当前检查 `DualHeadDetectionLoss`、`IR labels:`、
`dual_labels_configured=True` 三项），缺失会打 `[!!]` 警告。

### 报数注意（重要）

`ir` 头用 IR 坐标系标签（`infrared.json`），`rgb`/`fuse` 头用 RGB 坐标系标签（`visible.json`），
两者不同源 —— **三路 mAP 不可跨模态直接比大小**，只能说「各自模态内，融合是否更优」。
论文主指标取 `metrics/fuse/metrics/mAP50-95(B)`。

---

## 五、12 小时上限：多 session 续训

Kaggle 单次 session 上限 12 小时，5.6 万张图 / 150 轮在 T4 上跑不完，必须分次。

```
session 1: MODE="fresh",  EPOCHS=60
   ↓ 跑完或被 11h 守护停掉 → 右上角 Save Version
session 2: Add Input → Your Work → 挂载 session 1 的 Output
            MODE="resume", EPOCHS 不变
   ↓ 接着把剩下的轮数跑完（resume 沿用上次的 total epochs）
session 3: 若已跑满 60 轮还想继续 → MODE="extend", EXTEND_EPOCHS=150
```

| 模式 | 行为 | 注意 |
|---|---|---|
| `resume` | 沿用 `last.pt` 全部状态（优化器/EMA/数据顺序/总轮数） | **不增加**总轮数 |
| `extend` | `--route C --pretrained best.pt` 另起一段训练 | 重置优化器状态，LR 按新 schedule |

**两条硬性要求**：

1. `PREV_OUTPUT_SLUG` 必须指向上一版 Output，且 `RUN_NAME` 保持一致。
   `resume` 依赖 `get_latest_run()` 在**当前工作目录下递归找最新 `last*.pt`**，
   `RUN_NAME` 变了就找不到、或找错到别的实验上（cell 7 会清理其它 `last*.pt` 防这个）。
2. `resume` 只允许覆盖 `imgsz / batch / device / close_mosaic`。
   想改 `workers`、`--mm-dfs-weight`、`--epochs` 一律不生效，得用 `extend`。

### 怎么定 `EPOCHS`

不要凭感觉填。先跑一次让它**至少完成 1 个 epoch**，从日志读单轮耗时，
再用 `11h ÷ 单轮耗时` 反推本 session 能跑多少轮，把 `EPOCHS` 设成略小于这个数。

---

## 六、常见问题速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 双路骨干 + 三头更吃显存 | `BATCH` 手填 8（甚至 4）；**不要降 `imgsz`**（目标中位仅 37 px） |
| `Validation loss dimension mismatch ... got 9 ... expects 3` | 代码包是修复前的 | 走路线 A 重传，或按附录跑补丁 cell |
| 日志无 `IR labels:` / `dual_labels_configured=False` | 双标签没生效，IR 头裸奔 | 查 `data.yaml` 副本的 `labels` 映射与是否真用 `--config dual-fusion` |
| Output 超 20 GB | 产物写进了 `/kaggle/working` | 模板默认把数据/代码/日志放 `/kaggle/temp`；若 Output 出现 `code/` 说明 temp 不可写 |
| 启动后长时间停在扫描数据集 | 数据层要遍历 22 万文件建索引 | 首次数分钟属正常，索引建好后每 epoch 快很多 |
| 12h 到点被硬杀 | — | `MAX_HOURS=11` 会在硬杀前主动停，`last.pt` 每 epoch 更新，最多丢一个 epoch |
| P100 上 `amp` 效果有限 | P100 无 fp16 加速 | 正常现象；换卡后 `RUN_NAME` 加后缀区分，避免曲线混在一起 |

---

## 七、验收清单

- [ ] 代码 Dataset 已重传（或补丁 cell 已跑）
- [ ] 数据 Dataset 的 `labels_ir/` 与 `images/` 严格 1:1（cell 6 断言通过）
- [ ] 日志出现 `DualHeadDetectionLoss` + `IR labels:` + `分模态评测器` 三条证据
- [ ] `results.csv` 含 `train/{box,cls,dfl}_loss_{rgb,ir,fuse}` 与三路 mAP、融合增益列
- [ ] 融合增益 `fusion_gain_over_rgb > 0`（论文核心结论）
- [ ] Output 已 Save Version，`weights/best.pt` 可下载

---

## 附录：路线 B —— 用补丁 cell 免重传

若暂时不想重新上传 166 MB 代码包，在 Notebook 的 **cell 4（准备代码）之后、cell 5 之前**
插入一个 cell，内容为 `scripts/dual_head_fix_patch.py`，运行一次即可。

补丁特性（已自校验 10/10）：

- **锚点替换**：锚点不唯一会显式报错，不会静默改坏源码
- **语法自检**：改完 `ast.parse` 验证
- **幂等**：已打过则跳过，重复运行安全
- **清缓存**：清空 `sys.modules` 里的 `ultralytics.*`，确保后续 import 到补丁后源码

> 之后再重传代码包时，补丁 cell 会自动检测到标记并跳过，不影响流程。
> 但它**每次新建 session 都要重跑**——忘记跑就会重现 9 vs 3 崩溃。作为长期方案不如路线 A。

---

## 八、双卡（T4 ×2）怎么配

**先说结论**：双卡可用、代码路径静态审计无阻断项，但**加速比约 1.5~1.8 而非 2.0**；
且必须同步调整 `batch` 与 `lr0`，否则等于偷偷降低学习率，**曲线与之前单卡实验不可比**。

### 8.1 为什么吃不满两张卡

三个原因，都是机制性的：

1. **验证/绘图/保存只在 rank0 执行**（`trainer.py` 有 12 处 `RANK in {-1, 0}`）。
   每轮验证期间 rank1 空等，而双头验证要跑三路 NMS，这段开销不小。
2. **T4 无 NVLink**，NCCL 梯度同步走 PCIe，通信是纯开销。
3. **数据侧受 workers 限制**：Kaggle 一般 4 核，双卡共享，`workers` 给大会争抢。

你的环境实测 `batch=16  workers=4`，即单卡舒适值 —— 这本身就是个信号：
数据管线未必喂得饱两张卡。

### 8.2 参数怎么配（关键）

DDP 下 `batch` 是 **per-rank** 的，总 batch = `batch × 卡数`：

```python
# 配置区
BATCH  = 16        # per-rank；16 % 2 == 0 满足整除约束（框架强制要求）
DEVICE = "0,1"     # 改 cell 9 里的 "--device", "0" 为 "0,1"
```

总 batch 从 16 变 32，**`lr0` 要按线性缩放同步放大**：

| 方案 | per-rank batch | 总 batch | lr0 建议 | 相对吞吐 |
|---|---|---|---|---|
| 单卡 `0` | 16 | 16 | 0.01（默认） | 1.0 |
| **双卡 `0,1`** | **16** | **32** | **~0.02** | **~1.6** |
| 双卡 `0,1`（显存紧） | 8 | 16 | 0.01 | ~1.5 |

> `lr0` 不在 Notebook 配置区里，需在 `train_full.py` 的 `train_kwargs` 中加
> `lr0=0.02`，或改 `DEFAULTS`。**不改 lr 就上双卡 = 等效降学习率**，
> 收敛变慢，且与已有单卡结果无法对比。

### 8.3 硬约束

- `batch % 卡数 == 0`，否则框架直接抛错（`torch_utils.py:316`）。
  16 和 8 都能被 2 整除，安全。
- **消融实验必须固定卡数**。`DFS_WEIGHT` 的绝对影响随 world_size 变化
  （DFSC 项按 batch_size 缩放，DDP 再按卡数归一），
  跨卡数比 `DFS_WEIGHT=0.5` 与 `0.0` 的差异会失真。
  要么全程单卡，要么全程双卡，别混。

### 8.4 建议

**追求实验可比性 → 单卡。** 你现有的单卡 baseline 曲线已经建立，
上双卡会同时改变 batch、lr、梯度噪声三个变量，等于换了套实验条件。

**追求墙钟时间 → 双卡，但按 8.2 配齐 batch + lr**，且 `RUN_NAME` 加 `-dual` 后缀
区分，避免与单卡曲线混在同一目录。

一个折中：**先单卡跑通完整 1 个 epoch**（确认双头/双标签在 Kaggle 真的生效），
再决定是否切双卡。这比一上来就双卡、出问题时分不清是 DDP 还是数据的问题省事得多。

---

## 附：本次修复涉及的改动（供重传时核对）

| 文件 | 改动 |
|---|---|
| `ultralytics/models/yolo/multimodal/train.py` | `get_validator` 增加双头架构分派；新增 `_is_dual_head_architecture` / `_dual_head_loss_names` |
| `ultralytics/nn/mm/dual_val.py` | contract-aware `preprocess`；`_swap_labels` 帧索引同源；`_resolve_val_loss_names` 按实际项数标注；补 `speed` 契约与权重路径加载；`postprocess` 兼容 dict；新增 AFSS 打分接口 |

详细的根因分析与验证记录见 `双头验证期崩溃修复说明.md`。
