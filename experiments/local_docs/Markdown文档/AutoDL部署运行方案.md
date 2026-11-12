# 双光谱双头融合模型 · AutoDL 部署运行方案

**为什么 AutoDL 比 Kaggle 合适**：`train_full.py` 的 `DEFAULTS` 本来就是按 AutoDL 写的
（`data: /root/autodl-tmp/uav_data/data.yaml`），路径零改动；单次可跑满 12h 以上不必分次续训；
数据可常驻 `/root/autodl-tmp`（大数据盘）不用每次解压。

***

## 一、租什么机器

| 项   | 建议                                         | 理由                                                         |
| --- | ------------------------------------------ | ---------------------------------------------------------- |
| GPU | **RTX 4090 24G**                           | `DEFAULTS` 按 4090 调优（`batch=32` 实测占 14–17G）；显存充裕可上更大 batch |
| 备选  | RTX 3090 24G / A100 40G                    | 显存 ≥ 24G 都合适                                               |
| 镜像  | **PyTorch 2.1+ / Python 3.10 / CUDA 12.1** | 与本项目 `新手指南.txt` 建议一致                                       |
| 数据盘 | 选**扩容的数据盘**（如 50G+）                        | 数据 4.61 GB + 训练产物（`ResTest/`）会持续增长                         |

> 单卡足够。本项目跑的是 DDP-friendly 的常规训练，但多卡性价比低（详见附录）。

***

## 二、要传什么（三样，共约 4.63 GB）

### 2.1 清单

| # | 传什么           | 本地路径                     | 体积                 | 传到服务器哪里                                        |
| - | ------------- | ------------------------ | ------------------ | ---------------------------------------------- |
| 1 | **代码仓库**      | `MutilModel_3485988898/` | 15.3 MB            | `/root/autodl-tmp/code/MutilModel_3485988898/` |
| 2 | **数据集**       | `D:\uav_data\`           | 4.61 GB / 22.4 万文件 | `/root/autodl-tmp/uav_data/`                   |
| 3 | **预训练权重**（可选） | `yolo11n.pt`             | 5.4 MB             | `/root/autodl-tmp/yolo11n.pt`                  |

**必须传的是 1 + 2。** 第 3 项仅当你想走「路线 B（COCO 权重初始化骨干）」时才需要；
走路线 A（随机初始化）可以不传。

> **不需要传的**：`ResTest/`（历史训练输出）、`logs/`、`__pycache__/`、`.git/`、`.idea/`。
> 这些会显著增大体积且无用。

### 2.2 关于数据集：两种方案

你在 Kaggle 上用的是整目录 `uav_data`（含 `images/ images_ir/ labels/ labels_ir/ data.yaml`）。
AutoDL 上有两个选择：

**方案 A（推荐）：只传** **`data.yaml`，数据在服务器上重新生成**

AutoDL 有公网可下载的 Anti-UAV300 原始数据集吗？多数情况没有，需要自行上传原始数据。
**但更省事的是直接传处理好的** **`uav_data`** —— 因为 `prepare_mm_dataset.py` 依赖
`config.py` 里的本地路径（`E:/双光谱数据集/...`），服务器上没有。

**结论：直接把** **`D:\uav_data`** **整个传上去**（4.61 GB）。这是最稳的路径。

### 2.3 传输方式（三选一）

**方式 1：AutoDL 官方 JupyterLab 上传（最省事，适合代码）**

实例开机后进 JupyterLab，直接把代码仓库的 **zip** 拖进去解压。

```bash
# 在服务器上解压
cd /root/autodl-tmp
unzip MutilModel_3485988898.zip -d /root/autodl-tmp/code/
```

**方式 2：`scp`** **/** **`rsync`（适合大数据集，推荐）**

AutoDL 实例详情页有 SSH 连接信息（host / port / 密码）。

```powershell
# 本地 PowerShell —— 先传代码
scp -P <端口> -r "C:\Users\不如喝茶去\Desktop\双光谱毕设\MutilModel_3485988898" `
    root@<host>:/root/autodl-tmp/code/

# 再传数据（4.6 GB，耗时取决于上行带宽）
scp -P <端口> -r "D:\uav_data" root@<host>:/root/autodl-tmp/
```

**更推荐** **`rsync`**（断点续传、可排除，比 scp 稳）：

```bash
# 在有 rsync 的环境（WSL / Git Bash）里执行
rsync -avz --progress -e "ssh -p <端口>" \
    --exclude='ResTest' --exclude='logs' --exclude='__pycache__' \
    --exclude='.git' --exclude='.idea' \
    "/mnt/c/Users/不如喝茶去/Desktop/双光谱毕设/MutilModel_3485988898/" \
    root@<host>:/root/autodl-tmp/code/MutilModel_3485988898/
```

**方式 3：AutoDL 数据盘 + 网盘（数据超大时）**

AutoDL 支持从阿里云盘/百度网盘拉取，先把 `uav_data` 打包上传网盘，再在服务器上下载。

> 上传前建议先打包成 tar，22.4 万个小文件逐个传极慢：
>
> ```powershell
> tar -cf D:\uav_data.tar -C D:\uav_data .
> # 传上去后在服务器解压
> mkdir -p /root/autodl-tmp/uav_data && tar -xf /root/autodl-tmp/uav_data.tar -C /root/autodl-tmp/uav_data
> ```
>
> 单文件传输比 22 万个散文件快数倍。

***

## 三、服务器上的环境准备

```bash
# 1) 确认 GPU
nvidia-smi

# 2) 建环境（镜像自带 conda 时）
conda create -n yolomm python=3.10 -y
conda activate yolomm

# 3) 装带 CUDA 的 PyTorch（按镜像实际情况；若镜像已带 torch 可跳过）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) 装其余依赖
pip install numpy opencv-python pillow pyyaml requests matplotlib pandas seaborn tqdm psutil scipy

# 5) 项目额外依赖（训练必须）
pip install polars ultralytics-thop

# 6) 【关键】不要 pip install ultralytics ！
#    项目用的是仓库内置的修改版 ultralytics（含双头+融合分支），
#    装官方版会覆盖掉。项目只需在根目录下运行即可 import 到。
```

***

## 四、目录结构确认

上传完成后，服务器上应是这样：

```
/root/autodl-tmp/
├── code/
│   └── MutilModel_3485988898/          <- 仓库根（含 train_full.py + ultralytics/）
│       ├── train_full.py
│       ├── ultralytics/
│       └── tools/
├── uav_data/                            <- 数据（含 data.yaml）
│   ├── data.yaml
│   ├── images/{train,val,test}/
│   ├── images_ir/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── labels_ir/{train,val,test}/
└── yolo11n.pt                          (可选)
```

`train_full.py` 的默认 `data` 是 `/root/autodl-tmp/uav_data/data.yaml`，
**与上面的结构完全对应，无需改任何路径**。

***

## 五、跑之前：数据自检

```bash
cd /root/autodl-tmp/code/MutilModel_3485988898

# 1) 通用自检：数据配对、尺寸比、模型可构建、权重盘点
python tools/preflight_train.py --data /root/autodl-tmp/uav_data/data.yaml
```

**注意** **`preflight_train.py`** **只抽查** **`labels/`** **这一路**（它的 `check_label_pairing` 写死了
`"labels/" + split`），**不会检查** **`labels_ir/`**。而双标签恰恰是双头架构的命门，
所以必须单独确认：

```bash
# 2) 双标签专项自检（preflight 不覆盖，必须自己跑）
python - <<'PY'
from pathlib import Path
import yaml

root = Path("/root/autodl-tmp/uav_data")
cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))

print("labels 映射 :", cfg.get("labels"))
print("modalities :", cfg.get("modalities"), "| Xch:", cfg.get("Xch"), "| nc:", cfg.get("nc"))

assert isinstance(cfg.get("labels"), dict), \
    "data.yaml 缺 labels 映射 -> 双头会静默降级为单标签，IR 头拿不到监督"
assert cfg["labels"].get("ir") == "labels_ir", \
    f"labels.ir 应为 labels_ir，实际 {cfg['labels'].get('ir')}"

for split in ("train", "val"):
    n = {d: len(list((root / d / split).glob("*"))) for d in
         ("images", "images_ir", "labels", "labels_ir")}
    print(f"{split:<6}", n)
    assert len(set(n.values())) == 1, \
        f"{split} 四路数量不一致 {n} -> 双标签必须与图像严格 1:1"
print("\n双标签自检通过。")
PY
```

**期望结果**：`labels` 映射为 `{'rgb': 'labels', 'ir': 'labels_ir'}`，
且 `train`/`val` 下 `images / images_ir / labels / labels_ir` 四路文件数完全一致。

> 缺 `labels_ir` 时数据层会**静默降级**为单标签，IR 头拿不到监督却不报错，
> 训练能跑完但结果无意义。若自检不过，回本地跑
> `python scripts/gen_labels_ir.py --root D:\uav_data` 修复后重传该目录。

***

## 六、启动训练

### 6.1 推荐命令（消融 A0-2：双头 + DFSC）

```bash
cd /root/autodl-tmp/code/MutilModel_3485988898

python train_full.py \
    --route A \
    --config dual-fusion \
    --label-from keep \
    --data /root/autodl-tmp/uav_data/data.yaml \
    --epochs 150 \
    --batch 32 \
    --imgsz 640 \
    --workers 16 \
    --device 0 \
    --cache none \
    --mm-dfs-weight 0.5 \
    --name abl_A0-2_dual_head_dfs \
    2>&1 | tee /root/autodl-tmp/train_A0-2.log
```

**参数说明**：

| 参数                     | 取值            | 说明                             |
| ---------------------- | ------------- | ------------------------------ |
| `--route A`            | 随机初始化         | 路线 B 需 `--pretrained`；路线 C 是续训 |
| `--config dual-fusion` | 双头+融合分支       | 三个独立 Detect 头                  |
| `--label-from keep`    | 不改写 data.yaml | 映射已在上传的数据里，别让脚本再改              |
| `--batch 32`           | 4090 24G      | 实测占 14–17G；显存紧降到 16            |
| `--imgsz 640`          | **不要降**       | 目标中位仅 37 px，降分辨率伤精度            |
| `--mm-dfs-weight 0.5`  | DFSC 开启       | 消融 A0-3 用 `0.0`                |

### 6.2 消融 A0-3（双头，关闭 DFSC）

```bash
python train_full.py --route A --config dual-fusion --label-from keep \
    --data /root/autodl-tmp/uav_data/data.yaml \
    --epochs 150 --batch 32 --imgsz 640 --workers 16 --device 0 \
    --cache none --mm-dfs-weight 0.0 \
    --name abl_A0-3_dual_head_nodfs \
    2>&1 | tee /root/autodl-tmp/train_A0-3.log
```

### 6.3 后台运行（防断连中断训练）

SSH 断开会杀掉前台进程。用 `nohup` 或 `tmux`：

```bash
# 方式 1：nohup
nohup python train_full.py --route A --config dual-fusion --label-from keep \
    --data /root/autodl-tmp/uav_data/data.yaml \
    --epochs 150 --batch 32 --imgsz 640 --workers 16 --device 0 \
    --cache none --mm-dfs-weight 0.5 --name abl_A0-2_dual_head_dfs \
    > /root/autodl-tmp/train_A0-2.log 2>&1 &

tail -f /root/autodl-tmp/train_A0-2.log     # 看进度

# 方式 2：tmux（推荐，可随时回去交互）
tmux new -s train
# ... 在 tmux 里跑训练命令 ...
# Ctrl+B 然后 D 脱离；tmux attach -t train 回来
```

***

## 七、训练启动后必须确认的三条证据

日志里**必须**都能看到，缺任一条说明配置没生效：

```
[OK] DualHeadDetectionLoss: nc=1, branches=['rgb', 'ir', 'fuse'], weights={...}
[OK] IR labels: N images, ... dir=labels_ir
[OK] 双头 + 融合分支：启用分模态评测器，loss_names=[...9 项...]
```

以及验证阶段（每轮结束）的三路指标：

```
DualHead 分模态 mAP50-95(B): rgb=.. ir=.. fuse=..
DualHead 融合增益(vs RGB) = +0.0xxxx  ✔ 融合更优
```

> `train_full.py` 跑完会自动打印这些自检要点与等价复现命令。

### 报数注意

`ir` 头用 IR 坐标系标签（`infrared.json`），`rgb`/`fuse` 头用 RGB 坐标系标签
（`visible.json`），两者不同源 —— **三路 mAP 不可跨模态直接比大小**，
只能说「各自模态内，融合是否更优」。论文主指标取 `metrics/fuse/`。

***

## 八、训练完取回产物

产物在 `/root/autodl-tmp/code/MutilModel_3485988898/ResTest/<RUN_NAME>/`：

```
ResTest/abl_A0-2_dual_head_dfs/
├── weights/best.pt        <- 最优权重
├── weights/last.pt
├── results.csv            <- 三路 mAP + 融合增益 + 9 项 loss 列
├── results.png
└── args.yaml
```

用小文件下载（AutoDL 控制台「文件」面板下载，或 `scp` 拉回）：

```powershell
scp -P <端口> root@<host>:/root/autodl-tmp/code/MutilModel_3485988898/ResTest/abl_A0-2_dual_head_dfs/weights/best.pt .
scp -P <端口> root@<host>:/root/autodl-tmp/code/MutilModel_3485988898/ResTest/abl_A0-2_dual_head_dfs/results.csv .
```

> AutoDL 实例的**数据盘可保留**（关机不计费），下次开机数据还在，不必重传。

***

## 九、常见问题

| 症状                                                           | 原因                                 | 处理                                                           |
| ------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------ |
| `Validation loss dimension mismatch ... got 9 ... expects 3` | 传的是修复前的旧代码                         | 确认传的是**已修复**的 `dual_val.py` / `multimodal/train.py`（见附录改动清单） |
| `DualHeadDetectionLoss: ... 缺少标签键` 或 IR 头 mAP 恒为 0           | `labels_ir` 没生效                    | 跑 `preflight_train.py` 确认 `dual_labels_configured=True`      |
| `CUDA out of memory`                                         | batch 太大                           | 降到 16 或 8；**不要降 imgsz**                                      |
| `ModuleNotFoundError: polars` / `thop`                       | 依赖缺失                               | `pip install polars ultralytics-thop`                        |
| `WindowsPath` / 中文路径报错                                       | 仅 Windows 才有                       | Linux 上不会出现                                                  |
| 训练速度远低于预期                                                    | `workers` 不足或 `cache=none` 读盘慢     | 4090 机器 `workers 16`；数据盘剩余 >60G 可试 `--cache disk`            |
| 误装了官方 ultralytics                                            | `pip install ultralytics` 覆盖了 fork | `pip uninstall ultralytics -y`，项目会从根目录 import                |

***

## 附录 A：本次修复涉及的改动（传代码前核对）

| 文件                                            | 改动                                                                                                                                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `ultralytics/models/yolo/multimodal/train.py` | `get_validator` 增加双头架构分派；新增 `_is_dual_head_architecture` / `_dual_head_loss_names`                                                        |
| `ultralytics/nn/mm/dual_val.py`               | contract-aware `preprocess`；`_swap_labels` 帧索引同源；`_resolve_val_loss_names` 按实际项数标注；补 `speed` 契约与权重路径加载；`postprocess` 兼容 dict；新增 AFSS 打分接口 |

本地自检命令（确认改的是这两个文件）：

```powershell
Select-String -Path "C:\Users\不如喝茶去\Desktop\双光谱毕设\MutilModel_3485988898\ultralytics\nn\mm\dual_val.py" -Pattern "BRANCH_LOSS_NAMES"
Select-String -Path "C:\Users\不如喝茶去\Desktop\双光谱毕设\MutilModel_3485988898\ultralytics\models\yolo\multimodal\train.py" -Pattern "_is_dual_head_architecture"
```

两条都应有命中。详细根因见 `双头验证期崩溃修复说明.md`。

***

## 附录 B：要不要用多卡

**A100 40G × N 时可以考虑，但本项目收益有限**：

- 验证/绘图/保存只在 rank0 跑（`trainer.py` 有 12 处 `RANK in {-1, 0}`），其余卡空等

- `batch` 是 **per-rank** 的，总 batch = `batch × 卡数`，**`lr0`** **必须按总 batch 线性放大**
  （`trainer.py:486` 有 `self.loss *= world_size`），否则等效降学习率

- `batch % 卡数 == 0` 是硬约束（`torch_utils.py:316`）

- **消融实验别混卡数**：DFSC 项按 batch\_size 缩放、DDP 再按卡数归一，
  跨卡数比 `--mm-dfs-weight 0.5` vs `0.0` 的差异会失真

单卡 4090 上是本项目最省心的配置（`DEFAULTS` 就是照它调的）。
