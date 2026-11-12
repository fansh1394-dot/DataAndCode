# RKNN 兼容性验证方法

> 目的：在投入大量训练算力**之前**，确认所选模型的算子能被 RKNN-Toolkit2 转换，
> 避免"训完才发现部署不了"的重大风险。
> 创建：2026-09-19 ｜ 目标平台：RK3588（6 TOPS NPU，INT8）

---

## 〇、为什么要先做这件事（以及不需要做什么）

### 需要做

**验证算子兼容性是"选型阶段"的工作，不是"部署阶段"的工作。**

你的创新点里包含自定义算子（`Poly_Kernel_Inception_Block`、`C3k2_PKIModule`）。RKNN-Toolkit2 对算子的支持是**有限集合**，遇到不支持的算子通常有两种结果：

| 情况 | 后果 |
|---|---|
| 转换直接失败 | 好情况——当场就知道，代价小 |
| 静默回退到 CPU | 坏情况——能转换，但推理极慢，实时性目标落空 |

第二种更危险，因为表面"成功"了。所以验证必须**看逐层分析报告**，不能只看是否报错。

### 不需要做

**验证 RKNN 兼容性不需要剪枝或蒸馏。** 这两个是并列的、相互独立的后处理步骤：

```
训练完成
   ├─→ 导出 ONNX ─→ RKNN 转换 ─→ 部署      ← 兼容性验证在这一步
   └─→ 剪枝 / 蒸馏（可选，为省算力）        ← 独立步骤，与兼容性无关
```

**理由**：

**其一，兼容性由算子类型决定，与通道数无关。** 剪枝只是改变通道数，不会把"不支持的算子"变成"支持的算子"。所以先剪枝再验证，等于白白多做一步。

**其二，剪枝会引入新风险。** 剪枝后的结构可能因通道对齐问题产生额外的转换障碍，反而干扰你判断"到底是哪个算子不支持"。

**其三，正确的顺序是"先验证、后剪枝"。** 先用完整模型确认算子都能转，再在此基础上剪枝、再次验证。

**所以：先做算子兼容性验证，剪枝蒸馏放到训练完成之后。**

---

## 一、验证流程总览

```
① 构造随机权重模型（不训练）
      ↓
② 导出 ONNX
      ↓
③ 检查 ONNX 算子清单
      ↓
④ RKNN-Toolkit2 转换 + 逐层分析
      ↓
⑤ 判定：支持 / 回退CPU / 失败
```

**关键点**：第 ① 步用**随机权重**即可，不需要等训练完成。因为兼容性只跟**网络结构**有关，与权重数值无关。**训练开始前就能做，耗时约半天。**

---

## 二、第一步：构造随机权重模型并导出 ONNX

### 方法一：用框架内置导出（推荐）

框架的 `exporter.py` 已原生支持 RKNN 格式（`format='rknn'`）。训练前用随机初始化的模型试导出：

```python
# scripts/calib/rknn_check.py
"""RKNN 兼容性预检：随机权重模型 -> ONNX -> 算子检查。"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from ultralytics import YOLOMM

# 待验证的配置（换成你要检查的目标配置）
CONFIGS = {
    "baseline":  "ultralytics/cfg/models/mm/yolo11n-mm-mid.yaml",
    "pki":       "ultralytics/cfg/models/mm/yolo11n-mm-mid-pki.yaml",
    "c3k2-pki":  "ultralytics/cfg/models/mm/detect/yolo11/extraction/yolo11n-mm-c3k2_pkimodule.yaml",
}

for name, yaml_rel in CONFIGS.items():
    print("=" * 70)
    print(f"验证配置: {name}  ({yaml_rel})")
    print("=" * 70)
    model = YOLOMM(str(PROJ / yaml_rel))     # 随机初始化，无需训练
    try:
        out = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        print(f"  [ONNX] 导出成功: {out}")
    except Exception as e:
        print(f"  [ONNX] 导出失败: {type(e).__name__}: {e}")
```

> ⚠️ **多模态导出的实际情况（已实测发现）**：
> 本项目的多模态模型导出时，框架会把 RGB(3ch) 与 IR(3ch) **拼接为单输入 6 通道张量**。
> 实测日志显示：`input shape (1, 6, 640, 640) BCHW`。
>
> **这反而是好消息**：ONNX 图是标准单输入结构，避免了双输入签名的转换麻烦。
> 但**部署端需要自己做拼接**：从两个相机取图 → 各自预处理 → 拼成 6 通道 → 喂给 NPU。
> 这是工程实现细节，不影响 RKNN 转换可行性。

### 方法二：导出后手工检查 ONNX 算子

```python
# 检查 ONNX 图中用到的所有算子类型
import onnx
from collections import Counter

m = onnx.load("yolo11n-mm-mid-pki.onnx")
ops = Counter(n.op_type for n in m.graph.node)
print("算子清单：")
for op, cnt in sorted(ops.items(), key=lambda x: -x[1]):
    print(f"  {op:24} × {cnt}")
```

**用途**：这份清单是后续判断的基础。若出现自定义算子名（如 `Poly_Kernel_Inception_Block` 未被展开），说明导出时没有降级为标准算子，RKNN 大概率不支持。

**理想情况**：所有算子都是标准 ONNX 算子（Conv / Mul / Add / Concat / Resize / Sigmoid / Softmax 等），这样 RKNN 支持度最高。

---

## 三、第二步：RKNN-Toolkit2 转换

### 环境准备

RKNN-Toolkit2 **只能在 x86 Linux 上运行**（不支持 Windows/ARM）。AutoDL 的服务器是 x86 Linux，可以直接用。

```bash
# 建议单独建 conda 环境，避免与训练环境的依赖冲突
conda create -n rknn python=3.8 -y
conda activate rknn

# 从 Rockchip 官方仓库获取（需匹配版本）
pip install rknn-toolkit2
```

> **版本匹配很重要**：RKNN-Toolkit2 版本需与开发板上的 `librknnrt.so` 运行时版本对应。
> 建议先确认开发板环境，再选工具链版本。官方 GitHub：`airockchip/rknn-toolkit2`

### 转换脚本

```python
# rknn_convert.py  —— 在 rknn 环境中运行
from rknn.api import RKNN

ONNX_MODEL = "yolo11n-mm-mid-pki.onnx"
RKNN_OUT   = "yolo11n-mm-mid-pki.rknn"
TARGET     = "rk3588"          # 或 rk3588s

rknn = RKNN(verbose=True)

# ---- 1) 配置 ----
rknn.config(
    mean_values=[[0, 0, 0], [0, 0, 0]],     # 双输入需给两组
    std_values=[[255, 255, 255], [255, 255, 255]],
    target_platform=TARGET,
    quantized_dtype="w8a8",                  # INT8 量化
    optimization_level=3,
)

# ---- 2) 载入 ONNX ----
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print("❌ load_onnx 失败 —— 算子或图结构不被支持")
    exit(ret)

# ---- 3) 构建（此处会做算子分析与量化）----
ret = rknn.build(do_quantization=True, dataset="calib_dataset.txt")
if ret != 0:
    print("❌ build 失败 —— 通常意味着存在不支持的算子")
    exit(ret)

# ---- 4) 导出 ----
ret = rknn.export_rknn(RKNN_OUT)
if ret != 0:
    print("❌ export 失败")
    exit(ret)

print("✅ 转换成功")
rknn.release()
```

**`calib_dataset.txt`**：量化校准集清单，每行一个图片路径。建议 200-500 张，覆盖白天/夜间/逆光/薄雾等场景。

---

## 四、第三步：判读结果（最关键的一步）

转换成功**不等于**可部署。必须检查以下三项：

### 检查 1：逐层分析报告

`rknn.build()` 会在日志中输出每层算子的分配情况，形如：

```
ID   OpType                 Target    ...
0    InputOperator          CPU
1    Conv                   NPU
2    Mul                    NPU
...
17   SomeCustomOp           CPU       ← ⚠️ 回退到 CPU
```

**判读标准**：

| 情况 | 判定 |
|---|---|
| 绝大多数算子在 NPU | ✅ 可用 |
| 少量边缘算子在 CPU（如 Input/Output/Reshape） | ✅ 正常，可接受 |
| **核心算子（Conv / 融合模块）在 CPU** | ❌ **不可用**，会成为性能瓶颈 |
| 出现 `Not supported` / `unsupported` | ❌ 需要改造 |

**重点关注**：融合模块、注意力模块所在层是否落在 NPU 上。

### 检查 2：量化精度损失

INT8 量化必然带来精度损失，需评估是否可接受：

```python
# 用同一批图，对比 FP32（PyTorch）与 INT8（RKNN）的输出差异
# 关注 mAP 下降幅度
```

**经验判据**：mAP 下降 < 2% 属正常；> 5% 需考虑敏感层保留 FP16（混合量化）。

### 检查 3：真机性能实测

**必须上真机**，仿真器的速度不代表实际：

```bash
# 在 RK3588 开发板上
./rknn_benchmark yolo11n-mm-xxx.rknn
```

关注输出中的：

| 指标 | 说明 |
|---|---|
| Total Time | 端到端推理耗时 |
| NPU 占用 | 应接近 100%，若低说明有算子跑 CPU |
| FPS | 是否满足实时要求（建议 ≥ 15 FPS） |

---

## 五、风险预判与应对

基于对两个自定义算子的分析，预判风险等级：

| 算子 | 风险 | 理由 | 应对 |
|---|---|---|---|
| `C3k2_PKIModule` | **中低** | 基于 C3k2 改造，主体是标准 Conv/Concat，若 PKI 部分能展开为标准算子则可转 | 优先展开为标准算子形式 |
| `Poly_Kernel_Inception_Block` | **中** | "多项式核"若用大卷积核或多分支 Inception 实现，需确认每支都是标准算子 | 同上 |

**通用应对策略（按优先级）**：

**其一，算子展开（首选）**。很多自定义模块在导出 ONNX 时会自动展开为标准算子组合（如多分支 Inception 展开为多个 Conv + Concat）。**先看 ONNX 算子清单**——如果已经全部是标准算子，风险基本消除。

**其二，结构替换**。若某模块确实无法转换，用一个**功能相近且 RKNN 支持**的模块替代。例如门控融合可换成 SE 式通道注意力（纯 Conv + Sigmoid + Mul，RKNN 完全支持）。

**其三，混合精度**。对不支持的少数层，配置为 FP16 执行（RKNN 支持混合量化），代价是略慢。

**其四，调整创新点落点**。若融合模块整体不可部署，可把创新点转向**训练策略**（模态 dropout、配准扰动增强、退化降级逻辑）——这些不改变推理图结构，天然兼容。

---

## 六、执行建议

### 时间安排

| 时机 | 动作 | 耗时 |
|---|---|---|
| **训练开始前** | 随机权重 + ONNX 导出 + 算子清单检查 | 2-4 h |
| **训练开始前** | RKNN 转换 + 逐层分析（三个配置都试） | 4-8 h |
| 训练完成后 | 用真实权重重跑一遍转换 + 量化精度评估 | 1-2 天 |
| 部署阶段 | 真机性能实测 + 调优 | 3-5 天 |

**第一、二项在训练前做**，与训练并行不冲突（训练跑在 GPU 上，转换验证用 CPU 即可）。

### 决策节点

```
算子清单全是标准算子？
  ├─ 是 → 直接做 RKNN 转换验证（半天）
  └─ 否 → 先解决算子展开问题，再验证

RKNN 转换成功且核心算子全在 NPU？
  ├─ 是 → ✅ 方案可行，放心训练
  └─ 否 → ⚠️ 立即调整模块选型，不要先训后改
```

---

## 七、与你项目当前状态的衔接

**关键提醒**：目前**还没有任何训练完成的多模态权重可用于导出验证**（唯一的 `best.pt` 只训了 2 epoch，但结构完整，可用于验证）。

实际上，**那个 2-epoch 的 `best.pt` 正好可以拿来做首次验证**——因为它已经是完整的 PKI 架构，结构上等价于训练完成的模型。**不必等训练跑完。**

```bash
# 用已有的 2-epoch 权重直接试导出
python -c "
from ultralytics import YOLOMM
m = YOLOMM('ResTest/yolomm-c3k2-pkimodule/weights/best.pt')
m.export(format='onnx', imgsz=640, opset=12)
"
```

这是**性价比最高的一步验证**——成本极低（几分钟），但能提前暴露"自定义算子能否导出"这个根本问题。

---

## 附：待确认事项

1. **多模态 ONNX 导出是否被框架支持**（双输入签名问题）——这是最优先要确认的
2. RKNN-Toolkit2 的具体版本（需与开发板运行时匹配）
3. 开发板是否有现成的 `librknnrt.so`，版本是多少
4. 校准集（200-500 帧）的准备方式——可从验证集抽样，注意覆盖昼夜

---

*本方法随验证推进持续更新；每次验证后记录算子清单与判定结论。*
