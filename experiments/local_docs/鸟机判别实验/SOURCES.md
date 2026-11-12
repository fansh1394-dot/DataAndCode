# 鸟机判别数据源登记（SOURCES）

更新时间：2026-11-10

原则：每个来源登记名称、URL、许可、引用、下载日期、校验和与**对齐性质**；许可不明的一律不进训练，只进内部压力测试并在论文标注。

## 1. AVID（主训练源）

| 项 | 内容 |
|---|---|
| 角色 | bird + drone 主训练源；小目标/弱对齐压力测试 |
| 本地路径 | `/root/rivermind-fs/AVID` |
| 规模 | 32 040 对帧、115 视频、35 952 框（bird 15 471 / drone 20 481） |
| 原生分辨率 | 320×256 |
| 划分 | 官方 Train 102 视频 / Val 13 视频；本项目在官方 Train 内按视频 80/20 分层生成 train/val，官方 Val **只作外部 test** |
| 标签 | YOLO `class cx cy w h`，`0=bird`、`1=drone`；每帧一套 RGB 坐标标签，**无独立 IR 框** |
| 对齐 | **弱对齐**：论文 32.3% 的配对全局位移 >5 px，16.3% 超过一个中位目标边长；对应帧不等于像素配准 |
| 本项目处理 | 已对 IR 做逐帧离线相位相关配准（±12 px 截断，见 §5）；`bd_nc2` 只用配准后的 IR，未配准版本仅保留作对齐消融/压力测试 |
| 许可 | CC BY-NC 4.0（源影像 CC0）；**仅非商业学术使用** |
| 引用 | Raghavan, Rao, Selvi. *AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets.* arXiv:2610.40949；Svanström, Alonso-Fernandez, Englund. *Drone Detection Dataset* |
| 校验和 | `AVID_MANIFEST.tsv` SHA-256 `b7e7ad7da9004db56055a742ae315954631136914b4498d30cfd0888bdb1c008` |
| 已知限制 | 0 鸟机同框、0 纯背景帧；39 段纯鸟 + 76 段纯机；68.1% 目标边长 8–16 px |

## 2. Mendeley EO/IR（UAV 护栏与背景）

| 项 | 内容 |
|---|---|
| 角色 | drone 正样本、背景帧、UAV 精度护栏、≥16 px 验收 |
| 本地路径 | `/3250410011/root/rivermind-data/mendeley_split` |
| 规模 | train 4 604（背景 629）/ val 616（背景 88）/ test 614（背景 66） |
| 对齐 | **像素对齐**；`labels_ir` 与 `labels` 逐框相同（5 834 帧中 783 帧两模态均为空、5 050 帧数值一致，仅 1 帧框数不一致，训练以 `labels` 为准） |
| 许可 | 项目已确认可用于本研究；具体条款以 Mendeley Data 页面为准，论文提交前复核 |
| 校验和 | `data.yaml` SHA-256 `7e3189ee322e160c2e75c2c4200ab6296e710aaaf36f7b39a3aee5d28bae97b5`；`split_stats.json` SHA-256 `0b7f7f60ba6c124112920a7e9311d346da9e8bbb745625c483e6a5558931f18a` |
| 已知限制 | 783 帧背景；99.3% 单目标帧；与 AVID 的对齐性质不同 |

## 3. 待接入来源

| 来源 | 角色 | 状态 | 许可 |
|---|---|---|---|
| Drone-vs-Bird Challenge（DVB 2019/2021） | RGB-only 补充、跨域 | 未下载 | 研究用途，逐条核 |
| Anti-UAV 系列 | 多目标/复杂场景补充 | 未下载 | 未核 |
| 自采（校园/公园） | 同框、背景、真实复杂场景、域内测试 | 待采集 | 自建，需拍摄记录与伦理说明 |

## 4. 混训与对齐规则（硬约束）

1. **只允许已对齐数据进入 6 通道检测训练**。AVID 的 IR 必须先离线配准到 RGB，才能与 Mendeley 混训；否则标签只对 RGB 成立。
2. 配准只作用于训练数据，**不改变部署链路**；部署端按相机自身标定处理。
3. 每个样本保留 `source` 标记；论文指标分源报告，禁止只报混合总数。
4. AVID 官方 Val 只能跑一次最终测试，**不得用于调参/选模**。
5. AVID 为 CC BY-NC 4.0，不得用于商业用途；论文中须同时引用 AVID 与原始 Drone Detection Dataset。
6. 任何新来源接入前，先补全本文件的许可、引用、校验和与对齐性质，否则不得进入训练。

## 5. AVID IR 离线配准与 bd_nc2 构建记录（2026-11-10）

**配准方法**：320×256 原生尺度灰度图 + Hann 窗，`cv2.phaseCorrelate` 估计全局平移；仅平移、不旋转/缩放；|shift|>12 px 判失败按 0 位移处理；输出 JPEG q95。
脚本：`scripts/make_avid_reg.py`；产物：`/root/rivermind-fs/birddrone/avid_ir_reg/{train,val}`、`avid_ir_reg/reg_stats.json`。

| 项 | 数值 |
|---|---|
| 配准帧数 | 32 040（train 28 790 / val 3 250） |
| 超 ±12 px 判失败 | 1 650（5.15%），保持 0 位移 |
| 全局位移中位数 | 0.56 px（p90 3.82 px） |
| 位移 >5 px 占比 | 7.07%（仅统计估计成功的帧） |
| 相位相关响应均值 | 0.279 |

与论文「32.3% >5 px」的差异来自统计量不同：论文是**目标级**位移，本表是**全图全局平移**；且超限帧按 0 位移保留，属残余风险，**不作为已对齐证据**，论文中须同时给出未配准对照（E30）。

**bd_nc2 数据集**（`/root/rivermind-fs/birddrone/bd_nc2`，扁平化，前缀 `avid__` / `men__`）：

| split | 总帧 | AVID（IR 已配准） | Mendeley | bird 框 | drone 框 |
|---|---|---|---|---|---|
| train | 27 658 | 23 054 | 4 604 | 11 413 | 18 663 |
| val | 6 352 | 5 736 | 616 | 2 700 | 4 211 |
| test | 3 864 | 3 250（官方 Val） | 614 | 1 358 | 2 698 |

- 统一 `nc=2`：`0=bird`、`1=drone`；Mendeley 原 `0=UAV` 全部 remap 为 `1`；
- 图像/IR/标签/IR 标签四者数量逐 split 相等，无缺配；
- **不采用**「AVID 未配准 IR 直接混 Mendeley」的做法；Mendeley 本身像素对齐，无需再配准；
- `data.yaml` 的 `align: false` 只表示来源对齐性质不统一，**不代表允许错位输入**。

## 6. other 类原始图源（2026-11-11）

用于「气球/风筝/飞机」等非鸟非机目标的抠图与合成注入，**仅作训练增强来源**，不作为 other 的真实评测集。

| 来源 | 内容 | 许可 | 用途 |
|---|---|---|---|
| COCO val2017（via HF `LibreYOLO/coco-val2017`） | kite 91 图 / 327 实例；airplane 97 图 / 143 实例（YOLO 框） | 标注 CC BY 4.0；图像来自 Flickr，各自许可 | 抠图后 copy-paste 合成 other 帧；论文须引用 Lin et al., ECCV 2014 |
| Roboflow balloon（via HF `nielsr/balloon`） | 74 图 / 305 实例（多边形掩膜），实为**派对气球**而非热气球 | 见 HF dataset card（Roboflow public dataset） | 气球抠图；多数为手持/室内，剪出后仅用目标像素 |

限制：

1. 气球图多为手持或室内背景，直接整图当“空中气球”会引入域偏差，必须抠图后贴到天空背景；
2. COCO 风筝常为远景小目标，尺度与 AVID 接近，适合直接注入；
3. 这批图**不能**用于 other 的最终召回评测，真实评测需另采 100–200 帧（哪怕手机拍摄）；
4. 论文须声明 other 训练集为合成（synthetic），并给出合成比例与注入参数。

### 6.1 合成注入产物 bd_nc3（2026-11-11）

- 抠图：`other_cutouts/` 共 763 个 RGBA（balloon 305 用多边形掩膜；kite 315、airplane 143 用 GrabCut + 填洞 + alpha 归一化）；
- 背景：只用 AVID 视频，**只读**；train/val 背景视频来自不同 split，无泄漏；
- 抠图按 8:2 分 train/val，互不重叠；
- 注入规则：天空掩膜内、与已有 GT 框 IoU<0.1、等效边长 10–42 px、气球 0.2–0.6 px/帧、风筝/飞机 0.6–2.0 px/帧、±6° 旋转、±10% 亮度、速度>1 时加运动模糊；
- 规模：train 21 582 帧（基础 18 416 + 合成 3 166，合成占 **14.7%**；class2 框 4 611）；val 1 471 帧（基础 1 232 + 合成 239；class2 框 334）；
- 标签：`0=bird`、`1=drone`、`2=other`；合成帧的 IR 为 RGB 灰度复制（`ir=fallback`），labels_ir 与 labels 相同；
- 完整性：列出的图像/IR/标签缺失均为 **0**（`check_bd_nc3.py`）；
- 原始 AVID 与项目代码均未修改；全部新脚本位于 `/root/rivermind-fs/birddrone/`。
