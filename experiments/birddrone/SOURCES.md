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
| 本项目处理 | E28 前对 IR 做离线相位相关配准，逐帧记录 shift；未配准版本只用于对齐消融与压力测试 |
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
| 对齐 | **像素对齐**；`labels_ir` 与 `labels` 逐框相同，两模态共用一套标签 |
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
