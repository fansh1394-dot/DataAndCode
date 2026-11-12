# 双光谱低慢小目标（鸟/无人机）检测 —— 数据与代码

本仓库汇总毕业设计「面向边缘部署的双光谱低慢小无人机检测」的**代码、实验脚本、训练结果与文档**。
数据集（AVID、Mendeley、bd_nc2/bd_nc3 等）体积过大且受原始许可约束，**不在仓库内**。

## 目录

| 路径 | 说明 |
|---|---|
| `code/project/` | 项目根代码：训练入口 `train_v26.py`（YOLO26）、`train_full.py`（通用/YOLO11）、评测与工具脚本、docs、reports |
| `code/ultralytics/` | 改造后的 ultralytics 框架（双光谱分支、MM 配置、损失与融合模块） |
| `experiments/birddrone/` | 鸟机判别实验：数据构建、阈值标定、分源评测脚本；报告；配置；端侧模型（onnx/rknn） |
| `experiments/logs/` | 第一批 150 轮训练日志 |
| `experiments/local_docs/` | 本地文档：实验记录、交接文档、方案、开题/汇报 PPT、文献笔记 |
| `results/yolo26_restest/` | YOLO26 各次训练（results.csv、args.yaml、曲线图、metrics、best/last 权重） |
| `results/yolo11_restest/` | YOLO11 各次训练（同上） |

## 说明

- 每组结果保留 `best.pt` / `last.pt` 与全部指标文件；中间 `epoch*.pt` 未纳入。
- GitHub 单文件上限 100MB，个别超限权重（见 `oversized.txt`）未上传，仍保留在服务器
  `/root/rivermind-fs/birddrone/upload_repo/` 对应路径。
- 训练口径：对比/消融统一 150 epoch、patience 30；E28b/E28c/E36 为早期历史口径。
