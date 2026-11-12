# -*- coding: utf-8 -*-
"""生成鸟机判别实验记录.xlsx（安排 + 结果 + 环境）。"""
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

OUT = Path(r"c:\Users\不如喝茶去\Desktop\双光谱毕设\鸟机判别实验\实验记录.xlsx")
wb = openpyxl.Workbook()

HDR = PatternFill("solid", fgColor="DDEBF7")
RUN = PatternFill("solid", fgColor="FFF2CC")
DONE = PatternFill("solid", fgColor="E2EFDA")

# ---------------- sheet 1 ----------------
ws = wb.active
ws.title = "实验安排与结果"
cols = ["批次", "编号", "实验", "目的", "初始化", "数据", "预算(ep)/关键参数", "判定线", "状态", "结果", "结论"]
ws.append(cols)
rows = [
    ("已完成", "E27", "D0 鸟提议召回", "现有 nc=1 检测器能否作鸟类提议器", "M0 concat", "AVID val 4 段鸟视频 1 104 帧",
     "640/736，IoU0.3", "≥0.90 保留 R1", "已完成", "640 55.4% / 736 54.2%", "未过，R1 级联不可行，只走 R2"),
    ("已完成", "E28", "nc=2 主训练 seed0", "二类检测器 warm start", "M0 concat", "bd_nc2 27 658 train",
     "150ep→早停59，batch32", "Mendeley ≥0.762、bird ≥0.40", "已完成", "Mendeley 0.672、bird 0.412", "UAV 回退 10 pp，需重配比"),
    ("已完成", "E28b", "1:1 平衡微调", "恢复 UAV 精度", "E28 best", "train_bal 9 208（AVID 4 604 + Men 4 604）",
     "40ep，lr0 0.002", "Mendeley ≥0.762", "已完成", "Mendeley 0.716、bird 0.403", "差 4.6 pp，继续加 Mendeley"),
    ("已完成", "E28c", "3:1 收尾", "进一步恢复 UAV", "E28b last", "train_3to1 18 416",
     "25ep，batch24，lr0 0.001", "Mendeley ≥0.762、bird ≥0.40", "已完成", "Mendeley 0.739、bird 0.380", "差 2.3/2.0 pp，为最佳折中"),
    ("已完成", "E36", "三类检测器", "加入 other 抑制气球误报", "E28c 扩头 nc3_init", "bd_nc3 21 582（合成 14.7%）",
     "20ep，batch32，lr0 0.002", "other→drone ≤2%、bird ≥0.40", "已完成", "other→drone 0.3%、other recall 91.3%、Mendeley 0.724、bird 0.388", "三类成立；bird→drone 升至 2.74%"),
    ("已完成", "E32", "阈值标定 + 轨迹告警", "确定部署阈值", "E36 best", "AVID val 轨迹 + 合成 other",
     "K=5/N=3，τ 0.2–0.9", "A 级 0 误告警、drone 100%", "已完成", "τ=0.5：A 级 8/8、bird 0/9、other 0/27", "推荐 τ=0.5 + <16px 门控"),
    ("已完成", "E33", "RKNN int8 转换", "端侧模型", "E36 best", "校准 300 帧（AVID/Men/合成）",
     "RK3588，rgb/ir 双输入", "cos ≥0.99", "进行中（模型就绪）", "cos 0.99641，8.0 MB", "待板端适配与联调"),
    ("第一批", "A1", "RGB-only（YOLO11n）", "主流单模态基线", "yolo11n.pt COCO", "bd_nc3",
     "80", "同口径对比 mAP", "排队", "—", "—"),
    ("第一批", "A2", "IR-only（YOLO26n）", "单模态 IR 分母", "随机初始化", "bd_nc3",
     "80", "同口径对比 mAP", "排队", "—", "—"),
    ("第一批", "B1", "IR 配准消融", "配准是否必要", "nc3_init", "bd_nc3_unreg（未配准 IR）",
     "40", "与 E36 对比", "排队", "—", "—"),
    ("第一批", "C1", "nc=3 seed1", "稳健性", "nc3_init", "bd_nc3",
     "40", "与 E36 差 ≤1 pp", "排队", "—", "—"),
    ("第二批", "B2", "other 比例 0%（nc=2，合成帧空标签）", "合成 other 是否必要", "nc3_init 扩 2 类", "bd_nc3 帧（去掉 class2 标签）",
     "40", "与 E36 other 指标对比", "排队", "—", "—"),
    ("第二批", "B3", "other 比例 30%（nc=3）", "合成比例敏感性", "nc3_init", "bd_nc3 + 额外合成帧（比例 30%）",
     "40", "other→drone ≤2%、bird/drone 不降", "排队", "—", "—"),
    ("第二批", "B4", "融合结构 gated（bd_nc3）", "融合方式对比", "nc3_init", "bd_nc3",
     "40", "与 concat 对比", "排队", "—", "—"),
    ("第二批", "B5", "融合结构 cga3h（bd_nc3）", "融合方式对比", "nc3_init", "bd_nc3",
     "40", "与 concat 对比", "排队", "—", "—"),
    ("第二批", "B6", "融合结构 cmce（bd_nc3）", "融合方式对比", "nc3_init", "bd_nc3",
     "40", "与 concat 对比", "排队", "—", "—"),
    ("第三批", "A3", "主流双模态 DEYOLO/ICAFusion", "外部方法对照", "官方权重/论文设置", "bd_nc3",
     "80", "同口径 mAP/参数量/时延", "待适配", "—", "需 1–2 天代码适配"),
    ("第三批", "C2", "nc=3 seed2", "稳健性（三 seed）", "nc3_init", "bd_nc3",
     "40", "与 E36/seed1 差 ≤1 pp", "排队", "—", "—"),
    ("第三批", "C3", "跨域 Men→AVID", "跨域泛化", "nc3_init", "仅 Mendeley 训练，AVID 测试",
     "40", "AVID bird/drone 召回", "排队", "—", "—"),
    ("第四批", "D1", "int8 vs fp16 vs fp32（板端）", "量化掉点与时延", "E36 best", "A/B 集",
     "0", "分类掉点 ≤1 pp、E2E ≤120 ms", "待硬件", "—", "需相机/板子"),
    ("第四批", "D2", "640 vs 736（板端）", "输入分辨率取舍", "E36 best", "A/B 集",
     "0", "精度/时延权衡", "待硬件", "—", "—"),
    ("第四批", "D3", "阈值/尺寸门控敏感性", "部署参数敏感性", "E36 best", "AVID 轨迹 + 合成 other",
     "0", "误告警 0、漏告警 0", "部分完成", "E32 已扫 τ 0.2–0.9", "可补 0.45/0.55 细扫"),
]
for r in rows:
    ws.append(r)
    fill = DONE if r[8].startswith("已完成") else (RUN if r[8] in ("排队", "进行中（模型就绪）") else None)
    for c in range(1, len(cols) + 1):
        cell = ws.cell(ws.max_row, c)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if fill:
            cell.fill = fill
for c in range(1, len(cols) + 1):
    ws.cell(1, c).font = Font(bold=True)
    ws.cell(1, c).fill = HDR
    ws.cell(1, c).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
widths = [8, 8, 22, 26, 18, 30, 24, 26, 14, 40, 34]
for i, w in enumerate(widths, start=1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.freeze_panes = "A2"
ws.auto_filter.ref = f"A1:K{ws.max_row}"

# ---------------- sheet 2 ----------------
ws2 = wb.create_sheet("结果对比")
ws2.append(["模型", "类别数", "Mendeley val 50-95", "AVID bird 50-95", "AVID drone 50-95", "other 50-95", "备注"])
data = [
    ("M0（UAV 基线）", 1, 0.772, "—", "—", "—", "同口径 Mendeley val 616"),
    ("E28 best", 2, 0.654, 0.412, 0.246, "—", "混合训练 150ep 早停"),
    ("E28b last", 2, 0.716, 0.403, 0.260, "—", "1:1 平衡 40ep"),
    ("E28c last", 2, 0.739, 0.380, 0.262, "—", "3:1 收尾 25ep"),
    ("E36 best", 3, 0.724, 0.388, 0.251, 0.660, "三类；other 为合成 val"),
]
for d in data:
    ws2.append(list(d))
for c in range(1, 8):
    ws2.cell(1, c).font = Font(bold=True)
    ws2.cell(1, c).fill = HDR
    ws2.column_dimensions[get_column_letter(c)].width = [22, 8, 20, 18, 18, 12, 30][c - 1]
ws2.freeze_panes = "A2"

# ---------------- sheet 3 ----------------
ws3 = wb.create_sheet("环境与路径")
ws3.append(["项", "值"])
env = [
    ("服务器", "root@10.252.18.70 -p 30775"),
    ("项目根", "/3250410011/root/rivermind-data/MutilModel_3485988898"),
    ("实验目录", "/root/rivermind-fs/birddrone"),
    ("数据集", "/root/rivermind-fs/birddrone/bd_nc2（2类）、bd_nc3（3类）、bd_nc3_unreg（消融）、avid_flat（E27）"),
    ("权重", "mdp150_M0BD_nc2_3to1_s0（E28c）、mdp150_M0BD_nc3_s0（E36）、nc3_init.pt"),
    ("端侧", "deploy_nc3/model_nc3_rk3588.rknn（8.0 MB, cos 0.99641）"),
    ("解释器", "/opt/conda/envs/rivermind/bin/python，需 PYTHONPATH=."),
    ("训练入口", "train_v26.py（YOLO26）；train_full.py（YOLO11 等通用）"),
    ("日志", "/root/rivermind-fs/birddrone/batch1/*.log"),
]
for e in env:
    ws3.append(list(e))
for c in (1, 2):
    ws3.cell(1, c).font = Font(bold=True)
    ws3.cell(1, c).fill = HDR
ws3.column_dimensions["A"].width = 14
ws3.column_dimensions["B"].width = 110

wb.save(OUT)
print("saved", OUT)
