# -*- coding: utf-8 -*-
"""回写 E26/E27/E28 状态与结果，并统一 IR 口径描述。"""
from pathlib import Path

import openpyxl
from copy import copy

p = Path(r"c:\Users\不如喝茶去\Desktop\双光谱毕设\待做实验清单.xlsx")
wb = openpyxl.load_workbook(p)
ws = wb["实验追踪"]

# 沿用「已完成」单元格样式
done_src = ws["K29"]
run_src = None
for r in range(2, ws.max_row + 1):
    if ws.cell(r, 10).value == "进行中":
        run_src = ws.cell(r, 10)
        break
if run_src is None:
    run_src = done_src

rows = {
    30: {
        "status": ("已完成", done_src),
        "result": ("split.json：train 82 视频 / 23 054 帧（bird 11 413、drone 14 655）、"
                   "val 20 视频 / 5 736 帧（2 700 / 3 680）、test 13 视频 / 3 250 帧（1 358 / 2 146）；"
                   "SOURCES.md 已登记许可、校验和与对齐性质"),
        "conclusion": ("按视频划分、无跨集，官方 Val 只作 test；32 040 对全部配对，PASS；"
                       "同框与背景帧仍需自采"),
    },
    31: {
        "status": ("已完成", done_src),
        "result": ("640：IoU0.3 提议召回 55.4%（conf≥0.01 时 40.4%）、IoU0.5 仅 16.6%；"
                   "736：54.2% / 37.9%，无增益；24% 帧零提议；目标中位 12 px"),
        "conclusion": ("未达 0.90 门禁（差约 35 pp），R1 级联不可行，只走 R2；"
                       "详见 reports/E27_D0_鸟提议召回.md"),
    },
    32: {
        "status": ("进行中", run_src),
        "result": ("bd_nc2：train 27 658 / val 6 352 / test 3 864（四类文件逐 split 等量）；"
                   "AVID 32 040 帧 IR 已离线配准（5.15% 超 ±12 px 置零，中位位移 0.56 px）；"
                   "nc=2 warm start，batch 16、11.2 GB、1 729 it/epoch、约 2.4 it/s、150 ep"),
        "conclusion": ("seed0 训练中（日志 E28/e28_train.log）；UAV 护栏失败则回退 R1 级联"),
    },
}
for r, v in rows.items():
    ws.cell(r, 10).value = v["status"][0]
    ws.cell(r, 10).fill = copy(v["status"][1].fill)
    ws.cell(r, 10).font = copy(v["status"][1].font)
    ws.cell(r, 10).alignment = copy(v["status"][1].alignment)
    ws.cell(r, 11).value = v["result"]
    ws.cell(r, 12).value = v["conclusion"]
    ws.cell(r, 13).value = "2026-11-10"

# 鸟机判别页：IR 口径与最终处理保持一致
ws2 = wb["鸟机判别"]
for r in range(1, ws2.max_row + 1):
    val = ws2.cell(r, 1).value
    if isinstance(val, str) and val.startswith("IR："):
        ws2.cell(r, 1).value = (
            "IR：AVID 为弱对齐（论文 32.3% 全局位移 >5 px）。6 通道训练前先做逐帧离线相位相关配准"
            "（±12 px 截断，5.15% 超限置零，中位 0.56 px）；部署端不做配准，按相机自身标定。"
            "未配准版本只留作 E30 消融。"
        )

wb.save(p)
print("updated", p)
