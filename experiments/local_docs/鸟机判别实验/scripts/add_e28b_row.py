# -*- coding: utf-8 -*-
"""在实验追踪表追加 E28b，并扩展自动筛选范围。"""
import openpyxl
from copy import copy

p = r"c:\Users\不如喝茶去\Desktop\双光谱毕设\待做实验清单.xlsx"
wb = openpyxl.load_workbook(p)
ws = wb["实验追踪"]

src = 32  # E28 行，用于复制样式
vals = [
    "E28b", "高", "鸟机判别·检测器",
    "nc=2 平衡微调（AVID:Mendeley≈1:1）",
    "E28 的 UAV 回退源于 AVID 小目标主导，平衡配比应能恢复 Mendeley 精度",
    "E28 best 续训；train_bal 4 604 AVID + 4 604 Mendeley；val_bal 616+616；lr0 0.002；40 ep/patience 15",
    "mdp150_M0BD_nc2_bal_s0（E28 best 初始化）",
    "Mendeley val mAP50-95 ≥0.762（M0−1 pp）且 AVID bird mAP50-95 ≥0.40",
    "中（GPU，约 1/3 E28 时长）",
    "进行中",
    "训练中：train 9 208 帧（AVID 抽样 4 604，bird 2 154 / drone 3 018 框；Mendeley 4 604）；batch 32、21.9 GB、288 it/epoch",
    "选模用平衡 val，避免 AVID 主导；若护栏仍不过，则改用两阶段（AVID→Mendeley 单域收尾）",
    "2026-11-11",
]
row = ws.max_row + 1
for c, v in enumerate(vals, start=1):
    cell = ws.cell(row, c)
    cell.value = v
    base = ws.cell(src, c)
    cell.font = copy(base.font)
    cell.fill = copy(base.fill)
    cell.alignment = copy(base.alignment)
    cell.border = copy(base.border)

ws.auto_filter.ref = f"A1:M{row}"
wb.save(p)
print("added row", row, "filter", ws.auto_filter.ref)
