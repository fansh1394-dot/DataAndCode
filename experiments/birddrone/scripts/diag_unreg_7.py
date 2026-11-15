#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐条比对 A0（配准）与 B1（未配准）的 args.yaml，找出除数据外的一切差异。"""
import os
import yaml

R = "/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest"
A0 = os.path.join(R, "bdc3_nc3_s0_150", "args.yaml")
B1 = os.path.join(R, "bdc3_nc3_unreg_s0", "args.yaml")

a = yaml.safe_load(open(A0, encoding="utf-8"))
b = yaml.safe_load(open(B1, encoding="utf-8"))
keys = sorted(set(a) | set(b))
diffs = 0
for k in keys:
    va, vb = a.get(k, "<缺失>"), b.get(k, "<缺失>")
    if va != vb:
        diffs += 1
        print(f"{k}:\n    A0={va}\n    B1={vb}")
print(f"\n共 {diffs} 处不同（含 data 路径）")
