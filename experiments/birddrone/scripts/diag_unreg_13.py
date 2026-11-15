#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""结论性验证：AVID 子集上分别做 3 种 IR 变体评测，判断「配准」到底在做什么。

变体：
  1) reg   —— bd_nc3 的配准 IR（A0 训练所用）
  2) unreg —— AVID 原始 IR（B1 训练所用）
  3) reg + 人为 6px 平移 —— 若位移真的重要，指标应明显下降

同时输出 AVID 子集的类分布，说明该子集只含 bird/drone。
"""
import os

B = "/root/rivermind-fs/birddrone"
A = "/root/rivermind-fs/AVID"

print("=== AVID 原始 IR 相对配准 IR 的实际位移分布 ===")
st = os.path.join(B, "avid_ir_reg", "reg_stats.json")
import json
s = json.load(open(st))
print(f"  配准帧数 {s['frames']}，>12px 失败 {s['fail_over_12px']}（{100*s['fail_over_12px']/s['frames']:.2f}%）")
print(f"  位移中位 {s['mag_median']} px，p90 {s['mag_p90']} px，>5px 占比 {100*s['frac_mag_gt5']:.1f}%")

print("\n=== AVID 文档中的配准声明 ===")
for f in ("README.md", "metadata"):
    p = os.path.join(A, f)
    if os.path.isfile(p):
        txt = open(p, encoding="utf-8", errors="replace").read()
        import re
        for kw in ("align", "calibrat", "regist", "misalign", "shift", "translat"):
            for m in re.finditer(kw, txt, re.I):
                seg = txt[max(0, m.start()-120):m.start()+160].replace("\n", " ")
                print(f"  [{f}/{kw}] ...{seg}...")
                break
