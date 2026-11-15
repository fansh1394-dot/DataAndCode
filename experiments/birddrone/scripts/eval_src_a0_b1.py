#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分源评测：把 val 拆成 AVID / Mendeley / 合成 三个子集，分别评测 A0 与 B1。"""
import json
import os
import subprocess

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
PY = "/opt/conda/envs/rivermind/bin/python"
B = "/root/rivermind-fs/birddrone"
R = os.path.join(PROJ, "YOLO26/ResTest")
OUT = os.path.join(B, "batch1", "eval_src")
DATA3 = os.path.join(B, "bd_nc3", "data_nc3.yaml")
DATAU = os.path.join(B, "bd_nc3_unreg", "data_unreg.yaml")
RUNS = {"A0": (os.path.join(R, "bdc3_nc3_s0_150/weights/best.pt"), DATA3),
        "B1": (os.path.join(R, "bdc3_nc3_unreg_s0/weights/best.pt"), DATAU)}

os.makedirs(OUT, exist_ok=True)
# 生成三个子集的图像列表（相对 dataset path 的路径）
for root in ("bd_nc3", "bd_nc3_unreg"):
    lines = [l.strip() for l in open(os.path.join(B, root, "val_nc3.txt")) if l.strip()]
    groups = {"avid": [], "men": [], "syn": []}
    for l in lines:
        nm = os.path.basename(l)
        if "__syn" in nm:
            groups["syn"].append(l)
        elif nm.startswith("avid__"):
            groups["avid"].append(l)
        elif nm.startswith("men__"):
            groups["men"].append(l)
    for k, v in groups.items():
        p = os.path.join(B, root, f"val_{k}.txt")
        open(p, "w").write("\n".join(v) + "\n")
        print(f"{root}/val_{k}.txt : {len(v)} 帧")

for tag, (w, data) in RUNS.items():
    for grp in ("avid", "men", "syn"):
        dst = os.path.join(OUT, f"{tag}_{grp}.json")
        if os.path.exists(dst):
            continue
        root = "bd_nc3" if tag == "A0" else "bd_nc3_unreg"
        cmd = [PY, "scripts/eval_split.py", "--weights", w, "--data", data,
               "--split", "val", "--val-list", f"val_{grp}.txt",
               "--labels-mode", "dual", "--imgsz", "640", "--batch", "16",
               "--tag", f"src_{tag}_{grp}", "--out", dst]
        r = subprocess.run(cmd, cwd=PROJ, env={**os.environ, "PYTHONPATH": "."},
                           capture_output=True, text=True)
        print(f"[{tag}/{grp}] rc={r.returncode}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-600:], flush=True)
print("DONE")
