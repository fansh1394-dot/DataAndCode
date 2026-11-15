#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用各 run 的 best.pt 在 bd_nc3 val 上统一评测（eval_split，dual 口径）。"""
import os
import subprocess

R = "/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest"
PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
PY = "/opt/conda/envs/rivermind/bin/python"
OUT = "/root/rivermind-fs/birddrone/batch1/eval_avidsrc"
DATA = "/root/rivermind-fs/birddrone/bd_nc3/data_nc3.yaml"

RUNS = {
    "A0_nc3_s0": "bdc3_nc3_s0_150",
    "B1_unreg_s0": "bdc3_nc3_unreg_s0",
    "C1_nc3_s1": "bdc3_nc3_s1",
    "A1_rgbonly": "bdc3_rgbonly_yolo11n_s0",
    "A2_ironly": "bdc3_ironly_yolo26n_s0",
}

os.makedirs(OUT, exist_ok=True)
for tag, name in RUNS.items():
    w = os.path.join(R, name, "weights", "best.pt")
    if not os.path.exists(w):
        print(f"[skip] {tag} 无 best.pt", flush=True); continue
    dst = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(dst):
        print(f"[skip] {tag} 已有结果", flush=True); continue
    cmd = [PY, "scripts/eval_split.py", "--weights", w, "--data", DATA,
           "--split", "val", "--labels-mode", "dual", "--imgsz", "640",
           "--batch", "16", "--tag", f"b1_{tag}", "--out", dst]
    print("[run]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=PROJ, env={**os.environ, "PYTHONPATH": "."},
                       capture_output=True, text=True)
    print(r.stdout[-2500:], flush=True)
    if r.returncode != 0:
        print("[stderr]", r.stderr[-1500:], flush=True)
print("ALL EVAL DONE", flush=True)
