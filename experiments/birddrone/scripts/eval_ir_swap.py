#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决定性对照：同一模型 A0 在 AVID 子集上，IR 分别用「配准版」与「未配准版」评测。

若配准真的有益，A0 用配准 IR 评测应优于用未配准 IR；反之说明配准无正贡献。
做法：构造两套 val 数据集目录（只对 AVID 帧建立 IR 变体），各评一次。
"""
import json
import os
import shutil
import subprocess

B = "/root/rivermind-fs/birddrone"
PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
PY = "/opt/conda/envs/rivermind/bin/python"
W = os.path.join(PROJ, "YOLO26/ResTest/bdc3_nc3_s0_150/weights/best.pt")
OUT = os.path.join(B, "batch1", "eval_swap")
os.makedirs(OUT, exist_ok=True)

# 用 bd_nc3 作为基底（RGB/标签与 unreg 相同），只替换 AVID 帧的 IR 来源
src = os.path.join(B, "bd_nc3")
unreg_ir = os.path.join(B, "bd_nc3_unreg", "images_ir")

for variant in ("reg", "unreg"):
    dst = os.path.join(B, f"bd_nc3_valIR_{variant}")
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
        for sub in ("images", "labels", "labels_ir"):
            for split in ("train", "val"):
                s = os.path.join(src, sub, split)
                if os.path.exists(s):
                    os.makedirs(os.path.join(dst, sub, split), exist_ok=True)
        # 只建 val 的软链接（评测只需要 val）
        for sub in ("images", "labels", "labels_ir"):
            d = os.path.join(dst, sub, "val")
            for f in os.listdir(os.path.join(src, sub, "val")):
                p = os.path.join(src, sub, "val", f)
                t = os.path.join(d, f)
                if not os.path.exists(t):
                    os.symlink(p, t)
        d_ir = os.path.join(dst, "images_ir", "val")
        os.makedirs(d_ir, exist_ok=True)
        for f in os.listdir(os.path.join(src, "images_ir", "val")):
            base = src if variant == "reg" else None
            if variant == "reg":
                p = os.path.join(src, "images_ir", "val", f)
            else:
                p = os.path.join(unreg_ir, "val", f)
            t = os.path.join(d_ir, f)
            if not os.path.exists(t) and os.path.exists(p):
                os.symlink(p, t)
        for f in ("val_nc3.txt", "val_avid.txt", "val_men.txt", "val_syn.txt"):
            s = os.path.join(src, f)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(dst, f))
        y = open(os.path.join(src, "data_nc3.yaml"), encoding="utf-8").read().replace(src, dst)
        open(os.path.join(dst, f"data_{variant}.yaml"), "w", encoding="utf-8").write(y)
        print(f"[build] {dst}")

for variant in ("reg", "unreg"):
    dst_json = os.path.join(OUT, f"A0_valIR_{variant}.json")
    if os.path.exists(dst_json):
        print(f"[skip] {variant}"); continue
    py = os.path.join(B, f"bd_nc3_valIR_{variant}", f"data_{variant}.yaml")
    cmd = [PY, "scripts/eval_split.py", "--weights", W, "--data", py,
           "--split", "val", "--val-list", "val_avid.txt", "--labels-mode", "dual",
           "--imgsz", "640", "--batch", "16",
           "--tag", f"swap_{variant}", "--out", dst_json]
    r = subprocess.run(cmd, cwd=PROJ, env={**os.environ, "PYTHONPATH": "."},
                       capture_output=True, text=True)
    print(f"[{variant}] rc={r.returncode}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-800:], flush=True)
print("DONE")
