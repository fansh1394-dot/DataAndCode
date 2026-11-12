# -*- coding: utf-8 -*-
"""Export per-frame fuse boxes under IR-feature attenuation ALPHA at the fusion concat."""
import os
import sys
import json

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import torch
from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.mm import dual_val as dv

ALPHA = float(os.environ.get("ALPHA", "1.0"))
LEVEL = os.environ.get("LEVEL", "0")

_fwd_orig = Concat.forward


def _fwd(self, x):
    y = _fwd_orig(self, x)
    a = getattr(self, "_mm_alpha", 1.0)
    if a != 1.0 and self.d == 1:
        h = y.shape[1] // 2
        y = torch.cat([y[:, :h], y[:, h:] * a], dim=1)
    return y


Concat.forward = _fwd

REC = {}
_bm_orig = dv.DualHeadDetectionValidator._update_branch_metrics


def _bm(self, name, preds, batch):
    if name == "fuse":
        files = batch.get("im_file")
        if files is None:
            files = batch.get("img_path")
        for i, p in enumerate(preds):
            f = str(files[i]) if files is not None and i < len(files) else f"idx{i}"
            REC[os.path.basename(f)] = {
                "xyxy": p["bboxes"].detach().cpu().tolist() if p.get("bboxes") is not None else [],
                "conf": p["conf"].detach().cpu().tolist() if p.get("conf") is not None else [],
                "cls": p["cls"].detach().cpu().tolist() if p.get("cls") is not None else [],
            }
    return _bm_orig(self, name, preds, batch)


dv.DualHeadDetectionValidator._update_branch_metrics = _bm

_init_orig = dv.DualHeadDetectionValidator.init_metrics


def _init(self, model):
    layers = list(getattr(model, "model", []))
    marked = []
    for i, lay in enumerate(layers):
        if not isinstance(lay, Concat):
            continue
        f = getattr(lay, "f", None)
        if isinstance(f, (list, tuple)) and len(f) == 2 and all(
            isinstance(t, int) and t >= 0 for t in f
        ):
            lay._mm_alpha = ALPHA
            marked.append(i)
    if len(marked) != 3:
        raise RuntimeError(f"expected 3 fusion concat nodes, got {marked}")
    return _init_orig(self, model)


dv.DualHeadDetectionValidator.init_metrics = _init

import eval_split as es

OUT = os.path.join(PROJ, "alpha_boxes")
os.makedirs(OUT, exist_ok=True)
extra = [] if LEVEL == "0" else ["--ir-shift", f"{LEVEL},0"]
sys.argv = [
    "eval_split.py",
    "--weights", "YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt",
    "--data", "/3250410011/root/rivermind-data/mendeley_split/data.yaml",
    "--split", "test", "--labels-mode", "dual",
    "--imgsz", "640", "--batch", "8", "--workers", "4", "--device", "0",
    "--tag", f"ab_a{ALPHA}_l{LEVEL}",
    "--out", os.path.join(OUT, f"_eval_a{ALPHA}_l{LEVEL}.json"),
] + extra

rc = 0
try:
    rc = es.main()
except SystemExit as e:
    rc = e.code
finally:
    with open(os.path.join(OUT, f"boxes_a{ALPHA}_l{LEVEL}.json"), "w") as f:
        json.dump(REC, f)
    print(f"[alpha-boxes] alpha={ALPHA} level={LEVEL} frames={len(REC)} rc={rc}", flush=True)
