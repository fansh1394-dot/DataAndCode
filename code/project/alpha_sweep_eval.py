# -*- coding: utf-8 -*-
"""Test-time IR feature attenuation at the fusion concat (no retraining).

Scales the IR half of each fusion Concat by ALPHA, then runs the normal
dual-head test evaluation. Answers: is there headroom for misalignment-aware
gating of the IR branch?
"""
import os
import sys

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import torch
from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.mm import dual_val as dv

ALPHA = float(os.environ.get("ALPHA", "1.0"))
LEVEL = os.environ.get("LEVEL", "0")
TAG = os.environ.get("TAG", "alpha")

_fwd_orig = Concat.forward


def _fwd(self, x):
    y = _fwd_orig(self, x)
    a = getattr(self, "_mm_alpha", 1.0)
    if a != 1.0 and self.d == 1:
        h = y.shape[1] // 2
        y = torch.cat([y[:, :h], y[:, h:] * a], dim=1)
    return y


Concat.forward = _fwd

_init_orig = dv.DualHeadDetectionValidator.init_metrics


def _mark(model):
    layers = list(getattr(model, "model", []))
    marked = []
    for i, lay in enumerate(layers):
        if not isinstance(lay, Concat):
            continue
        f = getattr(lay, "f", None)
        # Neck concats are [-1, k]; cross-modal fusion concats are [i, j] with i, j >= 0.
        if isinstance(f, (list, tuple)) and len(f) == 2 and all(
            isinstance(t, int) and t >= 0 for t in f
        ):
            lay._mm_alpha = ALPHA
            marked.append(i)
    print(f"[alpha] marked fusion concat idx={marked} alpha={ALPHA}", flush=True)
    if len(marked) != 3:
        raise RuntimeError(f"expected 3 fusion concat nodes, got {len(marked)}: {marked}")


def _init(self, model):
    _mark(model)
    return _init_orig(self, model)


dv.DualHeadDetectionValidator.init_metrics = _init

import eval_split as es

OUT = os.path.join(PROJ, "ResTest", "_test_eval", "alpha_sweep")
os.makedirs(OUT, exist_ok=True)
extra = [] if LEVEL == "0" else ["--ir-shift", f"{LEVEL},0"]
sys.argv = [
    "eval_split.py",
    "--weights", "YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt",
    "--data", "/3250410011/root/rivermind-data/mendeley_split/data.yaml",
    "--split", "test", "--labels-mode", "dual",
    "--imgsz", "640", "--batch", "8", "--workers", "4", "--device", "0",
    "--tag", f"{TAG}_a{ALPHA}_l{LEVEL}",
    "--out", os.path.join(OUT, f"a{ALPHA}_l{LEVEL}.json"),
] + extra

rc = 0
try:
    rc = es.main()
except SystemExit as e:
    rc = e.code
print(f"[alpha] done alpha={ALPHA} level={LEVEL} rc={rc}", flush=True)
