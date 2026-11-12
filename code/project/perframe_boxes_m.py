# -*- coding: utf-8 -*-
"""Export per-frame NMS boxes for RGB/IR/fusion heads of an arbitrary checkpoint."""
import sys, os, json
PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))
from ultralytics.nn.mm import dual_val as dv

TAG = os.environ.get("TAG", "model")
WEIGHTS = os.environ["WEIGHTS"]
LEVEL = os.environ.get("LEVEL", "0")
REC = {}
_orig = dv.DualHeadDetectionValidator._update_branch_metrics


def as_list(x):
    if x is None:
        return []
    if hasattr(x, "detach"):
        return x.detach().cpu().tolist()
    return x


def patched(self, name, preds, batch):
    files = batch.get("im_file")
    if files is None:
        files = batch.get("img_path")
    for i, p in enumerate(preds):
        f = str(files[i]) if files is not None and i < len(files) else f"idx{i}"
        REC.setdefault(os.path.basename(f), {})[name] = {
            "xyxy": as_list(p.get("bboxes")),
            "conf": as_list(p.get("conf")),
            "cls": as_list(p.get("cls")),
        }
    return _orig(self, name, preds, batch)


dv.DualHeadDetectionValidator._update_branch_metrics = patched
import eval_split as es

DATA = "/3250410011/root/rivermind-data/mendeley_split/data.yaml"
OUTDIR = os.path.join(PROJ, f"perframe_boxes_{TAG}")
os.makedirs(OUTDIR, exist_ok=True)
extra = [] if LEVEL == "0" else ["--ir-shift", f"{LEVEL},0"]
sys.argv = ["eval_split.py", "--weights", WEIGHTS,
            "--data", DATA, "--split", "test", "--labels-mode", "dual",
            "--tag", f"pfbox_{TAG}_{LEVEL}", "--batch", "8", "--workers", "2",
            "--device", "cpu", "--out", os.path.join(OUTDIR, f"_eval_{LEVEL}.json")] + extra
rc = 0
try:
    rc = es.main()
except SystemExit as e:
    rc = e.code
finally:
    with open(os.path.join(OUTDIR, f"boxes_{LEVEL}.json"), "w") as f:
        json.dump(REC, f)
    print(f"[perframe-boxes] tag={TAG} level={LEVEL} frames={len(REC)} rc={rc}", flush=True)
