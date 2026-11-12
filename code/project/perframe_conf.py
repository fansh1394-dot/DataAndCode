# -*- coding: utf-8 -*-
"""逐帧导出 rgb/ir/fuse 三头的最高置信度（挂钩验证器，不重写推理）

用法：换 LEVEL 环境变量，跑 6 次；每次 614 帧。
钩子位置：DualHeadDetectionValidator._update_branch_metrics(name, preds, batch)
  —— 每帧每分支都会经过，且 _frame_conf 正是验证器自己用的置信度定义（floor=0，保留原值）
"""
import sys, os, json

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

from ultralytics.nn.mm import dual_val as dv

REC = {}
_orig = dv.DualHeadDetectionValidator._update_branch_metrics


def patched(self, name, preds, batch):
    files = batch.get("im_file")
    if files is None:
        files = batch.get("img_path")
    for i, p in enumerate(preds):
        f = str(files[i]) if files is not None and i < len(files) else f"idx{i}"
        REC.setdefault(os.path.basename(f), {})[name] = self._frame_conf(p, 0.0)
    return _orig(self, name, preds, batch)


dv.DualHeadDetectionValidator._update_branch_metrics = patched

import eval_split as es

LEVEL = os.environ.get("LEVEL", "0")
W = os.environ.get("WEIGHTS",
                   "YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt")
DATA = "/3250410011/root/rivermind-data/mendeley_split/data.yaml"
OUTDIR = os.path.join(PROJ, "perframe_conf")
os.makedirs(OUTDIR, exist_ok=True)

extra = [] if LEVEL == "0" else ["--ir-shift", f"{LEVEL},0"]
sys.argv = ["eval_split.py",
            "--weights", W, "--data", DATA, "--split", "test",
            "--labels-mode", "dual", "--tag", f"pfconf_{LEVEL}",
            "--batch", "8", "--workers", "2", "--device", "cpu",
            "--out", os.path.join(OUTDIR, f"_eval_{LEVEL}.json")] + extra

rc = 0
try:
    rc = es.main()
except SystemExit as e:
    rc = e.code
finally:
    with open(os.path.join(OUTDIR, f"conf_{LEVEL}.json"), "w") as fh:
        json.dump(REC, fh)
    print(f"[perframe] level={LEVEL} frames={len(REC)} rc={rc}", flush=True)
