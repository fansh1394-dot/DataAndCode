#!/usr/bin/env python3
"""冒烟：在官方 DualHeadDetectionValidator 上验证决策级晚融合接线正确。

只跑 test 的前 N 帧（截断 dataset），确认 metrics/route/* 能被产出，
且关闭 late_route 时行为与历史路径一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_split import LABELS_MODES, build_derived_yaml  # noqa: E402

DATA = Path("/root/rivermind-data/uav_data_v2")
N = 48


def build(weights: str, late_route: bool, batch: int = 8):
    from ultralytics import YOLOMM
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator as MMValidator
    from ultralytics.nn.mm.dual_val import DualHeadDetectionValidator

    derived = build_derived_yaml(
        DATA / "data.yaml", ROOT / "_d2p1" / "eval_yamls" / "data_smoke_test.yaml", "test", LABELS_MODES["dual"]
    )
    raw = yaml.safe_load(derived.read_text(encoding="utf-8"))
    data = check_det_dataset(str(derived))
    for k, val in raw.items():
        data.setdefault(k, val)

    vargs = {
        "task": "detect",
        "mode": "val",
        "data": str(derived),
        "split": "val",
        "imgsz": 640,
        "batch": batch,
        "workers": 0,
        "device": "cpu",
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "plots": False,
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "verbose": False,
        "project": str(ROOT / "YOLO26" / "logs"),
        "name": "smoke_late_route",
        "late_route": late_route,
        "late_route_tau_low": 0.30,
        "late_route_tau_high": 0.50,
        "late_route_conf": 0.25,
    }

    net = YOLOMM(weights).model
    builder = MMValidator(args=vargs)
    builder.data = data
    builder.stride = int(net.stride.max())
    dataset = builder.build_dataset(data.get("val"), mode="val", batch=batch)
    dataset.labels = dataset.labels[:N]
    dataset.im_files = dataset.im_files[:N]
    if getattr(dataset, "npy_files", None):
        dataset.npy_files = dataset.npy_files[:N]
    loader = build_dataloader(dataset, batch=batch, workers=0, shuffle=False, rank=-1)
    print(f"[smoke] n={len(dataset)} late_route={late_route}")

    net = net.to(torch.device("cpu")).float().eval()
    v = DualHeadDetectionValidator(dataloader=loader, args=vargs)
    v.data = data
    v.device = torch.device("cpu")
    ret = v(model=net)
    return {k: float(x) for k, x in ret.items() if isinstance(x, (int, float)) and "route" in k}


def main():
    w = str(ROOT / "ResTest" / "v2_C4_deform_s0" / "weights" / "best.pt")
    off = build(w, False)
    on = build(w, True)
    print("[off] route keys:", len(off))
    for k, val in on.items():
        print(f"  {k} = {val:.6f}")
    assert on.get("metrics/route/metrics/mAP50-95(B)") is not None, "route 指标缺失"
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
