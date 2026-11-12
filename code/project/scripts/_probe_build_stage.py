# -*- coding: utf-8 -*-
"""分阶段探针：先构建 M0 基线，再逐个构建替换后的 YAML，每步打印时间戳。

用途：定位「整网构建卡死」到底卡在哪一步（YAML 解析 / parse_model / 前向）。
"""
import faulthandler
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

# 容器内禁止 ptrace，py-spy 用不了；用进程内 faulthandler 定时打印栈，
# 这样卡住时也能看到卡在哪个调用。放在重导入之后启动，避免刷屏。
faulthandler.dump_traceback_later(45, repeat=True)

BASE_DIR = ROOT / "ultralytics/cfg/models/mm/detect/yolo26/base"


def t0():
    return time.strftime("%H:%M:%S")


def step(label):
    print(f"[{t0()}] {label}", flush=True)


def build(p: Path, tag: str):
    step(f"{tag}: 读 YAML")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    step(f"{tag}: head 层数={len(cfg.get('head', []))}")
    step(f"{tag}: 开始 DetectionModel()")
    m = DetectionModel(cfg, ch=6, nc=1, verbose=False)
    step(f"{tag}: 构建完成 layers={len(m.model)} params={sum(p.numel() for p in m.parameters()) / 1e6:.4f}M")
    step(f"{tag}: 开始前向 2x6x256x256")
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 6, 256, 256))
    step(f"{tag}: 前向完成")
    return m


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "yolo26n-mm-dual-fusion-concat.yaml",
        "yolo26n-mm-dual-fusion-cga3h.yaml",
        "yolo26n-mm-dual-fusion-pim3h.yaml",
        "yolo26n-mm-dual-fusion-cmce3h.yaml",
    ]
    for n in names:
        p = BASE_DIR / n
        if not p.exists():
            step(f"缺失 {n}")
            continue
        try:
            build(p, n)
        except Exception as e:
            step(f"{n}: 失败 {type(e).__name__}: {str(e)[:300]}")
    step("全部结束")
