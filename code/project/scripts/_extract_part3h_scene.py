# -*- coding: utf-8 -*-
"""从 d2c1 原始 JSON 中提取 part3h 的分场景指标，与 M0/M3 并列。

d2c1_scene_modality.py 的汇总表列是写死的（只含 M0/M3），part3h 虽已评测
但不会出现在 .md 里，故单独抽一次。
"""
import json
from pathlib import Path

RAW = Path("bench/d2c1/raw")
SCENES = ["flight", "indoor", "ground", "synthetic", "background_only"]
MODELS = [("M0_concat", "M0"), ("M3_dual", "M3"), ("M3part3h", "part3h")]
KEY = "metrics/{b}/metrics/mAP50-95(B)"


def load(tag):
    p = RAW / f"{tag}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("metrics") or {}


print(f"{'场景':<16}{'帧数':>6}{'M0-fuse':>10}{'M3-fuse':>10}"
      f"{'part3h':>10}{'Δ(3h−M0)':>12}{'3h-rgb':>9}{'3h-ir':>9}")
print("-" * 82)
N = {"flight": 380, "indoor": 28, "ground": 33, "synthetic": 118, "background_only": 55}
deltas = []
for sc in SCENES:
    vals = {}
    for mname, label in MODELS:
        m = load(f"final_v3_{mname}_{sc}") or {}
        vals[label] = m.get(KEY.format(b="fuse"))
        if label == "part3h":
            vals["rgb"] = m.get(KEY.format(b="rgb"))
            vals["ir"] = m.get(KEY.format(b="ir"))
    m0, m3, p3 = vals.get("M0"), vals.get("M3"), vals.get("part3h")
    d = (p3 - m0) if (p3 is not None and m0 is not None) else None
    if d is not None and sc != "background_only":
        deltas.append(d)
    f = lambda v: f"{v:.4f}" if v is not None else "  N/A "
    print(f"{sc:<16}{N[sc]:>6}{f(m0):>10}{f(m3):>10}{f(p3):>10}"
          f"{(f'{d:+.4f}' if d is not None else 'N/A'):>12}"
          f"{f(vals.get('rgb')):>9}{f(vals.get('ir')):>9}")

if deltas:
    print("-" * 82)
    print(f"part3h − M0 净增益：均值 {sum(deltas)/len(deltas):+.4f}，"
          f"最大 {max(deltas):+.4f}，最小 {min(deltas):+.4f}")
