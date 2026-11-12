# -*- coding: utf-8 -*-
"""数据集画像：规模、序列结构、目标框尺寸分布、帧间重复度。"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/rivermind-data/uav_data_v2")
W, H = 640, 512


def boxes(p: Path):
    out = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.split()
            if len(s) >= 5:
                out.append(tuple(float(x) for x in s[1:5]))
    return out


for split in ("train", "val", "test"):
    fr = json.loads((ROOT / "metadata" / f"{split}.json").read_text(encoding="utf-8"))["frames"]
    seq = collections.Counter(v["sequence"] for v in fr.values())
    kind = collections.Counter(v["kind"] for v in fr.values())
    v = sorted(seq.values())
    print(f"\n=== {split} ===")
    print(f"  frames={len(fr)}  sequences={len(seq)}  kinds={dict(kind)}")
    print(f"  per-seq frames: min={v[0]} median={v[len(v)//2]} max={v[-1]}")
    print(f"  最长 5 个序列帧数: {v[-5:]}  最短 5 个: {v[:5]}")
    print(f"  大序列(>=500 帧) 贡献帧数占比: "
          f"{sum(x for x in v if x >= 500)/max(sum(v),1)*100:.1f}%")

    ws, hs = [], []
    for stem, info in fr.items():
        if info["kind"] in ("both", "only_rgb"):
            for b in boxes(ROOT / "labels" / split / f"{stem}.txt"):
                ws.append(b[2] * W)
                hs.append(b[3] * H)
        elif info["kind"] == "only_ir":
            for b in boxes(ROOT / "labels_ir" / split / f"{stem}.txt"):
                ws.append(b[2] * W)
                hs.append(b[3] * H)
    if ws:
        a = np.array(ws)
        b = np.array(hs)
        side = np.sqrt(a * b)
        print(f"  目标框宽(px): p10={np.percentile(a,10):.1f} med={np.median(a):.1f} p90={np.percentile(a,90):.1f}")
        print(f"  目标框高(px): p10={np.percentile(b,10):.1f} med={np.median(b):.1f} p90={np.percentile(b,90):.1f}")
        print(f"  等效边长: <16px {np.mean(side<16)*100:.1f}% | <32px {np.mean(side<32)*100:.1f}% "
              f"| <64px {np.mean(side<64)*100:.1f}% | >=64px {np.mean(side>=64)*100:.1f}%")
        print(f"  目标数: {len(ws)}  （每帧均值 {len(ws)/max(len(fr),1):.2f}）")
