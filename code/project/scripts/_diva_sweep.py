# -*- coding: utf-8 -*-
"""DIV-A 参数扫描：不同 tau 下的保留率与抽帧后冗余度，判断抽帧是否真的有效。

关键判据不是「保留率」，而是 **抽帧后相邻帧差异是否显著变大**：
若抽掉一半帧后，剩余相邻帧依然几乎一样，说明冗余不是「帧太密」造成的，
而是「序列内本就缺乏变化」（悬停、目标缓慢移动），抽帧无法解决。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
ROOT = Path("/root/rivermind-data/uav_data_v2")
WORK = (160, 128)


def load_small(p):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, WORK, interpolation=cv2.INTER_AREA).astype(np.float32)


def img_path(split, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = ROOT / "images" / split / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def main():
    split = "train"
    meta = json.loads((ROOT / "metadata" / f"{split}.json").read_text(encoding="utf-8"))
    frames = meta["frames"]
    seq2stems = defaultdict(list)
    for s, info in frames.items():
        seq2stems[info["sequence"]].append(s)
    for q in seq2stems:
        seq2stems[q].sort(key=lambda x: frames[x].get("frame_index", 0))

    seqs = sorted(seq2stems)
    rng = np.random.default_rng(0)
    pick = [seqs[i] for i in sorted(rng.choice(len(seqs), 40, replace=False))]

    # 预加载 40 个序列的图像
    cache = {}
    for q in pick:
        for s in seq2stems[q]:
            p = img_path(split, s)
            if p:
                cache[s] = load_small(p)

    all_small = np.stack([v for v in cache.values() if v is not None])
    rnd_mad = []
    for _ in range(3000):
        a, b = rng.choice(len(all_small), 2, replace=False)
        rnd_mad.append(float(np.abs(all_small[a] - all_small[b]).mean()))
    rnd = float(np.mean(rnd_mad))

    print(f"随机两帧 MAD = {rnd:.3f}（参照基准）")
    print(f"扫描 {len(pick)} 个序列，全部帧参与\n")
    print(f"{'tau':>6}{'gap':>5}{'保留率':>9}{'相邻MAD':>10}{'冗余度':>9}{'每序列中位保留率':>18}")

    base_adj = []
    for q in pick:
        ss = [s for s in seq2stems[q] if s in cache]
        for a, b in zip(ss, ss[1:]):
            base_adj.append(float(np.abs(cache[a] - cache[b]).mean()))
    print(f"{'-':>6}{'-':>5}{1.0:>9.3f}{np.mean(base_adj):>10.3f}"
          f"{np.mean(base_adj)/rnd:>9.3f}{1.0:>18.3f}   ← 原数据")

    for tau in (2, 3, 4, 5, 6, 8):
        for gap in (10,):
            kept_all, adj_all, rates = [], [], []
            for q in pick:
                ss = [s for s in seq2stems[q] if s in cache]
                keep = [ss[0]]
                last, last_i = cache[ss[0]], 0
                for i in range(1, len(ss)):
                    d = float(np.abs(last - cache[ss[i]]).mean())
                    if d >= tau or (i - last_i) >= gap:
                        keep.append(ss[i])
                        last, last_i = cache[ss[i]], i
                kept_all += keep
                rates.append(len(keep) / len(ss))
                for a, b in zip(keep, keep[1:]):
                    adj_all.append(float(np.abs(cache[a] - cache[b]).mean()))
            tot = sum(len([s for s in seq2stems[q] if s in cache]) for q in pick)
            print(f"{tau:>6}{gap:>5}{len(kept_all)/tot:>9.3f}{np.mean(adj_all):>10.3f}"
                  f"{np.mean(adj_all)/rnd:>9.3f}{np.median(rates):>18.3f}")

    print("\n判读：若 tau 降到 2 仍无法把冗余度拉回 0.45 以上，")
    print("说明序列内本身缺乏变化，抽帧不是有效手段。")


if __name__ == "__main__":
    main()
