# -*- coding: utf-8 -*-
"""数据集充分性与目标特性统计。

回答：现有数据量够不够？关键不是总数，而是
  (a) 有效样本数（多少帧真有目标）
  (b) 目标尺寸分布（小目标占比）
  (c) 场景/序列多样性（去重后的独立录制数量）
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/3250410011/root/rivermind-data/mendeley_split")
SPLITS = ("train", "val", "test")


def boxes(lbl_dir, split):
    """返回每帧的框数量与归一化尺寸。"""
    per_frame = []
    for f in sorted((lbl_dir / split).glob("*.txt")):
        w, h = 0.0, 0.0
        n = 0
        for line in f.read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 5:
                n += 1
                w = max(w, float(p[3]))
                h = max(h, float(p[4]))
        per_frame.append((n, w, h))
    return per_frame


def stats(name, split):
    pf = boxes(ROOT / "labels", split)
    frames = len(pf)
    empty = sum(1 for n, _, _ in pf if n == 0)
    boxes_n = sum(n for n, _, _ in pf)
    # 以 640 输入、目标边长像素计
    areas = []
    for n, w, h in pf:
        if n > 0:
            # 归一化边长 -> 640 像素空间的边长
            areas.append((w * 640) * (h * 640))
    areas.sort()
    print(f"\n[{name}]")
    print(f"  帧数={frames}  含目标={frames - empty}  空帧={empty} "
          f"({empty / frames * 100:.1f}%)  总框数={boxes_n}")
    if areas:
        small = sum(1 for a in areas if a < 32 ** 2)
        med = sum(1 for a in areas if 32 ** 2 <= a < 96 ** 2)
        large = sum(1 for a in areas if a >= 96 ** 2)
        tot = len(areas)
        print(f"  目标边长(640空间) 中位数={areas[tot // 2] ** 0.5:.0f}px")
        print(f"  小目标(<32px)={small / tot * 100:.1f}%  "
              f"中(32-96)={med / tot * 100:.1f}%  大(>96)={large / tot * 100:.1f}%")


if __name__ == "__main__":
    print("=== 数据充分性统计 ===")
    for s in SPLITS:
        stats(s, s)

    # 序列多样性：metadata 里的视频/场景字段
    for s in SPLITS:
        mp = ROOT / "metadata" / f"{s}.json"
        if not mp.exists():
            continue
        data = json.loads(mp.read_text(encoding="utf-8"))
        sample = data[0] if isinstance(data, list) else next(iter(data.values()))
        print(f"\n[{s}] metadata 字段示例: {list(sample.keys())[:10]}")
        # 尝试按序列/场景聚合
        for key in ("video", "seq", "sequence", "scene", "source", "clip"):
            if isinstance(sample, dict) and key in sample:
                vals = Counter(str(x.get(key)) for x in (data if isinstance(data, list) else data.values()))
                print(f"  按 '{key}' 聚合: {len(vals)} 个独立组")
                break
