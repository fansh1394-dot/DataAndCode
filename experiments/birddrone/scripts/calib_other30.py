#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校准 other30：按需复制已有 other 框，使 other 框占比接近 30%（目标 28–32%）。"""
import os
import random
from collections import Counter
from pathlib import Path

RAW = Path("/root/rivermind-fs/birddrone")
SRC = RAW / "bd_nc3"
D30 = RAW / "bd_nc3_other30"


def count(src_dir: Path, split: str):
    names = [x.strip() for x in (src_dir / f"{split}_nc3.txt").read_text().splitlines() if x.strip()]
    c = Counter()
    for n in names:
        p = src_dir / "labels" / split / f"{Path(n).stem}.txt"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    c[l.split()[0]] += 1
    return names, c


def rebuild(split: str, reps: int) -> None:
    names, _ = count(SRC, split)
    for sub in ("labels", "labels_ir"):
        for n in names:
            stem = Path(n).stem
            p = D30 / sub / split / f"{stem}.txt"
            if p.exists() and os.path.islink(p):
                p.unlink()
            s = SRC / sub / split / f"{stem}.txt"
            if not s.exists():
                continue
            lines = [l for l in s.read_text(encoding="utf-8").splitlines() if l.strip()]
            others = [l for l in lines if l.startswith("2 ")]
            extra = []
            for _ in range(reps):
                for ln in others:
                    t = ln.split()
                    cx, cy, w, h = float(t[1]), float(t[2]), float(t[3]), float(t[4])
                    cx2 = min(max(cx + random.uniform(-0.18, 0.18), 0.03), 0.97)
                    cy2 = min(max(cy + random.uniform(-0.12, 0.12), 0.03), 0.97)
                    sc = random.uniform(0.7, 1.15)
                    extra.append(f"2 {cx2:.6f} {cy2:.6f} {min(w*sc,0.3):.6f} {min(h*sc,0.3):.6f}")
            p.write_text("\n".join(lines + extra) + "\n", encoding="utf-8")


def ratio(split: str) -> float:
    names = [x.strip() for x in (D30 / f"{split}_list.txt").read_text().splitlines() if x.strip()]
    c = Counter()
    for n in names:
        p = D30 / "labels" / split / f"{Path(n).stem}.txt"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    c[l.split()[0]] += 1
    tot = sum(c.values())
    return 100 * c["2"] / max(tot, 1), dict(c)


random.seed(0)
for split in ("train", "val"):
    _, c0 = count(SRC, split)
    others = c0["2"]
    base_total = sum(c0.values()) - others
    # 目标: others*(1+reps) / (base_total + others*(1+reps)) ≈ 0.30
    target_others = 0.30 * base_total / 0.70
    reps = max(0, round(target_others / others - 1))
    rebuild(split, reps)
    r, c = ratio(split)
    print(f"{split}: reps={reps}  框 {c}  总 {sum(c.values())}  other 占比 {r:.1f}%")
