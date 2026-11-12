# -*- coding: utf-8 -*-
"""E32：阈值标定 + 轨迹级告警评估（K=5 / N=3）。

AVID：按视频分组，GT 框跨帧 IoU 关联成轨迹；
合成 other：按 clip 分组，同样关联；
每帧取与 GT IoU>=0.5 的最佳预测；轨迹任意连续 5 帧内 >=3 帧判为 drone 即告警。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path("/root/rivermind-fs/birddrone")
BD2, BD3, E36 = RAW / "bd_nc2", RAW / "bd_nc3", RAW / "E36"
K, N = 5, 3
TAUS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def load_preds(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[Path(d["im_file"]).name] = (
            np.asarray(d["bboxes"], np.float32).reshape(-1, 4),
            np.asarray(d["conf"], np.float32),
            np.asarray(d["cls"], np.int32),
        )
    return out


def read_gt(lb: Path, w, h):
    gts = []
    if not lb.exists():
        return gts
    for ln in lb.read_text().splitlines():
        t = ln.split()
        if len(t) >= 5:
            c = int(float(t[0])); cx, cy, bw, bh = [float(v) for v in t[1:5]]
            gts.append((c, np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                     (cx + bw / 2) * w, (cy + bh / 2) * h], np.float32)))
    return gts


def build_tracks(frames):
    """frames: [(order, [(cls, box)])]；返回 [{cls, frames:{order: box}}]"""
    tracks, active = [], []
    for order, boxes in frames:
        used, matched = set(), set()
        for gi, (c, b) in enumerate(boxes):
            best = None
            for ai, (t, last, lb, last_order) in enumerate(active):
                if ai in used or t["cls"] != c or abs(last_order - order) > 5:
                    continue
                v = iou(b, lb)
                if v >= 0.3 and (best is None or v > best[0]):
                    best = (v, ai)
            if best is not None:
                ai = best[1]
                used.add(ai)
                t = active[ai][0]
                t["frames"][order] = b
                active[ai] = (t, order, b, order)
                matched.add(gi)
        for gi, (c, b) in enumerate(boxes):
            if gi in matched:
                continue
            t = {"cls": c, "frames": {order: b}}
            tracks.append(t)
            active.append((t, order, b, order))
        active = [a for a in active if order - a[3] <= 5]
    return [t for t in tracks if len(t["frames"]) >= 3]


def collect(root: Path, rels, preds, syn: bool):
    groups = defaultdict(list)
    for rel in rels:
        stem = Path(rel).stem
        p = stem.split("__")
        if syn:
            groups[f"{p[1]}:{p[3]}"].append((int(p[2]), rel))
        else:
            groups[p[1]].append((int(p[2]), rel))

    tracks = []
    for g, items in groups.items():
        items.sort()
        frames, meta = [], {}
        for order, rel in items:
            img = root / rel
            with Image.open(img) as im:
                w, h = im.size
            lb = root / "labels/val" / f"{img.stem}.txt"
            boxes = read_gt(lb, w, h)
            frames.append((order, boxes))
            meta[order] = (img, boxes)
        for t in build_tracks(frames):
            sides = [np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1e-6))
                     for b in t["frames"].values()]
            votes = []
            for order in sorted(t["frames"].keys()):
                img, boxes = meta[order]
                gt = t["frames"][order]
                pb, pc, pk = preds.get(img.name, (np.zeros((0, 4), np.float32),
                                                 np.zeros(0, np.float32), np.zeros(0, np.int32)))
                if len(pb) == 0:
                    votes.append((order, -1, 0.0)); continue
                ious = np.array([iou(gt, b) for b in pb])
                j = int(ious.argmax())
                votes.append((order, int(pk[j]), float(pc[j])) if ious[j] >= 0.5
                             else (order, -1, 0.0))
            tracks.append({"cls": t["cls"], "votes": votes, "group": g,
                           "side": float(np.median(sides))})
    return tracks


def alarmed(votes, tau):
    seq = [1 if c == 1 and cf >= tau else 0 for _, c, cf in votes]
    return any(sum(seq[i:i + K]) >= N for i in range(len(seq)))


def main():
    avid_rels = [x.strip() for x in (BD2 / "avid_val.txt").read_text().splitlines() if x.strip()]
    syn_rels = [f"images/val/{p.name}" for p in sorted((BD3 / "images/val").glob("*__syn*.jpg"))]
    avid = collect(BD2, avid_rels, load_preds(E36 / "preds_avid.jsonl"), syn=False)
    syn = collect(BD3, syn_rels, load_preds(E36 / "preds_syn.jsonl"), syn=True)

    n_bird = sum(1 for t in avid if t["cls"] == 0)
    n_drone = sum(1 for t in avid if t["cls"] == 1)
    n_other = sum(1 for t in syn if t["cls"] == 2)
    def run_sweep(min_side=0.0):
        av = [t for t in avid if t["side"] >= min_side]
        sy = [t for t in syn if t["side"] >= min_side]
        nb = sum(1 for t in av if t["cls"] == 0)
        nd = sum(1 for t in av if t["cls"] == 1)
        no = sum(1 for t in sy if t["cls"] == 2)
        rows = []
        for tau in TAUS:
            b_fp = sum(1 for t in av if t["cls"] == 0 and alarmed(t["votes"], tau))
            d_tp = sum(1 for t in av if t["cls"] == 1 and alarmed(t["votes"], tau))
            o_fp = sum(1 for t in sy if t["cls"] == 2 and alarmed(t["votes"], tau))
            rows.append({"tau_drone": tau,
                         "drone_recall": round(d_tp / max(nd, 1), 4),
                         "bird_fp": b_fp, "bird_fp_rate": round(b_fp / max(nb, 1), 4),
                         "other_fp": o_fp, "other_fp_rate": round(o_fp / max(no, 1), 4)})
        return {"n": {"bird": nb, "drone": nd, "other": no}, "sweep": rows}

    res = {"K": K, "N": N, "tracks": {"avid_bird": n_bird, "avid_drone": n_drone,
                                      "syn_other": n_other}, "sweep": []}
    res["all"] = run_sweep(0.0)
    res["a_level_ge16px"] = run_sweep(16.0)
    res["sweep"] = res["all"]["sweep"]
    (E36 / "E32_calibration.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    for name in ("all", "a_level_ge16px"):
        blk = res[name]
        print(f"\n[{name}] tracks {blk['n']}")
        print(f"{'tau':>5} {'drone_rec':>10} {'bird_FP':>14} {'other_FP':>15}")
        for r in blk["sweep"]:
            print(f"{r['tau_drone']:>5} {r['drone_recall']:>10.3f} "
                  f"{r['bird_fp']:>4} ({r['bird_fp_rate']:.3f}) "
                  f"{r['other_fp']:>4} ({r['other_fp_rate']:.3f})")

    # 诊断：哪些 bird 轨迹在 tau=0.4 触发了告警
    print("\nbird FP 明细 (tau=0.4)：")
    for t in avid:
        if t["cls"] == 0 and alarmed(t["votes"], 0.4):
            d = [c for _, c, _ in t["votes"] if c == 1]
            print("  ", t["group"], "side", round(t["side"], 1), "frames", len(t["votes"]),
                  "drone_votes", len(d),
                  "max_conf", round(max([cf for _, c, cf in t["votes"] if c == 1] or [0]), 3))


if __name__ == "__main__":
    main()
