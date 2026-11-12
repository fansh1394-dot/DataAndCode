# -*- coding: utf-8 -*-
"""DIV-A：按时间冗余抽帧，导出 ``uav_data_v2_dedup``。

为什么需要它
------------
实测 v2 的**帧间冗余度仅 0.170（train）/ 0.202（test）**——相邻帧灰度 MAD 只有
随机两帧的 1/5，29,716 帧的有效独立样本量约相当于 5,000~6,000 帧。
抽帧能在基本不损失多样性的前提下提升单位算力的样本熵。

抽帧规则（逐序列、按 frame_index 升序）
--------------------------------------
    保留第 0 帧
    之后每一帧：若与「上一个保留帧」的灰度 MAD >= TAU 则保留
    同时强制最大间隔 GAP：连续 GAP 帧未保留则强制保留下一帧
                     （避免悬停/静止段被整段丢弃）

关键决策
--------
- **test 一律不抽帧**，保持全量 16,917 帧，作为固定评测集；否则与全部历史结果不可比。
- val 默认抽帧（它只用于早停选模），可用 ``--keep-val`` 关闭。
- 输出的 ``metadata/{split}.json`` 只保留被抽中的帧，``kind`` / ``sequence`` 原样继承，
  保证帧类型分桶（both / only_ir / only_rgb）的评测口径不变。
- 图像与标签**不复制**，用软链指向原数据集，节省约 30 GB 磁盘。

用法
----
    python scripts/export_v2_dedup.py --fit-only          # 只统计保留率，不落盘
    python scripts/export_v2_dedup.py --tau 12 --gap 10   # 正式导出
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

cv2.setNumThreads(1)

ROOT_DEFAULT = "/root/rivermind-data/uav_data_v2"
OUT_DEFAULT = "/root/rivermind-data/uav_data_v2_dedup"
WORK = (160, 128)          # 计算 MAD 的降采样尺寸
LINK_DIRS = ("images", "images_ir", "labels", "labels_ir")


# ------------------------------------------------------------------ 工具
def frame_path(root: Path, kind: str, split: str, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = root / kind / split / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def load_small(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, WORK, interpolation=cv2.INTER_AREA).astype(np.float32)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


# ------------------------------------------------------------------ 抽帧
def dedup_sequence(stems, frames_meta, smalls, tau, gap, log):
    """返回被保留的 stem 列表。stems 已按 frame_index 升序。"""
    keep = [stems[0]]
    last_small = smalls[stems[0]]
    last_idx = 0
    for i in range(1, len(stems)):
        s = stems[i]
        cur = smalls[s]
        gap_now = i - last_idx
        if cur is None or last_small is None:
            keep.append(s)
            last_idx = i
            continue
        d = mad(last_small, cur)
        if d >= tau or gap_now >= gap:
            keep.append(s)
            last_small = cur
            last_idx = i
    return keep


def process_split(root: Path, split: str, tau: float, gap: int, stride: int, log):
    mp = root / "metadata" / f"{split}.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    frames = meta["frames"]

    seq2stems = defaultdict(list)
    for stem, info in frames.items():
        seq2stems[info["sequence"]].append(stem)
    for seq in seq2stems:
        seq2stems[seq].sort(key=lambda s: frames[s].get("frame_index", 0))

    kept, per_seq = [], []
    t0 = time.time()
    for si, seq in enumerate(sorted(seq2stems), 1):
        stems = seq2stems[seq]
        # 只加载参与比较的帧（全部），smalls 缓存可在序列结束后释放
        smalls = {}
        for s in stems:
            p = frame_path(root, "images", split, s)
            smalls[s] = load_small(p) if p else None
        k = dedup_sequence(stems, frames, smalls, tau, gap, log)
        kept.extend(k)
        per_seq.append({"sequence": seq, "total": len(stems), "kept": len(k),
                        "rate": len(k) / max(len(stems), 1)})
        del smalls
        if si % 20 == 0 or si == len(seq2stems):
            log(f"    [{split}] 序列 {si}/{len(seq2stems)}  累计保留 {len(kept)} "
                f"({time.time()-t0:.0f}s)")

    # 冗余度校验：抽帧前后相邻帧 MAD / 随机帧 MAD
    def redundancy(stem_list):
        by_seq = defaultdict(list)
        for s in stem_list:
            by_seq[frames[s]["sequence"]].append(s)
        adj, cache = [], {}
        for seq, ss in by_seq.items():
            ss.sort(key=lambda x: frames[x].get("frame_index", 0))
            for a, b in zip(ss, ss[1:]):
                for x in (a, b):
                    if x not in cache:
                        p = frame_path(root, "images", split, x)
                        cache[x] = load_small(p) if p else None
                if cache[a] is not None and cache[b] is not None:
                    adj.append(mad(cache[a], cache[b]))
        rng = np.random.default_rng(0)
        pool = [s for s in stem_list
                if (p := frame_path(root, "images", split, s)) is not None][:2000]
        rnd = []
        for _ in range(min(2000, len(pool) // 2)):
            a, b = rng.choice(len(pool), 2, replace=False)
            pa = frame_path(root, "images", split, pool[a])
            pb = frame_path(root, "images", split, pool[b])
            ia, ib = load_small(pa), load_small(pb)
            if ia is not None and ib is not None:
                rnd.append(mad(ia, ib))
        return (float(np.mean(adj)) if adj else None,
                float(np.mean(rnd)) if rnd else None)

    adj_k, rnd_k = redundancy(kept)
    sample_all = [s for s in frames][:: max(1, len(frames) // 3000)]
    adj_o, rnd_o = redundancy(sample_all)

    stats = {
        "split": split,
        "frames_before": len(frames),
        "frames_after": len(kept),
        "keep_rate": len(kept) / max(len(frames), 1),
        "seqs": len(seq2stems),
        "redundancy_before": (adj_o / rnd_o) if (adj_o and rnd_o) else None,
        "redundancy_after": (adj_k / rnd_k) if (adj_k and rnd_k) else None,
        "adjacent_mad_before": adj_o,
        "adjacent_mad_after": adj_k,
        "random_mad_before": rnd_o,
        "random_mad_after": rnd_k,
        "tau": tau, "gap": gap,
        "per_seq": per_seq,
    }
    # 帧类型三分布是否仍齐全
    from collections import Counter
    kind_before = Counter(frames[s]["kind"] for s in frames)
    kind_after = Counter(frames[s]["kind"] for s in kept)
    stats["kind_before"] = dict(kind_before)
    stats["kind_after"] = dict(kind_after)
    return kept, stats


# ------------------------------------------------------------------ 导出
def export(root: Path, out: Path, split_kept: dict, log):
    out.mkdir(parents=True, exist_ok=True)
    # 软链四个内容目录
    for d in LINK_DIRS:
        dst = out / d
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        os.symlink(root / d, dst)
        log(f"  [链接] {dst} -> {root/d}")

    # metadata：只保留抽中的帧
    mdst = out / "metadata"
    if mdst.is_symlink() or mdst.exists():
        if mdst.is_symlink():
            mdst.unlink()
        else:
            shutil.rmtree(mdst)
    mdst.mkdir(parents=True)
    for split, kept in split_kept.items():
        src = json.loads((root / "metadata" / f"{split}.json").read_text(encoding="utf-8"))
        sub = {k: src["frames"][k] for k in kept}
        payload = {"version": src.get("version", 1), "split": split, "frames": sub}
        (mdst / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        log(f"  [metadata] {split}: {len(sub)} 帧")

    # data.yaml
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8")) or {}
    cfg["path"] = str(out)
    (out / "data.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log(f"  [yaml] {out/'data.yaml'}")


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="DIV-A 抽帧去冗余导出")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--tau", type=float, default=12.0, help="相邻帧 MAD 阈值")
    ap.add_argument("--gap", type=int, default=10, help="最大强制保留间隔（帧）")
    ap.add_argument("--splits", default="train,val", help="参与抽帧的 split；test 永不抽帧")
    ap.add_argument("--keep-val", action="store_true", help="val 也保持全量")
    ap.add_argument("--fit-only", action="store_true", help="只统计，不导出")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    log = lambda m: print(m, flush=True)

    if not (root / "data.yaml").is_file():
        log(f"[错误] 缺 {root/'data.yaml'}")
        return 2

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.keep_val and "val" in splits:
        splits.remove("val")
    log(f"root={root}\nout={out}\n抽帧 splits={splits} tau={args.tau} gap={args.gap}")
    log("[说明] test 一律不抽帧，保持全量作为固定评测集")

    split_kept, all_stats = {}, []
    for split in splits:
        if not (root / "metadata" / f"{split}.json").is_file():
            log(f"[跳过] 缺 metadata/{split}.json")
            continue
        kept, stats = process_split(root, split, args.tau, args.gap, 1, log)
        split_kept[split] = kept
        all_stats.append(stats)

    # ---- 汇总 ----
    log("\n=== 抽帧结果 ===")
    for s in all_stats:
        log(f"  {s['split']}: {s['frames_before']} -> {s['frames_after']} "
            f"(保留 {s['keep_rate']*100:.1f}%)")
        log(f"      帧间冗余度 {s['redundancy_before']:.3f} -> {s['redundancy_after']:.3f}")
        log(f"      相邻帧 MAD {s['adjacent_mad_before']:.2f} -> {s['adjacent_mad_after']:.2f}")
        log(f"      帧类型 {s['kind_before']} -> {s['kind_after']}")
        rates = np.array([p["rate"] for p in s["per_seq"]])
        log(f"      每序列保留率 min={rates.min():.2f} 中位={np.median(rates):.2f} "
            f"max={rates.max():.2f}")

    if args.fit_only:
        out.mkdir(parents=True, exist_ok=True)
        (out / "_dedup_fit.json").write_text(
            json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\n[fit-only] {out/'_dedup_fit.json'}")
        return 0

    export(root, out, split_kept, log)

    # test 需在 metadata 中保持全量，export 只写了抽帧 split，
    # 因此这里补一份全量 test metadata 的软链
    mdst = out / "metadata"
    tst = mdst / "test.json"
    if not tst.exists():
        os.symlink(root / "metadata" / "test.json", tst)
        log(f"  [metadata] test -> 全量（软链）")

    payload = {"config": {"tau": args.tau, "gap": args.gap, "splits": splits},
               "note": "test 未抽帧，保持全量 16917 帧作为固定评测集",
               "stats": all_stats}
    (out / "_dedup_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n[DIV-A] 完成 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
