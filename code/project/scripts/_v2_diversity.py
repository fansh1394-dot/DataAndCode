# -*- coding: utf-8 -*-
"""B：v2 数据集视觉多样性量化。

度量维度（全部可复现、可写进论文）
--------------------------------
1. **序列结构**：序列数、每序列帧数、序列时长
2. **背景多样性**：对每帧计算**全局颜色/亮度直方图**，跨序列做两两距离
   （用降采样直方图 + 余弦距离），给出「同序列 vs 跨序列」的距离比 ——
   比值越小说明序列间背景越雷同
3. **帧间冗余**：同序列相邻帧的图像差异（MAD），以及「随机抽 N 帧的平均差异」
   与「相邻帧平均差异」之比 → 冗余度
4. **环境类型聚类**：用简单的统计特征（均值/方差/梯度能量/天空占比估计）
   做 KMeans 聚类，看 v2 实际有几种不同「场景」
5. **目标形态**：框长宽比分布、框面积分布

只读取图像，不做任何训练。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(4)


def read_yolo(p):
    out = []
    if not os.path.isfile(p):
        return out
    try:
        txt = open(p, encoding="utf-8").read()
    except UnicodeDecodeError:
        return out
    for line in txt.splitlines():
        s = line.split()
        if len(s) >= 5:
            try:
                out.append(tuple(float(x) for x in s[1:5]))
            except ValueError:
                pass
    return out


def frame_feat(path):
    """低层统计特征 + 64-bin 灰度直方图（归一化）。"""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    small = cv2.resize(im, (160, 128), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0], None, [64], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-9)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    # 天空占比估计：亮度高且梯度低的区域比例
    bright_lowgrad = ((small > 150) & (np.sqrt(gx ** 2 + gy ** 2) < 10)).mean()
    return {"hist": hist, "mean": float(small.mean()), "std": float(small.std()),
            "grad": grad, "sky": float(bright_lowgrad), "small": small}


def hist_dist(a, b):
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/rivermind-data/uav_data_v2")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--per-seq", type=int, default=8, help="每序列抽样帧数")
    ap.add_argument("--max-seqs", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    root = Path(args.root)
    report = {}

    for split in args.splits.split(","):
        split = split.strip()
        mp = root / "metadata" / f"{split}.json"
        if not mp.is_file():
            continue
        frames = json.loads(mp.read_text(encoding="utf-8"))["frames"]
        seq2stems = defaultdict(list)
        for stem, info in frames.items():
            seq2stems[info["sequence"]].append(stem)
        seqs = sorted(seq2stems)
        if args.max_seqs:
            seqs = seqs[: args.max_seqs]

        print(f"\n{'='*60}\nsplit={split}: {len(frames)} 帧, {len(seq2stems)} 序列")
        lens = sorted(len(v) for v in seq2stems.values())
        print(f"  每序列帧数: min={lens[0]} 中位={lens[len(lens)//2]} max={lens[-1]}")

        # ---- 抽样特征 ----
        feats = {}
        for seq in seqs:
            stems = sorted(seq2stems[seq])
            idx = np.linspace(0, len(stems) - 1, min(args.per_seq, len(stems))).astype(int)
            for k in idx:
                stem = stems[k]
                p = None
                for ext in (".jpg", ".jpeg", ".png"):
                    q = root / "images" / split / f"{stem}{ext}"
                    if q.is_file():
                        p = q
                        break
                if p is None:
                    continue
                f = frame_feat(p)
                if f is not None:
                    f["frame_idx"] = k
                    f["seq_len"] = len(stems)
                    feats[stem] = f
        print(f"  抽样 {len(feats)} 帧用于多样性分析")

        stems = list(feats)
        seq_of = {s: frames[s]["sequence"] for s in stems}

        # ---- 1. 同序列 vs 跨序列 直方图距离 ----
        same_d, cross_d = [], []
        rng = np.random.default_rng(0)
        for _ in range(4000):
            a, b = rng.choice(len(stems), 2, replace=False)
            sa, sb = stems[a], stems[b]
            d = hist_dist(feats[sa]["hist"], feats[sb]["hist"])
            (same_d if seq_of[sa] == seq_of[sb] else cross_d).append(d)
        same_d, cross_d = np.array(same_d), np.array(cross_d)
        print(f"\n  [背景多样性] 直方图余弦距离（越大越不同）")
        print(f"    同序列内  mean={same_d.mean():.4f}" if same_d.size else "    同序列内  样本不足")
        print(f"    跨序列    mean={cross_d.mean():.4f}  p10={np.percentile(cross_d,10):.4f} "
              f"p90={np.percentile(cross_d,90):.4f}")
        if same_d.size:
            print(f"    跨/同 比值 = {cross_d.mean()/max(same_d.mean(),1e-9):.2f}"
                  f"  （≈1 表示序列间背景几乎无差异）")

        # ---- 2. 相邻帧冗余 ----
        adj_d, rnd_d = [], []
        by_seq = defaultdict(list)
        for s in stems:
            by_seq[seq_of[s]].append(s)
        for seq, ss in by_seq.items():
            ss.sort(key=lambda x: feats[x]["frame_idx"])
            for a, b in zip(ss, ss[1:]):
                adj_d.append(float(np.abs(feats[a]["small"].astype(float) -
                                          feats[b]["small"].astype(float)).mean()))
        for _ in range(2000):
            a, b = rng.choice(len(stems), 2, replace=False)
            rnd_d.append(float(np.abs(feats[stems[a]]["small"].astype(float) -
                                      feats[stems[b]]["small"].astype(float)).mean()))
        adj_d, rnd_d = np.array(adj_d), np.array(rnd_d)
        print(f"\n  [帧间冗余] 像素绝对差均值")
        if adj_d.size:
            print(f"    相邻帧 = {adj_d.mean():.2f}")
        print(f"    随机两帧 = {rnd_d.mean():.2f}")
        if adj_d.size:
            print(f"    冗余度 = 相邻/随机 = {adj_d.mean()/max(rnd_d.mean(),1e-9):.3f}"
                  f"  （越小越冗余）")

        # ---- 3. 场景聚类 ----
        X = np.array([[feats[s]["mean"], feats[s]["std"], feats[s]["grad"], feats[s]["sky"]]
                      for s in stems], np.float32)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd
        best = None
        for K in (2, 3, 4, 5, 6, 8):
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            _, lab, cen = cv2.kmeans(Xn, K, None, crit, 5, cv2.KMEANS_PP_CENTERS)
            lab = lab.ravel()
            # 轮廓式简易分离度：平均簇内距 / 平均簇间距
            intra = np.mean([np.linalg.norm(Xn[i] - cen[lab[i]]) for i in range(len(Xn))])
            inter = np.mean([np.linalg.norm(cen[i] - cen[j])
                             for i in range(K) for j in range(K) if i < j])
            ratio = intra / max(inter, 1e-9)
            if K == 4:
                best = (K, lab, cen, ratio)
            print(f"    K={K}: 簇内/簇间 = {ratio:.3f}")
        if best:
            K, lab, cen, ratio = best
            cnt = np.bincount(lab, minlength=K)
            seqs_per_cluster = [len({seq_of[stems[i]] for i in range(len(lab)) if lab[i] == k})
                                for k in range(K)]
            print(f"    K=4 时：每簇帧数 {cnt.tolist()}，覆盖序列数 {seqs_per_cluster}")
            print(f"    平均每簇序列数 = {np.mean(seqs_per_cluster):.1f}"
                  f"（越大说明同一视觉场景由多个序列重复贡献）")

        # ---- 4. 目标形态 ----
        ar, sides = [], []
        for s in stems:
            for b in read_yolo(root / "labels" / split / f"{s}.txt"):
                w_, h_ = b[2] * 640, b[3] * 512
                ar.append(w_ / max(h_, 1e-6))
                sides.append(float(np.sqrt(max(w_, 1) * max(h_, 1))))
        if ar:
            ar, sides = np.array(ar), np.array(sides)
            print(f"\n  [目标形态] n={len(ar)}")
            print(f"    长宽比: 中位={np.median(ar):.2f} p10={np.percentile(ar,10):.2f} "
                  f"p90={np.percentile(ar,90):.2f}")
            print(f"    等效边长: 中位={np.median(sides):.1f} "
                  f"<16px {np.mean(sides<16)*100:.1f}% ≥64px {np.mean(sides>=64)*100:.1f}%")

        report[split] = {
            "n_frames": len(frames), "n_seqs": len(seq2stems),
            "sampled": len(feats),
            "hist_same_seq": float(same_d.mean()) if same_d.size else None,
            "hist_cross_seq": float(cross_d.mean()),
            "hist_ratio": float(cross_d.mean() / max(same_d.mean(), 1e-9)) if same_d.size else None,
            "adjacent_mad": float(adj_d.mean()) if adj_d.size else None,
            "random_mad": float(rnd_d.mean()),
            "redundancy": float(adj_d.mean() / max(rnd_d.mean(), 1e-9)) if adj_d.size else None,
            "target_ar_median": float(np.median(ar)) if len(ar) else None,
            "target_side_median": float(np.median(sides)) if len(sides) else None,
        }

    out = root / "_diversity_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
