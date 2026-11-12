#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R1 配准数据工程：把 IR 对齐到 RGB 坐标系，导出 ``uav_data_v2_reg``。

两种模式（实测结论已定，见下）
------------------------------
``--mode per-seq``   逐序列单一仿射（原 R1.0 规格）。**实测无法通过验收**：
                     test 上最小二乘逐序列仿射只有 11.8% 的 both 帧误差 ≤2px，
                     纯平移（序列中位）仅 3.8%。原因是两模态位移**逐帧变化**
                     （序列内 dx 标准差 20~50px），不存在序列常数刚性偏移。
                     保留本模式仅为复现该负结果。

``--mode per-frame`` 逐帧平移（R1.3 天花板 / oracle）。每帧用自身的框对直接给出
                     dx,dy，把 IR 平移到 RGB 标签位置。这是「若配准完美」的上界，
                     必须标注为 oracle，**不得进入方法表**。only_ir 帧没有 RGB 框，
                     其位移按同序列相邻 both 帧的位移线性插值（视频内位移连续）。

科学依据（2026-10-21 实测）
--------------------------
- IR 热斑落在 IR 标签处、而非 RGB 标签处：32/32 帧（`_diag_blob`），故两套标注各自正确，
  图像确实未对齐；不能用「标注错了」解释。
- 图像级配准不可用：ECC 仿射 40/40 帧不收敛，相位相关因天空低纹理退化为 0 位移。
- 位移随帧变化：序列内 dx std 20~50px，个别帧达 -226px；这是视差 + 目标运动 + 云台抖动
  的合成结果，属**动态非对齐**，正是选题三（目标级匹配）与选题四（显式 FOV 建模）的动机，
  也说明选题一的「静态坐标系选择」前提需要限定。

产物
----
``<out>/`` images(软链) / images_ir(warp 后) / labels / labels_ir / metadata(软链) /
``data.yaml`` / ``_reg_meta.json`` / ``reg_quality.tsv``

用法
----
    python scripts/reg_export_v2.py --mode per-seq   --fit-only   # 复现负结果
    python scripts/reg_export_v2.py --mode per-frame --fit-only   # 天花板质检
    python scripts/reg_export_v2.py --mode per-frame --workers 48 # 正式导出
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT_DEFAULT = "/root/rivermind-data/uav_data_v2"
OUT_DEFAULT = "/root/rivermind-data/uav_data_v2_reg"
SPLITS_DEFAULT = "train,val,test"
W, H = 640, 512


# ------------------------------------------------------------------ 基础
def read_boxes(path: Path):
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        try:
            out.append((float(s[1]), float(s[2]), float(s[3]), float(s[4])))
        except ValueError:
            continue
    return out


def match_boxes(rgb, ir, thr=0.30):
    """贪心一对一匹配（中心归一化距离升序）。both 帧多为单目标，退化为直接配对。"""
    if not rgb or not ir:
        return []
    cand = sorted((math.hypot(b[1] - a[1], b[2] - a[2]), i, j)
                  for i, b in enumerate(ir) for j, a in enumerate(rgb))
    used_i, used_r, pairs = set(), set(), []
    for d, i, j in cand:
        if d > thr:
            break
        if i in used_i or j in used_r:
            continue
        used_i.add(i)
        used_r.add(j)
        pairs.append((i, j))
    return pairs


def frame_index(stem: str) -> int:
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def box_corners(b):
    cx, cy, bw, bh = b
    x0, x1 = (cx - bw / 2) * W, (cx + bw / 2) * W
    y0, y1 = (cy - bh / 2) * H, (cy + bh / 2) * H
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def fit_affine_ls(pts_ir, pts_rgb):
    """纯最小二乘仿射（框对均为真实对应，无需 RANSAC）。"""
    src = np.asarray(pts_ir, np.float64)
    dst = np.asarray(pts_rgb, np.float64)
    X = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    A = sol.T
    proj = src @ A[:, :2].T + A[:, 2]
    err = np.linalg.norm(proj - dst, axis=1)
    return A, {"n_pairs_fit": int(len(src)), "rmse_px": float(np.sqrt((err ** 2).mean())),
               "max_err_px": float(err.max())}


def fit_affine_ransac(pts_ir, pts_rgb, ransac_px=3.0):
    if len(pts_ir) < 3:
        return None, {"n_pairs_fit": len(pts_ir), "rmse_px": float("nan")}
    M, inl = cv2.estimateAffine2D(np.asarray(pts_ir, np.float64),
                                  np.asarray(pts_rgb, np.float64), method=cv2.RANSAC,
                                  ransacReprojThreshold=ransac_px, maxIters=5000, confidence=0.999)
    if M is None:
        return None, {"n_pairs_fit": len(pts_ir), "rmse_px": float("nan")}
    proj = np.asarray(pts_ir, np.float64) @ M[:, :2].T + M[:, 2]
    err = np.linalg.norm(proj - np.asarray(pts_rgb, np.float64), axis=1)
    mask = inl.ravel() > 0 if inl is not None else np.ones(len(err), bool)
    use = err[mask] if mask.any() else err
    return M, {"n_pairs_fit": int(len(pts_ir)), "inliers": int(mask.sum()),
               "rmse_px": float(np.sqrt((use ** 2).mean())), "max_err_px": float(use.max())}


def apply_affine(A, pts):
    p = np.asarray(pts, np.float64)
    return p @ A[:, :2].T + A[:, 2]


def transform_box_line(b, A):
    t = apply_affine(A, box_corners(b))
    x0, x1 = max(0.0, t[:, 0].min()), min(W, t[:, 0].max())
    y0, y1 = max(0.0, t[:, 1].min()), min(H, t[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    if bw < 2.0 or bh < 2.0:
        return None
    return f"0 {(x0 + bw / 2) / W:.6f} {(y0 + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}"


# ------------------------------------------------------------------ 采集
def collect(root: Path, splits, match_thr, log=print):
    """返回 frames[split][stem]={seq, idx, kind, centers, ...} 与 seq_frames[seq]。"""
    frames, seq_frames = {}, defaultdict(list)
    for split in splits:
        mp = root / "metadata" / f"{split}.json"
        if not mp.is_file():
            log(f"[警告] 缺 {mp}")
            continue
        fr = json.loads(mp.read_text(encoding="utf-8")).get("frames", {})
        frames[split] = {}
        n_both = 0
        for stem, info in fr.items():
            seq = info.get("sequence", stem.rsplit("_", 1)[0])
            idx = frame_index(stem)
            kind = info.get("kind")
            rec = {"seq": seq, "idx": idx, "kind": kind, "stem": stem, "split": split,
                   "ir_c": [], "rgb_c": [], "scale": [], "n_pairs": 0, "disp": None}
            if kind == "both":
                rb = read_boxes(root / "labels" / split / f"{stem}.txt")
                ib = read_boxes(root / "labels_ir" / split / f"{stem}.txt")
                pairs = match_boxes(rb, ib, match_thr)
                if pairs:
                    for i, j in pairs:
                        rec["ir_c"].append((ib[i][0] * W, ib[i][1] * H))
                        rec["rgb_c"].append((rb[j][0] * W, rb[j][1] * H))
                        # 框尺度比：仅用于报告配准后的残余尺度失配
                        rec["scale"].append((rb[j][2] / max(ib[i][2], 1e-6),
                                             rb[j][3] / max(ib[i][3], 1e-6)))
                    rec["n_pairs"] = len(pairs)
                    d = np.asarray(rec["rgb_c"]) - np.asarray(rec["ir_c"])
                    rec["disp"] = d.mean(axis=0)
                    n_both += 1
            frames[split][stem] = rec
            seq_frames[seq].append(rec)
        log(f"[采集] split={split}: both 可用 {n_both}")
    for seq in seq_frames:
        seq_frames[seq].sort(key=lambda r: (r["split"], r["idx"]))
    return frames, seq_frames


# ------------------------------------------------------------------ per-seq
def build_per_seq(seq_frames, method, ransac_px, min_pairs, log=print):
    seq_A, seq_meta = {}, {}
    all_i, all_r = [], []
    for seq, recs in seq_frames.items():
        pi, pr = [], []
        for r in recs:
            if r["kind"] != "both" or not r["n_pairs"]:
                continue
            for c in r["ir_c"]:
                pi += [(c[0], c[1])]
            for c in r["rgb_c"]:
                pr += [(c[0], c[1])]
        all_i += pi
        all_r += pr
        if len(pi) < min_pairs:
            continue
        if method == "l2":
            A, meta = fit_affine_ls(pi, pr)
        else:
            A, meta = fit_affine_ransac(pi, pr, ransac_px)
        if A is None:
            continue
        meta["fallback"] = "none"
        seq_A[seq], seq_meta[seq] = A, meta
    A_g, meta_g = fit_affine_ls(all_i, all_r) if all_i else (np.eye(2, 3), {"n_pairs_fit": 0})
    log(f"[全局] pairs={len(all_i)} rmse={meta_g.get('rmse_px')}")
    for seq in seq_frames:
        if seq not in seq_A:
            seq_A[seq] = A_g
            seq_meta[seq] = dict(meta_g, fallback="global")
    return seq_A, seq_meta, A_g, meta_g


# ------------------------------------------------------------------ per-frame
def build_per_frame(seq_frames, log=print):
    """每帧位移 = 该帧框对中心差；非 both 帧按帧号线性插值。"""
    frame_disp = {}
    for split in {r["split"] for recs in seq_frames.values() for r in recs}:
        for seq, recs in seq_frames.items():
            rs = [r for r in recs if r["split"] == split]
            if not rs:
                continue
            anchors = [(r["idx"], r["disp"]) for r in rs
                       if r["kind"] == "both" and r["disp"] is not None]
            if not anchors:
                # 整个序列没有可用框对：退化为零位移
                for r in rs:
                    frame_disp[(r["split"], r["stem"])] = np.zeros(2)
                continue
            anchors.sort()
            xs = np.array([a[0] for a in anchors], float)
            ys = np.stack([a[1] for a in anchors])
            for r in rs:
                if r["disp"] is not None:
                    frame_disp[(r["split"], r["stem"])] = np.asarray(r["disp"], float)
                else:
                    dx = float(np.interp(r["idx"], xs, ys[:, 0]))
                    dy = float(np.interp(r["idx"], xs, ys[:, 1]))
                    frame_disp[(r["split"], r["stem"])] = np.array([dx, dy])
    disps = np.stack(list(frame_disp.values()))
    log(f"[per-frame] 位移统计 |dx| mean={np.abs(disps[:,0]).mean():.1f} "
        f"|dy| mean={np.abs(disps[:,1]).mean():.1f} max={np.abs(disps).max():.1f}px "
        f"（高分辨率图坐标）")
    return frame_disp


# ------------------------------------------------------------------ 评估
def evaluate(frames, seq_A, seq_meta, frame_disp, mode, log=print):
    """per-seq: 逐帧中心重投影误差；per-frame: 报告构造残差 + 同帧 per-seq 对照。"""
    per_frame, per_seq_frames, scales = [], [], []
    for split, items in frames.items():
        for stem, r in items.items():
            if r["kind"] != "both" or not r["n_pairs"]:
                continue
            # per-seq 对照误差
            A_s = seq_A.get(r["seq"])
            e_seq = float("nan")
            if A_s is not None:
                t = apply_affine(A_s, r["ir_c"])
                e_seq = float(np.mean(np.linalg.norm(t - np.asarray(r["rgb_c"]), axis=1)))
            # 当前模式误差
            if mode == "per-frame":
                d = frame_disp[(split, stem)]
                t = np.asarray(r["ir_c"]) + d
                e_cur = float(np.mean(np.linalg.norm(t - np.asarray(r["rgb_c"]), axis=1)))
            else:
                e_cur = e_seq
            per_frame.append({"split": split, "stem": stem, "sequence": r["seq"],
                              "err_px": round(e_cur, 4), "err_perseq_px": round(e_seq, 4)})
            per_seq_frames.append((r["seq"], e_cur, e_seq))
            for s in r["scale"]:
                scales.append(s)

    err = np.array([p["err_px"] for p in per_frame])
    err_s = np.array([p["err_perseq_px"] for p in per_frame])
    by_seq = defaultdict(list)
    for seq, e_cur, e_seq in per_seq_frames:
        by_seq[seq].append((e_cur, e_seq))

    seq_rows = []
    for seq in sorted(by_seq):
        a_cur = np.array([x[0] for x in by_seq[seq]])
        a_seq = np.array([x[1] for x in by_seq[seq]])
        meta = seq_meta.get(seq, {})
        seq_rows.append({
            "sequence": seq, "frames": int(len(a_cur)),
            "fit_rmse_px": float(meta.get("rmse_px", float("nan"))),
            "n_pairs_fit": int(meta.get("n_pairs_fit", 0)),
            "fallback": meta.get("fallback", "-"),
            "frame_err_mean_px": float(a_cur.mean()),
            "frame_err_p95_px": float(np.percentile(a_cur, 95)),
            "frame_ok_frac": float((a_cur <= 2.0).mean()),
            "perseq_frame_err_mean_px": float(a_seq.mean()) if not np.isnan(a_seq).all() else float("nan"),
            "accept": bool((a_cur <= 2.0).mean() >= 0.8),
        })

    sc = np.array(scales) if scales else np.ones((1, 2))
    report = {
        "mode": mode,
        "oracle": bool(mode == "per-frame"),
        "n_both_frames": int(len(per_frame)),
        "frame_err_mean_px": float(np.nanmean(err)),
        "frame_err_median_px": float(np.nanmedian(err)),
        "frame_err_p95_px": float(np.nanpercentile(err, 95)),
        "frame_ok_frac_2px": float(np.nanmean(err <= 2.0)),
        "perseq_frame_err_mean_px": float(np.nanmean(err_s)),
        "perseq_frame_ok_frac_2px": float(np.nanmean(err_s <= 2.0)),
        "scale_ratio_w_mean": float(sc[:, 0].mean()),
        "scale_ratio_h_mean": float(sc[:, 1].mean()),
        "n_sequences": len(seq_rows),
        "seq_ok_frac": float(np.mean([r["accept"] for r in seq_rows])) if seq_rows else 0.0,
        "sequences": seq_rows,
        "per_frame": per_frame,
    }

    log("\n=== 配准质量 ===")
    log(f"  模式                 {mode}{'（oracle，不得入方法表）' if mode=='per-frame' else ''}")
    log(f"  both 帧数            {report['n_both_frames']}")
    log(f"  本模式帧误差 mean    {report['frame_err_mean_px']:.3f}px "
        f"median {report['frame_err_median_px']:.3f} p95 {report['frame_err_p95_px']:.3f}")
    log(f"  本模式 ≤2px 帧占比   {report['frame_ok_frac_2px']*100:.2f}%  （验收线 80%）")
    log(f"  同帧 per-seq 对照     mean {report['perseq_frame_err_mean_px']:.2f}px "
        f"≤2px {report['perseq_frame_ok_frac_2px']*100:.2f}%")
    log(f"  残余框尺度比         w {report['scale_ratio_w_mean']:.3f} / h {report['scale_ratio_h_mean']:.3f}"
        f"（1.0 表示纯平移已完全对齐）")
    worst = sorted(seq_rows, key=lambda r: -r["perseq_frame_err_mean_px"])[:5]
    log("  per-seq 最差序列：")
    for r in worst:
        log(f"    {r['sequence']:<26} frames={r['frames']:<4} per-seq={r['perseq_frame_err_mean_px']:.2f}px "
            f"本模式={r['frame_err_mean_px']:.2f}px")
    return report


# ------------------------------------------------------------------ 导出
def _warp_one(args):
    src, dst, A_list = args
    A = np.asarray(A_list, np.float64)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return "read_fail"
    h, w = img.shape[:2]
    out = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return "ok" if cv2.imwrite(dst, out, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) else "write_fail"


def export(root, out, splits, frames, seq_A, frame_disp, mode, report, workers, cfg, log=print):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("images", "metadata"):
        dst = out / name
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        os.symlink(root / name, dst)
        log(f"[链接] {dst} -> {root/name}")

    n_lab = 0
    for split in splits:
        for d in ("labels", "labels_ir", "images_ir"):
            (out / d / split).mkdir(parents=True, exist_ok=True)
        for stem, r in frames.get(split, {}).items():
            A = frame_disp[(split, stem)] if mode == "per-frame" else seq_A.get(r["seq"])
            if mode == "per-frame":
                A = np.array([[1.0, 0.0, A[0]], [0.0, 1.0, A[1]]])
            if A is None:
                continue
            rgb_boxes = read_boxes(root / "labels" / split / f"{stem}.txt")
            ir_boxes = read_boxes(root / "labels_ir" / split / f"{stem}.txt")
            if r["kind"] == "only_ir":
                lines = [x for x in (transform_box_line(b, A) for b in ir_boxes) if x]
            else:
                lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in rgb_boxes]
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            lines_ir = [x for x in (transform_box_line(b, A) for b in ir_boxes) if x]
            (out / "labels_ir" / split / f"{stem}.txt").write_text(
                "\n".join(lines_ir) + ("\n" if lines_ir else ""), encoding="utf-8")
            n_lab += 1
        log(f"[标签] split={split} 写入 {len(frames.get(split, {}))} 帧")

    tasks = []
    for split in splits:
        d = root / "images_ir" / split
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            r = frames.get(split, {}).get(p.stem)
            if r is None:
                continue
            if mode == "per-frame":
                dd = frame_disp.get((split, p.stem))
                if dd is None:
                    continue
                A = np.array([[1.0, 0.0, dd[0]], [0.0, 1.0, dd[1]]])
            else:
                A = seq_A.get(r["seq"])
                if A is None:
                    continue
            tasks.append((str(p), str(out / "images_ir" / split / p.name), A.tolist()))
    log(f"[warp] 任务 {len(tasks)} 帧，进程 {workers}；训练期间会与 dataloader 抢 CPU")

    t0, fails = time.time(), 0
    with Pool(processes=workers) as pool:
        for k, st in enumerate(pool.imap_unordered(_warp_one, tasks, chunksize=64), 1):
            if st != "ok":
                fails += 1
            if k % 5000 == 0:
                log(f"    {k}/{len(tasks)}  {(time.time()-t0)/k*1000:.0f} ms/帧 失败{fails}")
    log(f"[warp] 完成 {len(tasks)}，失败 {fails}，耗时 {time.time()-t0:.1f}s")

    import yaml
    cfg_y = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8")) or {}
    cfg_y["path"] = str(out)
    for k in ("train", "val", "test"):
        cfg_y[k] = f"images/{k}"
    cfg_y["modality"] = {"rgb": "images", "ir": "images_ir"}
    cfg_y["labels"] = {"rgb": "labels", "ir": "labels_ir"}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg_y, allow_unicode=True, sort_keys=False),
                                   encoding="utf-8")
    log(f"[yaml] {out/'data.yaml'}")

    payload = {
        "version": 1, "mode": mode, "oracle": report["oracle"],
        "direction": "ir_pixel -> rgb_pixel",
        "config": cfg,
        "acceptance": {
            "frame_ok_frac_2px": report["frame_ok_frac_2px"],
            "threshold": 0.80,
            "pass": bool(report["frame_ok_frac_2px"] >= 0.80),
            "note": ("per-frame 为构造性对齐（每帧用自身标签），通过不代表存在可实现的固定配准；"
                     "per-seq 失败才是 R1.0 的真实结论"),
        },
        "summary": {k: v for k, v in report.items() if k not in ("sequences", "per_frame")},
        "sequences": [dict(r,
                           matrix=(seq_A[r["sequence"]].tolist() if r["sequence"] in seq_A else None))
                      for r in report["sequences"]],
    }
    (out / "_reg_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    with open(out / "reg_quality.tsv", "w", encoding="utf-8") as f:
        f.write("sequence\tframes\tfit_rmse_px\tn_pairs_fit\tfallback\tframe_err_mean_px\t"
                "frame_err_p95_px\tframe_ok_frac\tperseq_frame_err_mean_px\taccept\n")
        for r in report["sequences"]:
            f.write(f"{r['sequence']}\t{r['frames']}\t{r['fit_rmse_px']:.4f}\t{r['n_pairs_fit']}\t"
                    f"{r['fallback']}\t{r['frame_err_mean_px']:.4f}\t{r['frame_err_p95_px']:.4f}\t"
                    f"{r['frame_ok_frac']:.4f}\t{r['perseq_frame_err_mean_px']:.4f}\t{r['accept']}\n")
    log(f"[报告] {out/'_reg_meta.json'} / {out/'reg_quality.tsv'}")
    return payload


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="R1 配准 + uav_data_v2_reg 导出")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--splits", default=SPLITS_DEFAULT)
    ap.add_argument("--mode", choices=["per-seq", "per-frame"], default="per-frame")
    ap.add_argument("--fit", choices=["l2", "ransac"], default="l2", help="per-seq 拟合方式")
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--min-pairs", type=int, default=20)
    ap.add_argument("--match-thr", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    log = lambda m: print(m, flush=True)

    if not (root / "data.yaml").is_file():
        log(f"[错误] 缺 {root/'data.yaml'}")
        return 2
    log(f"root={root}\nout={out}\nsplits={splits} mode={args.mode} fit={args.fit}")

    frames, seq_frames = collect(root, splits, args.match_thr, log)
    if not frames:
        log("[错误] 无 both 帧")
        return 3

    seq_A, seq_meta, A_g, meta_g = build_per_seq(seq_frames, args.fit, args.ransac,
                                                 args.min_pairs, log)
    frame_disp = build_per_frame(seq_frames, log)
    report = evaluate(frames, seq_A, seq_meta, frame_disp, args.mode, log)
    report["global_matrix"] = A_g.tolist()

    if args.fit_only:
        out.mkdir(parents=True, exist_ok=True)
        (out / "_fit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        log(f"[fit-only] {out/'_fit_report.json'}")
        return 0 if report["frame_ok_frac_2px"] >= 0.80 else 4

    cfg = {"fit": args.fit, "ransac_px": args.ransac, "min_pairs": args.min_pairs,
           "match_thr": args.match_thr}
    payload = export(root, out, splits, frames, seq_A, frame_disp, args.mode,
                     report, args.workers, cfg, log)
    log(f"\n[R1] 完成 mode={args.mode} pass={payload['acceptance']['pass']}")
    return 0 if payload["acceptance"]["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
