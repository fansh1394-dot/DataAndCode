# -*- coding: utf-8 -*-
"""E13：检测侧时序一致性评估（BJ / CFR / sigma_seq）。

只读离线评估：不改模型、不重新训练、不涉部署。
输入：perframe_boxes2/boxes_<tag>.json（letterbox 空间）+ ratio_<tag>.json + 测试集 GT
输出：reports/temporal_<tag>.json（明细）+ reports/temporal_summary.csv（每 tag 一行）

指标定义见
Markdown文档/VideoFusion核验与检测侧时序一致性指标定义.md §4：
  前置  轨迹构造：相邻帧 GT 贪心匹配（代价 1-IoU，阈值 iou_track，默认 0.3），只统计 T>=3
  BJ    框抖动：BJ(t) = 1 - IoU(P_t, P_t+1)，P 为与 GT 匹配（IoU>=0.5）的最高分预测
  CFR   闪烁率：CFR = sum_t |h_t - h_t+1| / (T-1)，h = 1[IoU>=0.5 且 conf>=tau]
  sigma 序列级 AP 方差：每个序列单独算 AP50-95，取 std（只统计 GT>=5 的序列）

坐标口径与 bucket_ap.py 一致：x = (x_l - pad_x) / gain_x。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

BINS = [("lt16", 0.0, 16.0), ("b16_32", 16.0, 32.0), ("b32_64", 32.0, 64.0), ("gt64", 64.0, 1e9)]
IOU_THRS = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
MATCH_IOU = 0.5          # 预测框与 GT 的匹配阈值
CONF_MIN = 0.001         # 与 bucket_ap 的 conf_min 一致，保证 AP 可对齐
DEFAULT_CONF = 0.25      # CFR 主口径，与 R@0.25 对齐
CONF_EXTRA = (0.10, 0.50)
TRACK_THRS = (0.20, 0.30, 0.50)
SEQ_MIN_GT = 5           # sigma_seq 只对 GT>=5 的序列计算
HEAD = "fuse"

REPO = "/3250410011/root/rivermind-data/MutilModel_3485988898"


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 1e-6, None) * np.clip(iy1 - iy0, 1e-6, None)
    ua = ((a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None]
          + (b[:, 2] - b[:, 0])[None, :] * (b[:, 3] - b[:, 1])[None, :] - inter)
    return inter / np.maximum(ua, 1e-6)


def side(b):
    return float(np.sqrt(max(b[2] - b[0], 1e-6) * max(b[3] - b[1], 1e-6)))


def bin_of(s):
    for name, lo, hi in BINS:
        if lo <= s < hi:
            return name
    return BINS[-1][0]


def read_gt(path, w, h):
    out = []
    if not os.path.isfile(path):
        return out
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def load_frames(boxes, ratio, gt_dir, head):
    """逐帧预计算：GT、检测框（还原到原图像素）、GT×检测 IoU 矩阵。"""
    frames, missing = [], 0
    for fname, heads in boxes.items():
        rp = ratio.get(fname)
        if rp is None:
            missing += 1
            continue
        H, W = rp["ori_shape"]
        gx, gy = rp["gain"]
        px, py = rp["pad"]

        gts = read_gt(os.path.join(gt_dir, os.path.splitext(fname)[0] + ".txt"), W, H)
        ent = heads.get(head) or {}
        d = np.asarray(ent.get("xyxy", []), dtype=np.float32).reshape(-1, 4)
        c = np.asarray(ent.get("conf", []), dtype=np.float32).reshape(-1)
        if len(d):
            d = np.stack([(d[:, 0] - px) / gx, (d[:, 1] - py) / gy,
                          (d[:, 2] - px) / gx, (d[:, 3] - py) / gy], axis=1)
            keep = c >= CONF_MIN
            d, c = d[keep], c[keep]

        stem = os.path.splitext(fname)[0]
        ga = np.asarray(gts, dtype=np.float32).reshape(-1, 4)
        frames.append({
            "fname": fname,
            "seq": stem[:2],
            "idx": int(stem),
            "gts": ga,
            "gtbins": [bin_of(side(g)) for g in gts],
            "dets": np.asarray(d, dtype=np.float32).reshape(-1, 4),
            "conf": np.asarray(c, dtype=np.float32).reshape(-1),
            "io": iou_mat(ga, np.asarray(d, dtype=np.float32).reshape(-1, 4)),
        })
    return frames, missing


def assign_tracks(seq_frames, thr):
    """相邻帧 GT 关联。

    规则：
      1) 两帧均为单目标 -> 关联唯一，直接接续（本项目 544/614 帧属此类，且帧间
         采样间隔中位 6.5 帧，IoU 阈值会让轨迹全部断掉，必须用这条）；
      2) 其余情况 -> 按 IoU 降序贪心一对一，阈值 thr；未匹配者新建轨迹。
    返回 (tracks, stats)。tracks: {tid: [(pos, gt_idx), ...]}
    """
    tids_per_frame, next_tid = [], 0
    n_unique = n_iou = 0
    for pos, f in enumerate(seq_frames):
        n = len(f["gts"])
        prev_n = len(seq_frames[pos - 1]["gts"]) if pos > 0 else 0
        if pos == 0 or prev_n == 0:
            t = list(range(next_tid, next_tid + n))
            next_tid += n
        elif n == 1 and prev_n == 1:
            t = [tids_per_frame[pos - 1][0]]      # 唯一对应，不依赖几何阈值
            n_unique += 1
        else:
            prev = seq_frames[pos - 1]
            io = iou_mat(prev["gts"], f["gts"])
            pairs = sorted(((float(io[i, j]), i, j)
                            for i in range(io.shape[0]) for j in range(io.shape[1])),
                           key=lambda x: -x[0])
            used_p, used_c = set(), set()
            t = [-1] * n
            for v, i, j in pairs:
                if v < thr or i in used_p or j in used_c:
                    continue
                used_p.add(i)
                used_c.add(j)
                t[j] = tids_per_frame[pos - 1][i]
                n_iou += 1
            for j in range(n):
                if t[j] == -1:
                    t[j] = next_tid
                    next_tid += 1
        tids_per_frame.append(t)

    tracks = defaultdict(list)
    for pos, f in enumerate(seq_frames):
        for gi, tid in enumerate(tids_per_frame[pos]):
            tracks[tid].append((pos, gi))
    return dict(tracks), {"assoc_unique": n_unique, "assoc_iou": n_iou}


def quant(vals):
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}


def match_det(f, gi, tau, thresh=MATCH_IOU):
    """与 GT gi 匹配（IoU>=thresh 且 conf>=tau）的最高分检测框下标；无则 None。"""
    if len(f["dets"]) == 0 or gi >= f["io"].shape[0]:
        return None
    cand = np.where(f["io"][gi] >= thresh)[0]
    cand = cand[f["conf"][cand] >= tau]
    if len(cand) == 0:
        return None
    return int(cand[np.argmax(f["conf"][cand])])


def ap_from(recs, npos):
    """101 点插值 AP（与 bucket_ap.py 一致）。"""
    if not recs or npos <= 0:
        return 0.0
    arr = sorted(recs, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in arr])
    fp = np.cumsum([r[2] for r in arr])
    rc = tp / npos
    pr = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rc, [1.0]])
    mpre = np.concatenate([[0.0], pr, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(idx, mrec, mpre)))


def ap_records(frames, target_bin):
    """按序列聚合 (conf, tp, fp) 记录；target_bin=None 时统计全部 GT。"""
    recs = defaultdict(lambda: defaultdict(list))   # seq -> thr -> records
    npos = defaultdict(int)
    for f in frames:
        seq = f["seq"]
        gtb = f["gtbins"]
        idx_gt = list(range(len(gtb))) if target_bin is None else \
            [i for i, b in enumerate(gtb) if b == target_bin]
        npos[seq] += len(idx_gt)
        if not idx_gt or len(f["dets"]) == 0:
            continue
        idx_set = set(idx_gt)
        others = [] if target_bin is None else [i for i in range(len(gtb)) if i not in idx_set]
        for thr in IOU_THRS:
            used = set()
            for di in np.argsort(-f["conf"]):
                best, bj = -1.0, -1
                for gi in idx_gt:
                    if gi in used:
                        continue
                    v = f["io"][gi, di]
                    if v > best:
                        best, bj = float(v), gi
                if best >= thr:
                    used.add(bj)
                    recs[seq][thr].append((float(f["conf"][di]), 1, 0))
                elif others and any(f["io"][gi, di] >= thr for gi in others):
                    continue          # ignore，与 bucket_ap 的跨桶规则一致
                else:
                    recs[seq][thr].append((float(f["conf"][di]), 0, 1))
    return recs, npos


def seq_ap_stats(frames, target_bin):
    recs, npos = ap_records(frames, target_bin)
    per_seq = {}
    for seq in npos:
        if npos[seq] < SEQ_MIN_GT:
            continue
        aps = [ap_from(recs.get(seq, {}).get(thr, []), npos[seq]) for thr in IOU_THRS]
        per_seq[seq] = {"nGT": int(npos[seq]), "AP50-95": float(np.mean(aps))}
    vals = np.asarray([per_seq[s]["AP50-95"] for s in sorted(per_seq)], dtype=np.float64)
    if len(vals) >= 2:
        k = max(1, int(np.ceil(0.1 * len(vals))))
        stat = {"n_seq": int(len(vals)), "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)), "worst10_mean": float(np.mean(np.sort(vals)[:k]))}
    else:
        stat = {"n_seq": int(len(vals)), "mean": float(vals.mean()) if len(vals) else 0.0,
                "std": 0.0, "worst10_mean": float(vals.mean()) if len(vals) else 0.0}
    return per_seq, stat


def warp_box(box, gt_a, gt_b):
    """按 GT 位移把框从 a 帧搬到 b 帧：中心平移 + 尺度缩放（不含预测自身的漂移）。"""
    wa = max(gt_a[2] - gt_a[0], 1e-6)
    ha = max(gt_a[3] - gt_a[1], 1e-6)
    wb = max(gt_b[2] - gt_b[0], 1e-6)
    hb = max(gt_b[3] - gt_b[1], 1e-6)
    sx, sy = wb / wa, hb / ha
    ca = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    cb = ((gt_a[0] + gt_a[2]) / 2, (gt_a[1] + gt_a[3]) / 2)
    db = ((gt_b[0] + gt_b[2]) / 2, (gt_b[1] + gt_b[3]) / 2)
    w = (box[2] - box[0]) * sx
    h = (box[3] - box[1]) * sy
    cx = ca[0] - cb[0] + db[0]
    cy = ca[1] - cb[1] + db[1]
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def temporal_metrics(by_seq, tracks, min_len, conf_tau):
    bj = defaultdict(list)
    bj_gap1 = defaultdict(list)
    bj_comp = defaultdict(list)
    cfr = defaultdict(list)
    drift = defaultdict(list)
    gaps, len_dist = [], []
    n_used = matched = total = 0

    for (_seq, _tid), path in tracks.items():
        if len(path) < min_len:
            continue
        sf = by_seq[_seq]
        n_used += 1
        len_dist.append(len(path))

        # BJ 用任意检测（IoU>=0.5，不限 conf），与定义 §4.2 一致
        det_any = [match_det(sf[p], gi, CONF_MIN) for p, gi in path]
        # CFR 的 h 用 conf>=tau
        det_tau = [match_det(sf[p], gi, conf_tau) for p, gi in path]
        h = [1 if d is not None else 0 for d in det_tau]
        total += len(path)
        matched += sum(h)

        sides = [side(sf[p]["gts"][gi]) for p, gi in path]
        bin_t = bin_of(float(np.median(sides)))

        for i in range(len(path) - 1):
            pa, ga = path[i]
            pb, gb = path[i + 1]
            gap = sf[pb]["idx"] - sf[pa]["idx"]
            gaps.append(gap)
            if det_any[i] is not None and det_any[i + 1] is not None:
                da, db = det_any[i], det_any[i + 1]
                ba = sf[pa]["dets"][da]
                bb = sf[pb]["dets"][db]
                gt_a = sf[pa]["gts"][ga]
                gt_b = sf[pb]["gts"][gb]
                b = bin_of(sides[i])
                raw = 1.0 - float(iou_mat(ba[None], bb[None])[0, 0])
                bj[b].append(raw)
                bj["all"].append(raw)
                if gap == 1:                       # 真·相邻帧，不受采样间隔影响
                    bj_gap1[b].append(raw)
                    bj_gap1["all"].append(raw)
                comp = 1.0 - float(iou_mat(warp_box(ba, gt_a, gt_b)[None], bb[None])[0, 0])
                bj_comp[b].append(comp)
                bj_comp["all"].append(comp)

        cf = sum(abs(h[i + 1] - h[i]) for i in range(len(h) - 1)) / max(len(path) - 1, 1)
        cfr[bin_t].append(cf)
        cfr["all"].append(cf)

        for i in range(len(path) - 1):
            if det_tau[i] is not None and det_tau[i + 1] is not None:
                c0 = float(sf[path[i][0]]["conf"][det_tau[i]])
                c1 = float(sf[path[i + 1][0]]["conf"][det_tau[i + 1]])
                if max(c0, c1) > 0:
                    drift[bin_of(sides[i])].append(abs(c1 - c0) / max(c0, c1))
                    drift["all"].append(abs(c1 - c0) / max(c0, c1))

    return {
        "tracks_total": int(len(tracks)),
        "tracks_used": n_used,
        "track_len": quant(len_dist),
        "coverage": float(matched / max(total, 1)),
        "frame_gap": quant(gaps),
        "gap_share_1": float(np.mean([g == 1 for g in gaps])) if gaps else 0.0,
        "BJ": {b: quant(v) for b, v in bj.items()},
        "BJ_gap1": {b: quant(v) for b, v in bj_gap1.items()},
        "BJ_comp": {b: quant(v) for b, v in bj_comp.items()},
        "CFR": {b: quant(v) for b, v in cfr.items()},
        "conf_drift": {b: quant(v) for b, v in drift.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--head", default=HEAD)
    ap.add_argument("--boxes", default=os.path.join(REPO, "perframe_boxes2"))
    ap.add_argument("--gt", default="/3250410011/root/rivermind-data/mendeley_split/labels/test")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "reports"))
    ap.add_argument("--ratio-fallback", default=os.path.join(REPO, "perframe_boxes", "ratio_pad.json"),
                    help="部分 tag 的 ratio_<tag>.json 为空，回退到共享比例文件（与 bucket_ap 一致）")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--iou-track", type=float, default=0.30)
    a = ap.parse_args()

    boxes = json.load(open(os.path.join(a.boxes, f"boxes_{a.tag}.json")))
    rp_tag = os.path.join(a.boxes, f"ratio_{a.tag}.json")
    ratio = json.load(open(rp_tag)) if os.path.isfile(rp_tag) else {}
    if len(boxes) and len(ratio) < 0.5 * len(boxes):
        print(f"ratio_{a.tag}.json 条目 {len(ratio)}/<{len(boxes)}，回退到 {a.ratio_fallback}")
        ratio = json.load(open(a.ratio_fallback))
    frames, missing = load_frames(boxes, ratio, a.gt, a.head)
    print(f"tag={a.tag} head={a.head} frames={len(frames)} missing_ratio={missing}")

    by_seq = defaultdict(list)
    for f in frames:
        by_seq[f["seq"]].append(f)
    for s in by_seq:
        by_seq[s].sort(key=lambda x: x["idx"])
    print(f"sequences={len(by_seq)} GT={sum(len(f['gts']) for f in frames)}")

    sens, main_metrics = {}, None
    for thr in sorted(set(TRACK_THRS) | {a.iou_track}):
        tracks = {}
        assoc = {"assoc_unique": 0, "assoc_iou": 0}
        for s, sf in by_seq.items():
            tr, st = assign_tracks(sf, thr)
            assoc["assoc_unique"] += st["assoc_unique"]
            assoc["assoc_iou"] += st["assoc_iou"]
            for tid, path in tr.items():
                tracks[(s, tid)] = path
        m = temporal_metrics(by_seq, tracks, a.min_len, a.conf)
        m["assoc"] = assoc
        sens[f"{thr:.2f}"] = m
        if abs(thr - a.iou_track) < 1e-9:
            main_metrics = m

    per_seq, stat_all = seq_ap_stats(frames, None)
    _, stat_lt16 = seq_ap_stats(frames, "lt16")

    report = {
        "tag": a.tag, "head": a.head,
        "n_frames": len(frames), "missing_ratio": missing, "n_sequences": len(by_seq),
        "config": {"iou_track": a.iou_track, "match_iou": MATCH_IOU,
                   "conf_tau": a.conf, "min_len": a.min_len, "conf_min": CONF_MIN},
        "gap_note": "测试帧按文件名编号抽样，相邻可用帧可能不连续（见 frame_gap）",
        "metrics": main_metrics,
        "sensitivity": sens,
        "seq_ap": {"all": stat_all, "lt16": stat_lt16, "per_seq": per_seq},
        "conf_extra": {},
    }
    for tau in CONF_EXTRA:
        tracks = {}
        for s, sf in by_seq.items():
            tr, _ = assign_tracks(sf, a.iou_track)
            for tid, path in tr.items():
                tracks[(s, tid)] = path
        report["conf_extra"][f"{tau:.2f}"] = temporal_metrics(by_seq, tracks, a.min_len, tau)["CFR"]

    os.makedirs(a.out_dir, exist_ok=True)
    jp = os.path.join(a.out_dir, f"temporal_{a.tag}.json")
    json.dump(report, open(jp, "w"), indent=2, ensure_ascii=False)
    print("saved ->", jp)

    def g(b, key, field="median"):
        return main_metrics[key].get(b, {}).get(field)

    def r4(v):
        return round(v, 4) if v is not None else ""

    row = {
        "tag": a.tag,
        "tracks": main_metrics["tracks_used"],
        "coverage": round(main_metrics["coverage"], 4),
        "BJ_med_lt16": r4(g("lt16", "BJ", "median")),
        "BJ_p90_lt16": r4(g("lt16", "BJ", "p90")),
        "BJgap1_med_lt16": r4(g("lt16", "BJ_gap1", "median")),
        "BJcomp_med_lt16": r4(g("lt16", "BJ_comp", "median")),
        "BJcomp_p90_lt16": r4(g("lt16", "BJ_comp", "p90")),
        "CFR_lt16": r4(g("lt16", "CFR", "mean")),
        "sigma_seq": round(stat_all["std"], 4),
        "worst10": round(stat_all["worst10_mean"], 4),
        "sigma_seq_lt16": round(stat_lt16["std"], 4),
        "n_seq": stat_all["n_seq"],
        "iou_track": a.iou_track,
    }
    cp = os.path.join(a.out_dir, "temporal_summary.csv")
    old = []
    if os.path.isfile(cp):
        with open(cp, newline="", encoding="utf-8") as fh:
            old = [r for r in csv.DictReader(fh) if r.get("tag") != a.tag]
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in row})
        w.writerow(row)
    print("saved ->", cp)

    print(f"\n{'bucket':>8} {'nBJ':>5} {'BJ_med':>8} {'BJ_p90':>8} "
          f"{'nCmp':>5} {'BJc_med':>8} {'BJc_p90':>8} {'nCFR':>5} {'CFR':>7}")
    for b in ["lt16", "b16_32", "b32_64", "gt64", "all"]:
        d = main_metrics["BJ"].get(b, {})
        if d.get("n", 0) == 0:
            continue
        c = main_metrics["BJ_comp"].get(b, {})
        e = main_metrics["CFR"].get(b, {})
        print(f"{b:>8} {d['n']:>5} {d['median']:>8.4f} {d['p90']:>8.4f} "
              f"{c.get('n', 0):>5} {c.get('median', 0):>8.4f} {c.get('p90', 0):>8.4f} "
              f"{e.get('n', 0):>5} {e.get('mean', 0):>7.4f}")
    print(f"\ntracks={main_metrics['tracks_used']} coverage={main_metrics['coverage']:.4f} "
          f"sigma_seq={stat_all['std']:.4f}(n={stat_all['n_seq']}) "
          f"sigma_seq_lt16={stat_lt16['std']:.4f}")
    gp = main_metrics["frame_gap"]
    print(f"frame_gap median={gp.get('median')} mean={gp.get('mean', 0):.2f} "
          f"p90={gp.get('p90')} share_gap1={main_metrics['gap_share_1']:.3f}")
    ac = main_metrics["assoc"]
    print(f"assoc unique={ac['assoc_unique']} iou={ac['assoc_iou']} (阈值 {a.iou_track})")


if __name__ == "__main__":
    main()
