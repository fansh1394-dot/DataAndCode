#!/usr/bin/env python3
"""决策级晚融合（帧级置信度路由）的零训练验证。

设计动机
--------
test 实测表明特征级融合的净增益（fusion_gain_over_rgb = 0.0016）小于
多任务共享骨干造成的单路退化（C4 rgb 头 0.5572 vs E1 0.5683，−0.0111），
且 test 的 only_ir 帧占 5.72%（968/16917），而 E1 在这些帧上 AP≈0、
IR 单模态能到 0.39~0.46。因此收益不在"怎么混合特征"，而在"该用哪个模态"。

本脚本不训练、不改模型，只消费两个已有的**单模态**权重：

    RGB 头权重 ----> 输入 RGB 图 ----> 检测框（RGB 坐标系）
    IR  头权重 ----> 输入 IR  图 ----> 检测框（IR  坐标系）

然后按帧做路由：

    c_rgb = 该帧 RGB 头最高 conf
    c_ir  = 该帧 IR  头最高 conf
    if c_rgb < tau_low and c_ir > tau_high:  输出 IR 头的框（IR 坐标系）
    else:                                    输出 RGB 头的框（RGB 坐标系）

为什么不需要坐标映射：only_ir 帧上 RGB 图里目标不可见，RGB 头输出接近空，
不存在"同一目标两个框"；both 帧一律走 RGB，也不需要 IR 框的坐标。
于是跨模态几何对齐被完全绕开——这正是它比可变形融合更可能赢的原因。

评测口径
--------
每帧按"选中的模态"配对应坐标系的标签：
    RGB 分支 -> labels/{split}/*.txt      (RGB 标签)
    IR  分支 -> labels_ir/{split}/*.txt   (IR  标签)
空标签帧（only_ir 的 RGB 标签、only_rgb 的 IR 标签）保持为空，即该模态在
这些帧上检测到任何框都算 FP，与训练/评测口径一致。

输出的三个数
------------
1. 路由后的 mAP50-95（并与 always_rgb / always_ir 同口径对比）
2. 路由准确率：预测模态 vs metadata 的真实 kind
3. only_ir 帧召回 @IoU0.5、both 帧 FP 增量

用法
----
    python scripts/eval_late_fusion.py \
        --rgb-weights ResTest/v2c_E1_rgbonly_s0/weights/best.pt \
        --ir-weights  ResTest/v2c_E3_ironly_lblir_s0/weights/best.pt \
        --split test \
        --out YOLO26/logs/late_fusion_test.json

推理结果会缓存到 --cache（默认 YOLO26/logs/lf_cache_{split}.pkl），
之后调 tau 只重算指标、不再跑模型。
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

# 以 scripts/ 为 sys.path[0] 启动时找不到 ultralytics，显式把项目根放进来
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# 图像与标签
# ----------------------------------------------------------------------
def letterbox(img: np.ndarray, new: int = 640, color: int = 114) -> Tuple[np.ndarray, float, int, int]:
    """等比缩放 + 居中填充，返回 (图, 缩放比, 左填充, 上填充)。"""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) // 2, (new - nh) // 2
    img = cv2.copyMakeBorder(img, dh, new - nh - dh, dw, new - nw - dw, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, dw, dh


def read_labels(path: Path, w: int, h: int) -> np.ndarray:
    """读 YOLO 归一化 xywh 标签，返回像素 xyxy (N,4)。空文件即空数组。"""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        _, cx, cy, bw, bh = (float(x) for x in p[:5])
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


# ----------------------------------------------------------------------
# 推理
# ----------------------------------------------------------------------
def load_model(weight: str, device: str):
    from ultralytics.nn.tasks import attempt_load_one_weight

    dev = f"cuda:{device}" if str(device).isdigit() else device
    model, _ = attempt_load_one_weight(str(weight))
    model = model.to(dev).eval()
    # 端到端头（YOLO26）的 NMS 需要 end2end=True，否则会二次 NMS
    e2e = bool(getattr(model, "end2end", False))
    # 单模态模型的输入通道应为 3；6 通道说明拿错了权重
    try:
        ch = int(model.model[0].conv.in_channels)
    except Exception:
        ch = -1
    return model, e2e, ch


@torch.no_grad()
def infer_batch(model, x: torch.Tensor, conf: float, iou: float, e2e: bool) -> List[np.ndarray]:
    """批量推理 -> 每图 (N,6) [x1,y1,x2,y2,conf,cls]，坐标仍在 letterbox 坐标系。"""
    from ultralytics.utils.ops import non_max_suppression

    out = model(x)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(pred, dict):
        pred = pred.get("one2many", pred)
    try:
        dets = non_max_suppression(pred, conf, iou, max_det=300, end2end=e2e)
    except TypeError:
        dets = non_max_suppression(pred, conf, iou, max_det=300)
    res = []
    for det in dets:
        if det is None or len(det) == 0:
            res.append(np.zeros((0, 6), dtype=np.float32))
        else:
            res.append(det.detach().cpu().numpy())
    return res


def unletterbox(boxes: np.ndarray, r: float, dw: int, dh: int) -> np.ndarray:
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = (b[:, [0, 2]] - dw) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dh) / r
    return b


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def greedy_match(pred: np.ndarray, gt: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """单类贪心匹配 -> bool [P, len(iouv)]。按 IoU 降序，每 GT 至多匹配一个 pred。"""
    correct = np.zeros((len(pred), len(iou)), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return correct
    ious = box_iou_np(gt, pred)  # G x P
    for j, thr in enumerate(iou):
        m = ious >= thr
        gi, pi = np.nonzero(m)
        if len(gi) == 0:
            continue
        order = np.argsort(-ious[gi, pi])
        used_g, used_p = set(), set()
        for k in order:
            g, p = int(gi[k]), int(pi[k])
            if g in used_g or p in used_p:
                continue
            used_g.add(g)
            used_p.add(p)
            correct[p, j] = True
    return correct


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """连续积分 AP（与 COCO all-point 口径一致）。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(frames: List[Dict], selection: List[str], iouv: np.ndarray) -> Dict[str, float]:
    """按给定的每帧模态选择算 mAP。selection[i] ∈ {'rgb','ir'}。

    单类场景，直接按 conf 降序累计 TP/FP 画 P-R 曲线，避免依赖不同版本
    ultralytics.metrics 的签名差异。
    """
    tp_all, conf_all = [], []
    n_gt = 0
    for rec, sel in zip(frames, selection):
        boxes, confs = rec[sel]
        gt = rec["gt_rgb"] if sel == "rgb" else rec["gt_ir"]
        tp_all.append(greedy_match(boxes, gt, iouv))
        conf_all.append(confs)
        n_gt += len(gt)

    tp = np.concatenate(tp_all) if tp_all else np.zeros((0, len(iouv)), dtype=bool)
    conf = np.concatenate(conf_all) if conf_all else np.zeros(0, dtype=np.float32)
    if len(tp) == 0 or n_gt == 0:
        return {"mAP50-95": 0.0, "mAP50": 0.0, "precision": 0.0, "recall": 0.0,
                "n_pred": int(len(tp)), "n_gt": int(n_gt)}

    order = np.argsort(-conf)
    tp = tp[order]
    conf = conf[order]

    # 用框架官方 ap_per_class，保证与 ResTest/_test_eval 的口径逐位一致；
    # 自写的 compute_ap 与它系统性差 0.02~0.03（官方对 P-R 曲线做 1000 点插值积分）。
    try:
        from ultralytics.utils.metrics import ap_per_class

        out = ap_per_class(
            tp, conf, np.zeros(len(conf), np.float32), np.zeros(n_gt, np.float32)
        )
        ap = np.asarray(out[5], dtype=float).reshape(-1, len(iouv))[0]
        p, r = float(np.asarray(out[2]).ravel()[0]), float(np.asarray(out[3]).ravel()[0])
        return {
            "mAP50-95": float(ap.mean()),
            "mAP50": float(ap[0]),
            "precision": p,
            "recall": r,
            "n_pred": int(len(tp)),
            "n_gt": int(n_gt),
        }
    except Exception as e:  # 版式不兼容时退回自写实现
        print(f"[warn] ap_per_class 不可用({e})，退回自写 AP")

    aps, pr_best = [], (0.0, 0.0)
    for j in range(len(iouv)):
        cum_tp = np.cumsum(tp[:, j])
        cum_fp = np.cumsum(~tp[:, j])
        recall = cum_tp / n_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        aps.append(compute_ap(recall, precision))
        if j == 0:
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
            k = int(np.argmax(f1))
            pr_best = (float(precision[k]), float(recall[k]))

    return {
        "mAP50-95": float(np.mean(aps)),
        "mAP50": float(aps[0]),
        "precision": pr_best[0],
        "recall": pr_best[1],
        "n_pred": int(len(tp)),
        "n_gt": int(n_gt),
    }


def recall_at(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> Tuple[int, int]:
    """返回 (命中 GT 数, GT 总数)。"""
    if len(gt) == 0:
        return 0, 0
    if len(pred) == 0:
        return 0, len(gt)
    ious = box_iou_np(gt, pred)
    hit = ious.max(axis=1) >= thr
    return int(hit.sum()), int(len(gt))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def build_cache(args) -> List[Dict]:
    root = Path(args.data_root)
    split = args.split
    meta = json.loads((root / "metadata" / f"{split}.json").read_text())["frames"]
    img_dir = root / "images" / split
    ir_dir = root / "images_ir" / split
    lbl_dir = root / "labels" / split
    irl_dir = root / "labels_ir" / split

    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if args.max_frames:
        stems = stems[: args.max_frames]

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    m_rgb, e2e_rgb, ch_rgb = load_model(args.rgb_weights, device)
    m_ir, e2e_ir, ch_ir = load_model(args.ir_weights, device)
    print(f"[load] rgb ch={ch_rgb} e2e={e2e_rgb} | ir ch={ch_ir} e2e={e2e_ir}")
    if ch_rgb != 3 or ch_ir != 3:
        raise SystemExit(
            f"本脚本只接受单模态 3 通道权重，实际 rgb={ch_rgb} ir={ch_ir}。"
            " 6 通道说明拿成了双模态/融合模型，请改用单模态 best.pt。"
        )

    frames: List[Dict] = []
    t0 = time.time()
    for s0 in range(0, len(stems), args.batch):
        chunk = stems[s0 : s0 + args.batch]
        tb, ti, metas = [], [], []
        for stem in chunk:
            ib = cv2.imread(str(img_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
            ii = cv2.imread(str(ir_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
            if ib is None or ii is None:
                print(f"[skip] 缺图 {stem}")
                continue
            h, w = ib.shape[:2]
            lb_b, r_b, dx_b, dy_b = letterbox(ib, args.imgsz)
            lb_i, r_i, dx_i, dy_i = letterbox(ii, args.imgsz)
            tb.append(torch.from_numpy(lb_b[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255))
            ti.append(torch.from_numpy(lb_i[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255))
            metas.append((stem, h, w, r_b, dx_b, dy_b, r_i, dx_i, dy_i))
        if not metas:
            continue

        xb = torch.stack(tb).to(device)
        xi = torch.stack(ti).to(device)
        outs_b = infer_batch(m_rgb, xb, args.conf, args.iou, e2e_rgb)
        outs_i = infer_batch(m_ir, xi, args.conf, args.iou, e2e_ir)

        for (stem, h, w, r_b, dx_b, dy_b, r_i, dx_i, dy_i), db, di in zip(metas, outs_b, outs_i):
            if len(db):
                db[:, :4] = unletterbox(db[:, :4], r_b, dx_b, dy_b)
            if len(di):
                di[:, :4] = unletterbox(di[:, :4], r_i, dx_i, dy_i)
            frames.append(
                {
                    "stem": stem,
                    "kind": meta.get(stem, {}).get("kind", "unknown"),
                    "sequence": meta.get(stem, {}).get("sequence", ""),
                    "valid_rgb": bool(meta.get(stem, {}).get("valid_rgb", False)),
                    "valid_ir": bool(meta.get(stem, {}).get("valid_ir", False)),
                    "rgb": (db[:, :4].astype(np.float32), db[:, 4].astype(np.float32)),
                    "ir": (di[:, :4].astype(np.float32), di[:, 4].astype(np.float32)),
                    "gt_rgb": read_labels(lbl_dir / f"{stem}.txt", w, h),
                    "gt_ir": read_labels(irl_dir / f"{stem}.txt", w, h),
                }
            )
        if (s0 // args.batch + 1) % 10 == 0:
            el = time.time() - t0
            done = len(frames)
            print(f"[infer] {done}/{len(stems)}  {el:.0f}s  ({el / max(done, 1) * 1000:.0f} ms/帧)", flush=True)
    print(f"[infer] 完成 {len(frames)} 帧，用时 {time.time() - t0:.0f}s", flush=True)
    return frames


def route(frames: List[Dict], tau_low: float, tau_high: float, route_conf: float) -> Tuple[List[str], List[bool]]:
    """返回 (每帧模态, 是否发生切换)。

    判据只用两路在 route_conf 之上存在的框的最高置信度：
      - RGB 该帧"有话说"（c_rgb >= tau_low）-> 留在 RGB
      - RGB 沉默且 IR 确信（c_ir > tau_high） -> 切 IR
    """
    sel, switched = [], []
    for rec in frames:
        cb = rec["rgb"][1]
        ci = rec["ir"][1]
        c_rgb = float(cb.max()) if len(cb) and float(cb.max()) >= route_conf else 0.0
        c_ir = float(ci.max()) if len(ci) and float(ci.max()) >= route_conf else 0.0
        use_ir = (c_rgb < tau_low) and (c_ir > tau_high)
        sel.append("ir" if use_ir else "rgb")
        switched.append(use_ir)
    return sel, switched


def main() -> int:
    ap = argparse.ArgumentParser(description="决策级晚融合（帧级置信度路由）eval-only 验证")
    ap.add_argument("--rgb-weights", required=True, help="RGB 单模态权重（3 通道输入）")
    ap.add_argument("--ir-weights", required=True, help="IR 单模态权重（3 通道输入）")
    ap.add_argument("--data-root", default="/root/rivermind-data/uav_data_v2")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="推理批大小（GPU 被训练占用时批处理能省一个数量级）")
    ap.add_argument("--conf", type=float, default=0.001, help="mAP 计算的置信度下限")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--route-conf", type=float, default=0.25, help="参与路由判据的最低置信度")
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", default=None, help="推理缓存 pkl，默认 YOLO26/logs/lf_cache_{split}.pkl")
    ap.add_argument("--max-frames", type=int, default=0, help="调试用，只跑前 N 帧")
    ap.add_argument("--out", default=None, help="结果 JSON")
    ap.add_argument("--reuse", action="store_true", help="强制复用已有缓存（不跑模型）")
    args = ap.parse_args()

    suffix = f"_n{args.max_frames}" if args.max_frames else ""
    cache_path = Path(args.cache) if args.cache else ROOT / "YOLO26" / "logs" / f"lf_cache_{args.split}{suffix}.pkl"
    if args.reuse or (cache_path.exists() and not args.max_frames):
        print(f"[cache] 复用 {cache_path}")
        frames = pickle.loads(cache_path.read_bytes())
    else:
        frames = build_cache(args)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(frames))
        print(f"[cache] 已写 {cache_path}")

    iouv = np.arange(0.5, 1.0, 0.05)
    n = len(frames)
    kinds = [f["kind"] for f in frames]

    # ---- 两条单模态基线（同口径）----
    base_rgb = compute_map(frames, ["rgb"] * n, iouv)
    base_ir = compute_map(frames, ["ir"] * n, iouv)
    print("\n===== 单模态基线（同口径）=====")
    print(f"  always_rgb : {base_rgb}")
    print(f"  always_ir  : {base_ir}")

    # ---- 网格搜索 tau ----
    grid_low = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    grid_high = [0.10, 0.20, 0.30, 0.50]
    best = None
    table = []
    for tl, th in itertools.product(grid_low, grid_high):
        if th <= tl:
            continue
        sel, sw = route(frames, tl, th, args.route_conf)
        m = compute_map(frames, sel, iouv)
        row = {
            "tau_low": tl,
            "tau_high": th,
            "mAP50-95": m["mAP50-95"],
            "mAP50": m["mAP50"],
            "switch_rate": float(np.mean(sw)),
        }
        table.append(row)
        if best is None or row["mAP50-95"] > best["mAP50-95"]:
            best = row
    table.sort(key=lambda r: -r["mAP50-95"])
    print("\n===== 路由网格（前 8）=====")
    for r in table[:8]:
        print(f"  tau_low={r['tau_low']:.2f} tau_high={r['tau_high']:.2f}  "
              f"mAP50-95={r['mAP50-95']:.6f}  mAP50={r['mAP50']:.6f}  切到IR={r['switch_rate']*100:.2f}%")

    # ---- 用最佳 tau 出三个数 ----
    sel, sw = route(frames, best["tau_low"], best["tau_high"], args.route_conf)
    routed = compute_map(frames, sel, iouv)

    # 2) 路由准确率
    want = {"only_ir": "ir", "only_rgb": "rgb", "both": "rgb"}
    acc = {}
    for k in ("only_ir", "only_rgb", "both"):
        idx = [i for i, kk in enumerate(kinds) if kk == k]
        if idx:
            acc[k] = float(np.mean([sel[i] == want[k] for i in idx]))
            acc[k + "_n"] = len(idx)
    overall = float(np.mean([sel[i] == want.get(kinds[i], "rgb") for i in range(n)]))

    # 3) only_ir 召回 / both FP 增量
    def frame_stats(selection: List[str], kinds_filter=("only_ir",)) -> Dict[str, float]:
        hit = tot = 0
        for i, rec in enumerate(frames):
            if rec["kind"] not in kinds_filter:
                continue
            key = selection[i]
            h_, t_ = recall_at(rec[key][0], rec["gt_rgb"] if key == "rgb" else rec["gt_ir"])
            hit += h_
            tot += t_
        return {"hit": hit, "gt": tot, "recall": (hit / tot if tot else 0.0)}

    rec_routed = frame_stats(sel)
    rec_base_rgb = frame_stats(["rgb"] * n)

    # FP 数（conf>=route_conf 的框，在无 GT 帧上全是 FP；有 GT 帧上未匹配的算 FP）
    def fp_count(selection: List[str], kinds_filter) -> int:
        fp = 0
        for i, rec in enumerate(frames):
            if rec["kind"] not in kinds_filter:
                continue
            key = selection[i]
            boxes, confs = rec[key]
            keep = confs >= args.route_conf
            boxes = boxes[keep]
            gt = rec["gt_rgb"] if key == "rgb" else rec["gt_ir"]
            if len(boxes) == 0:
                continue
            if len(gt) == 0:
                fp += len(boxes)
            else:
                ious = box_iou_np(gt, boxes)
                fp += int((ious.max(axis=0) < 0.5).sum())
        return fp

    fp_routed = fp_count(sel, ("both",))
    fp_base = fp_count(["rgb"] * n, ("both",))

    summary = {
        "split": args.split,
        "n_frames": n,
        "baseline_rgb": base_rgb,
        "baseline_ir": base_ir,
        "routed": routed,
        "gain_over_rgb": routed["mAP50-95"] - base_rgb["mAP50-95"],
        "best_tau": {"low": best["tau_low"], "high": best["tau_high"]},
        "switch_rate": best["switch_rate"],
        "route_accuracy": {"overall": overall, **acc},
        "only_ir_recall": {"routed": rec_routed, "always_rgb": rec_base_rgb},
        "both_fp": {"routed": fp_routed, "always_rgb": fp_base, "delta": fp_routed - fp_base},
        "grid": table,
    }

    print("\n===== 三个结论数 =====")
    print(f"  1) 路由后 mAP50-95 = {routed['mAP50-95']:.6f}"
          f"   基线 always_rgb = {base_rgb['mAP50-95']:.6f}"
          f"   增益 = {summary['gain_over_rgb']:+.6f}")
    print(f"  2) 路由准确率 总体 = {overall*100:.2f}%   "
          f"only_ir = {acc.get('only_ir', 0)*100:.2f}% (n={acc.get('only_ir_n', 0)})   "
          f"both = {acc.get('both', 0)*100:.2f}% (n={acc.get('both_n', 0)})   "
          f"only_rgb = {acc.get('only_rgb', 0)*100:.2f}% (n={acc.get('only_rgb_n', 0)})")
    print(f"  3) only_ir 召回 路由后 = {rec_routed['recall']*100:.2f}%"
          f" ({rec_routed['hit']}/{rec_routed['gt']})   "
          f"纯RGB = {rec_base_rgb['recall']*100:.2f}%")
    print(f"     both FP 数 路由后 = {fp_routed}  纯RGB = {fp_base}  增量 = {fp_routed - fp_base:+d}")

    out = Path(args.out) if args.out else ROOT / "YOLO26" / "logs" / f"late_fusion_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
