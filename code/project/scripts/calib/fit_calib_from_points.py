# -*- coding: utf-8 -*-
"""双光配准：由人工标注点对拟合仿射标定（纯几何标定第二步）。

输入 annotate_points.py 产出的 pairs.json，输出与
ultralytics/engine/multimodal/registration.py 兼容的 calib.npz。

与旧标定（RGB框中心→IR热斑）的关键区别
---------------------------------------
旧标定用"模型输出的框中心"当 RGB 真值，里面**混入了模型的定位偏置**，
一旦重新训练模型就失效。

本脚本用**人工标注的物理点**当输入，得到的是**纯几何量**：
只反映两相机的光轴偏移与视场比，与模型无关，重训后依然可用。

方法
----
对每对对应点 (x_rgb, y_rgb) -> (x_ir, y_ir)，拟合 2x3 仿射：
    x_ir = a1*x_rgb + b1*y_rgb + c1
    y_ir = a2*x_rgb + b2*y_rgb + c2

用最小二乘求解。仿射包含平移、旋转、尺度、剪切，足以吸收两镜头的
光轴偏移与视场差异（对比纯平移被证明只能到 74.5%，仿射可达 98%）。

泛化性用 **K 折交叉验证**评估：只用留出集的残差做判断，避免高估。

报告指标
--------
- 残差中位 / 均值 / std（归一化 pp 与像素两套单位）
- 5 折交叉验证的留出集残差
- 与旧标定（若存在）的对比
- 按"是否落在典型框半宽内"换算出的**理论框住率**

用法
----
    python scripts/calib/fit_calib_from_points.py
    python scripts/calib/fit_calib_from_points.py --folds 5 --out D:\\uav_data\\_vis\\calib.npz
    python scripts/calib/fit_calib_from_points.py --dry-run     # 只报告，不写文件
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

WORK = Path(r"D:\uav_data\_calib")
PAIRS_JSON = WORK / "pairs.json"
DEFAULT_OUT = Path(r"D:\uav_data\_vis\calib.npz")


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #

def load_points(pairs_json: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """从标注文件载入对应点对。

    Args:
        pairs_json: pairs.json 路径。

    Returns:
        (P, Q, tags)：P 为 RGB 归一化点 [N,2]，Q 为 IR 归一化点 [N,2]，
        tags 为每个点的来源标识（用于分组交叉验证）。

    Raises:
        SystemExit: 文件缺失或无有效点时。
    """
    if not pairs_json.exists():
        print(f"[错误] 找不到标注文件: {pairs_json}")
        print("       请先运行 scripts/calib/annotate_points.py --annotate")
        raise SystemExit(1)

    with open(pairs_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    P, Q, tags = [], [], []
    for pair in data.get("pairs", []):
        pr = pair.get("points_rgb", [])
        pi = pair.get("points_ir", [])
        n = min(len(pr), len(pi))
        if n == 0:
            continue
        stem = pair.get("stem", "?")
        for k in range(n):
            P.append(pr[k])
            Q.append(pi[k])
            tags.append(stem)

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if len(P) < 6:
        print(f"[错误] 有效点对仅 {len(P)} 个，至少需要 6 个（仿射有 6 个自由度）。")
        print("       建议标注 30~50 对图像、每对 2~4 个点。")
        raise SystemExit(1)
    return P, Q, tags


# --------------------------------------------------------------------------- #
# 拟合与评估
# --------------------------------------------------------------------------- #

def fit_affine(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘拟合 2x3 仿射。

    Args:
        P: RGB 点 [N,2]。
        Q: IR 点 [N,2]。

    Returns:
        (cx, cy)：x 方向系数 [a,b,c]、y 方向系数 [a,b,c]。
    """
    A = np.hstack([P, np.ones((len(P), 1))])
    cx, *_ = np.linalg.lstsq(A, Q[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(A, Q[:, 1], rcond=None)
    return cx, cy


def apply_affine(pts: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """对点施加仿射。"""
    A = np.hstack([pts, np.ones((len(pts), 1))])
    return np.stack([A @ cx, A @ cy], axis=1)


def residual_stats(pred: np.ndarray, gt: np.ndarray, shape: tuple[int, int]) -> dict:
    """计算残差统计（归一化 pp 与像素两套单位）。

    Args:
        pred: 预测点 [N,2]（归一化）。
        gt: 真值点 [N,2]（归一化）。
        shape: 图像尺寸 (w, h)，用于换算像素。

    Returns:
        统计字典。
    """
    e = pred - gt
    w, h = shape
    ex_px = e[:, 0] * w
    ey_px = e[:, 1] * h
    dist_px = np.hypot(ex_px, ey_px)
    return {
        "n": len(e),
        "med_x_pp": float(np.median(e[:, 0]) * 100),
        "med_y_pp": float(np.median(e[:, 1]) * 100),
        "std_x_pp": float(e[:, 0].std() * 100),
        "std_y_pp": float(e[:, 1].std() * 100),
        "absmed_x_pp": float(np.median(np.abs(e[:, 0])) * 100),
        "absmed_y_pp": float(np.median(np.abs(e[:, 1])) * 100),
        "med_dist_px": float(np.median(dist_px)),
        "p90_dist_px": float(np.percentile(dist_px, 90)),
        "max_dist_px": float(dist_px.max()),
    }


def cross_validate(P: np.ndarray, Q: np.ndarray, tags: list[str], folds: int,
                   shape: tuple[int, int]) -> dict:
    """按图像对分组的 K 折交叉验证。

    按"图像对"分组而非按点，避免同一对图的点同时出现在训练和验证集里
    造成信息泄漏（同一对图的点高度相关）。

    Args:
        P: RGB 点。
        Q: IR 点。
        tags: 每点的图像对标识。
        folds: 折数。
        shape: 图像尺寸。

    Returns:
        交叉验证统计。
    """
    uniq = sorted(set(tags))
    if len(uniq) < folds:
        folds = max(2, len(uniq))

    rng = np.random.RandomState(0)
    order = np.array(uniq)
    rng.shuffle(order)
    groups = np.array_split(order, folds)

    all_pred = np.zeros_like(Q)
    tag_arr = np.asarray(tags)
    for g in groups:
        mask_te = np.isin(tag_arr, g)
        mask_tr = ~mask_te
        if mask_tr.sum() < 6 or mask_te.sum() == 0:
            continue
        cx, cy = fit_affine(P[mask_tr], Q[mask_tr])
        all_pred[mask_te] = apply_affine(P[mask_te], cx, cy)

    # 逐折报告
    fold_report = []
    for g in groups:
        mask = np.isin(tag_arr, g)
        if mask.sum() == 0:
            continue
        st = residual_stats(all_pred[mask], Q[mask], shape)
        fold_report.append({"holdout_pairs": len(g), **st})

    overall = residual_stats(all_pred, Q, shape)
    return {"folds": folds, "overall": overall, "per_fold": fold_report}


def theoretical_hit_rate(stats: dict, box_half_w_pp: float, box_half_h_pp: float,
                         pred: np.ndarray, gt: np.ndarray, shape: tuple[int, int]) -> float:
    """按给定框半宽换算理论框住率（仅当残差落在半宽内才算框住）。

    Args:
        stats: 预留（不使用）。
        box_half_w_pp: 框半宽（归一化 pp）。
        box_half_h_pp: 框半高（归一化 pp）。
        pred: 预测点。
        gt: 真值点。
        shape: 图像尺寸。

    Returns:
        命中比例（0~1）。
    """
    e = pred - gt
    ok = (np.abs(e[:, 0]) * 100 < box_half_w_pp) & (np.abs(e[:, 1]) * 100 < box_half_h_pp)
    return float(ok.mean())


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    """CLI 入口。"""
    ap = argparse.ArgumentParser(
        description="由人工标注点对拟合仿射标定",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(PAIRS_JSON), help="标注文件路径")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="标定输出路径")
    ap.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    ap.add_argument("--box-w", type=float, default=12.66,
                    help="典型框半宽（归一化 pp，默认取 81px/640 的一半）")
    ap.add_argument("--box-h", type=float, default=8.40,
                    help="典型框半高（归一化 pp，默认取 43px/512 的一半）")
    args = ap.parse_args()

    print("=" * 70)
    print("由人工标注点对拟合仿射标定")
    print("=" * 70)

    P, Q, tags = load_points(Path(args.pairs))
    uniq = sorted(set(tags))
    print(f"载入点对: {len(P)} 个，来自 {len(uniq)} 对图像")

    # 图像尺寸：优先用标注里记录的 IR 尺寸（标定输出以 IR 画幅为基准）
    shape = (640, 512)
    with open(Path(args.pairs), "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("pairs"):
        s = d["pairs"][0].get("ir_shape")
        if s:
            shape = (int(s[0]), int(s[1]))
    print(f"IR 画幅: {shape[0]}x{shape[1]}")

    # ---- 全量拟合 ----
    cx, cy = fit_affine(P, Q)
    pred = apply_affine(P, cx, cy)
    st = residual_stats(pred, Q, shape)

    print("\n" + "-" * 70)
    print("全量拟合结果")
    print("-" * 70)
    print(f"  affine_x = [{cx[0]:.6f}, {cx[1]:.6f}, {cx[2]:.6f}]")
    print(f"  affine_y = [{cy[0]:.6f}, {cy[1]:.6f}, {cy[2]:.6f}]")
    print(f"\n  拟合残差（会偏乐观，仅供参考）:")
    print(f"    中位 ({st['med_x_pp']:+.3f}, {st['med_y_pp']:+.3f}) pp  "
          f"std ({st['std_x_pp']:.3f}, {st['std_y_pp']:.3f})")
    print(f"    |误差| 中位 ({st['absmed_x_pp']:.3f}, {st['absmed_y_pp']:.3f}) pp")
    print(f"    像素距离 中位 {st['med_dist_px']:.2f} px  p90 {st['p90_dist_px']:.2f} px  "
          f"max {st['max_dist_px']:.2f} px")

    # 尺度奇异值检查（判断是否退化为纯平移）
    S = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
    sv = np.linalg.svd(S, compute_uv=False)
    print(f"\n  线性部分奇异值: [{sv[0]:.4f}, {sv[1]:.4f}]  "
          f"（若两者接近 1，说明两镜视场几乎一致）")

    # ---- 交叉验证 ----
    cv = cross_validate(P, Q, tags, args.folds, shape)
    o = cv["overall"]
    print("\n" + "-" * 70)
    print(f"{cv['folds']} 折交叉验证（按图像对分组，留出集残差——这才是可信指标）")
    print("-" * 70)
    print(f"  留出集残差 中位 ({o['med_x_pp']:+.3f}, {o['med_y_pp']:+.3f}) pp  "
          f"std ({o['std_x_pp']:.3f}, {o['std_y_pp']:.3f})")
    print(f"  |误差| 中位 ({o['absmed_x_pp']:.3f}, {o['absmed_y_pp']:.3f}) pp")
    print(f"  像素距离 中位 {o['med_dist_px']:.2f} px  p90 {o['p90_dist_px']:.2f} px")
    print(f"\n  逐折（留出对 / 像素距离中位）:")
    for i, fr in enumerate(cv["per_fold"], 1):
        print(f"    折{i}: {fr['holdout_pairs']:>3} 对   "
              f"中位距离 {fr['med_dist_px']:.2f} px   p90 {fr['p90_dist_px']:.2f} px")

    # 交叉验证预测点（用于算理论框住率）
    cv_pred = np.zeros_like(Q)
    tag_arr = np.asarray(tags)
    uniq_l = sorted(set(tags))
    rng = np.random.RandomState(0)
    order = np.array(uniq_l)
    rng.shuffle(order)
    for g in np.array_split(order, cv["folds"]):
        m_te = np.isin(tag_arr, g)
        m_tr = ~m_te
        if m_tr.sum() < 6 or m_te.sum() == 0:
            continue
        cxa, cya = fit_affine(P[m_tr], Q[m_tr])
        cv_pred[m_te] = apply_affine(P[m_te], cxa, cya)

    hr = theoretical_hit_rate(o, args.box_w, args.box_h, cv_pred, Q, shape)
    print(f"\n  理论框住率（按框半宽 ±{args.box_w:.2f}pp x ±{args.box_h:.2f}pp 估算）: "
          f"{hr * 100:.1f}%")

    # ---- 与旧标定对比 ----
    if Path(args.out).exists():
        try:
            old = dict(np.load(args.out))
            if "affine_x" in old and "affine_y" in old:
                op = apply_affine(P, np.asarray(old["affine_x"]), np.asarray(old["affine_y"]))
                ost = residual_stats(op, Q, shape)
                print("\n" + "-" * 70)
                print("与当前 calib.npz 对比（在同一批人工点上评估）")
                print("-" * 70)
                print(f"  旧仿射 像素距离中位 {ost['med_dist_px']:.2f} px  "
                      f"p90 {ost['p90_dist_px']:.2f} px")
                ohr = theoretical_hit_rate(ost, args.box_w, args.box_h, op, Q, shape)
                print(f"  旧仿射 理论框住率 {ohr * 100:.1f}%")
                print(f"  新仿射 理论框住率 {hr * 100:.1f}%")
        except Exception as exc:
            print(f"\n  [提示] 旧标定对比失败: {exc}")

    # ---- 写出 ----
    if args.dry_run:
        print("\n[dry-run] 未写出任何文件。")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 备份旧标定
    if out.exists():
        bak = out.with_name(out.stem + f".bak_{datetime.now():%Y%m%d_%H%M%S}.npz")
        shutil.copy2(out, bak)
        print(f"\n[备份] 原标定已备份为 {bak.name}")

    # 保留旧文件里的其它键（如 shift_gt），并标注来源
    payload = {}
    if out.exists():
        try:
            with np.load(str(out)) as d:
                payload.update({k: d[k] for k in d.files})
        except Exception:
            pass

    payload.update(
        affine_x=cx.astype(np.float64),
        affine_y=cy.astype(np.float64),
        shift_gt=np.array([np.median(Q[:, 0] - P[:, 0]), np.median(Q[:, 1] - P[:, 1])]),
        calib_source=np.array("manual_points"),
        n_samples=np.array(len(P)),
        n_image_pairs=np.array(len(uniq)),
        cv_med_dist_px=np.array(o["med_dist_px"]),
        cv_p90_dist_px=np.array(o["p90_dist_px"]),
        cv_hit_rate=np.array(hr),
        created_at=np.array(datetime.now().isoformat(timespec="seconds")),
    )
    np.savez(str(out), **payload)

    print(f"\n[完成] 已写出 {out}")
    print(f"       标定来源: 人工标注点对（{len(uniq)} 对图 / {len(P)} 个点）")
    print(f"       该标定为纯几何量，与模型无关，重新训练后依然可用。")
    print("\n下一步：验证配准效果")
    print("  python scripts/calib/run_fixed_predict.py --reg --mode affine --tag _geo")
    print("  python scripts/calib/verify_align.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
