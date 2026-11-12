# -*- coding: utf-8 -*-
"""检验 images_ir 上的标注框（labels_ir）是否真的框住了无人机热成像目标。

用户观察：红外抽帧图上的框与无人机实际位置"大多有差别"。
需要回答：是 (a) 标注框画错了（框没落在目标上），还是 (b) 只是两模态坐标系不同
（框在 IR 图上是对的，但与 RGB 框本就不该重合）。

判别设计（不依赖热成像的白热/黑热极性，纯 CPU）
---------------------------------------------
对每一帧的每个框，计算「框内均值 − 框外环均值」的绝对对比：
    c(box) = |mean(inside) − mean(ring)|
若框真的落在目标上，框内与紧邻环应有明显强度差；若框是空画在天空/地面上，
这个差会退化到与"随便挪一挪的框"相当。

因此设**位移对照**：把同一个框平移 ±0.6 个框宽/高（8 个方向）得到 c_shift，
取 max 作为该帧的对照上界。判定量：
    ratio = c(真框) / max_shifts c(平移框)
  ratio > 1 表示真框比任何平移框都更"有对比" → 框大概率落在目标上；
  ratio < 1 表示真框还不如挪一挪 → 框大概率不在目标上（标注错误）。

对照模态（关键）：同一次计算同时套用
    * IR 框（labels_ir）      → 检验 IR 标注
    * RGB 框（labels）投到 IR 图上 → 检验「如果只是视差，那 RGB 框在 IR 图上应该也不太有对比」
    * IR 框投到 RGB 图上      → 反向交叉，作对称参照

另生成可视化拼图（原图、无增强）：裁剪覆盖两框并集的区域，画
    绿=IR 框   蓝=RGB 框   红=全图最亮连通块
人眼可直接判定无人机是否落在绿框内。

用法（服务器）：
  python scripts/check_ir_label_alignment.py --n 600 --montage-n 20 \
      --out-dir _d2p1/ir_align
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

ROOT = Path("/root/rivermind-data/uav_data_v2")
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

IR_DIR_CANDIDATES = [
    ROOT / "images_ir" / "val",
    ROOT / "images" / "ir" / "val",
    ROOT / "images_ir" / "val".replace("val", "val"),
]
RGB_DIR_CANDIDATES = [
    ROOT / "images" / "val",
    ROOT / "images_rgb" / "val",
]
LAB_IR_CANDIDATES = [ROOT / "labels_ir" / "val", ROOT / "labels_ir" / "val"]
LAB_RGB_CANDIDATES = [ROOT / "labels" / "val"]


def first_existing(cands):
    for c in cands:
        if c.is_dir():
            return c
    return None


def build_index(d: Path) -> dict:
    idx = {}
    if d is None:
        return idx
    for p in d.iterdir():
        if p.suffix.lower() in EXT and p.is_file():
            idx[p.stem] = p
    return idx


def load_gray(p: Path):
    if cv2 is None:
        return None
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)


def load_boxes(p: Path):
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        try:
            _cls, cx, cy, w, h = (float(x) for x in s[:5])
        except ValueError:
            continue
        out.append((cx, cy, w, h))
    return out


def box_px(b, W, H, dx=0.0, dy=0.0):
    """归一化 cx,cy,w,h -> 像素 x1,y1,x2,y2（可整体平移 dx,dy 个框宽/高）。"""
    cx, cy, w, h = b
    cx = cx + dx * w
    cy = cy + dy * h
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return x1, y1, x2, y2


def contrast(im, b, W, H, exp=2.0, dx=0.0, dy=0.0):
    """|框内均值 − 环均值|；环 = 框外扩 exp 倍后挖掉框。返回 (c, inner_mean, ring_mean)。"""
    x1, y1, x2, y2 = box_px(b, W, H, dx, dy)
    xi1, yi1 = int(round(x1)), int(round(y1))
    xi2, yi2 = int(round(x2)), int(round(y2))
    xi1, yi1 = max(0, xi1), max(0, yi1)
    xi2, yi2 = min(W, xi2), min(H, yi2)
    iw, ih = xi2 - xi1, yi2 - yi1
    if iw < 1 or ih < 1:
        return None
    inner = im[yi1:yi2, xi1:xi2]
    mx, my = (xi1 + xi2) / 2.0, (yi1 + yi2) / 2.0
    hw, hh = iw * exp / 2.0, ih * exp / 2.0
    ox1, oy1 = max(0, int(round(mx - hw))), max(0, int(round(my - hh)))
    ox2, oy2 = min(W, int(round(mx + hw))), min(H, int(round(my + hh)))
    outer = im[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return None
    rx1, ry1 = xi1 - ox1, yi1 - oy1
    m = np.ones(outer.shape, dtype=np.float32)
    ry2, rx2 = ry1 + ih, rx1 + iw
    ry2c, rx2c = min(ry2, outer.shape[0]), min(rx2, outer.shape[1])
    if ry1 < outer.shape[0] and rx1 < outer.shape[1]:
        m[ry1:ry2c, rx1:rx2c] = 0.0
    n = float(m.sum())
    if n < 1:
        return None
    ring_mean = float((outer * m).sum() / n)
    inner_mean = float(inner.mean())
    return abs(inner_mean - ring_mean), inner_mean, ring_mean


SHIFTS = [(0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6), (0.6, 0.6), (0.6, -0.6), (-0.6, 0.6), (-0.6, -0.6)]


def ratio_for(im, b, W, H, exp=2.0):
    """真框对比 / 最大平移框对比。返回 (ratio, c_label, c_shift_max)。"""
    r0 = contrast(im, b, W, H, exp=exp)
    if r0 is None:
        return None
    c0 = r0[0]
    cs = [contrast(im, b, W, H, exp=exp, dx=sx, dy=sy) for sx, sy in SHIFTS]
    cs = [c[0] for c in cs if c is not None]
    if not cs:
        return None
    cmax = max(cs)
    return c0 / max(cmax, 1e-6), c0, cmax


def iou_norm(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 1e-12 else 0.0


def tightness(im, b, W, H, pct=99.0, min_area=4):
    """框「紧不紧」：在框周围找与之关联的最亮连通块，返回 (IoU(框,块), 块面积/框面积)。

    IoU 高 → 框紧贴无人机热斑；IoU 低且面积比小 → 框比目标大很多（框松）。
    """
    if cv2 is None:
        return None
    thr = float(np.percentile(im, pct))
    mask = (im >= thr).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    x1, y1, x2, y2 = box_px(b, W, H)
    ex, ey = 0.5 * (x2 - x1), 0.5 * (y2 - y1)
    sx1, sy1, sx2, sy2 = x1 - ex, y1 - ey, x2 + ex, y2 + ey
    best, bestA = None, 0
    for i in range(1, n):
        cx, cy = cent[i]
        if not (sx1 <= cx <= sx2 and sy1 <= cy <= sy2):
            continue
        ar = int(stats[i, cv2.CC_STAT_AREA])
        if ar < min_area or ar > 0.5 * W * H:
            continue
        if ar > bestA:
            bestA, best = ar, i
    if best is None:
        return None
    bx = int(stats[best, cv2.CC_STAT_LEFT])
    by = int(stats[best, cv2.CC_STAT_TOP])
    bw = int(stats[best, cv2.CC_STAT_WIDTH])
    bh = int(stats[best, cv2.CC_STAT_HEIGHT])
    blobb = ((bx + bw / 2) / W, (by + bh / 2) / H, bw / W, bh / H)
    boxarea = max((x2 - x1) * (y2 - y1), 1e-6)
    return iou_norm(b, blobb), bestA / boxarea


def bright_blob(im, W, H, pct=99.5, min_area=6):
    """全图最亮阈值下的最大连通块 bbox（归一化 cx,cy,w,h）或 None。"""
    if cv2 is None:
        return None
    thr = float(np.percentile(im, pct))
    mask = (im >= thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    best, bestA = None, 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area or a > 0.5 * W * H:
            continue
        if a > bestA:
            bestA, best = a, i
    if best is None:
        return None
    x, y = int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP])
    w, h = int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT])
    return ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)


def draw_box(img, b, color, thick=1):
    x1, y1, x2, y2 = [int(round(v)) for v in box_px(b, img.shape[1], img.shape[0])]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="统计抽样的 both 帧数")
    ap.add_argument("--montage-n", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=210)
    ap.add_argument("--out-dir", default="_d2p1/ir_align")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path("/3250410011/root/rivermind-data/MutilModel_3485988898") / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_dir = first_existing(IR_DIR_CANDIDATES)
    rgb_dir = first_existing(RGB_DIR_CANDIDATES)
    lab_ir = first_existing(LAB_IR_CANDIDATES)
    lab_rgb = first_existing(LAB_RGB_CANDIDATES)
    print(f"[目录] images_ir={ir_dir}")
    print(f"[目录] images   ={rgb_dir}")
    print(f"[目录] labels_ir={lab_ir}")
    print(f"[目录] labels   ={lab_rgb}")
    if ir_dir is None or lab_ir is None:
        print("[错误] 未找到 IR 图像或 IR 标签目录")
        return 1

    ir_idx = build_index(ir_dir)
    rgb_idx = build_index(rgb_dir)
    print(f"[目录] IR 图像 {len(ir_idx)} 张，RGB 图像 {len(rgb_idx)} 张")

    meta = json.loads((ROOT / "metadata" / "val.json").read_text(encoding="utf-8"))
    frames = meta.get("frames", {})

    def is_both(info):
        return bool(info.get("valid_rgb")) and bool(info.get("valid_ir"))

    both = sorted([s for s, i in frames.items() if is_both(i)])
    print(f"[帧] both={len(both)}  （抽样上限 {args.n}）")

    step = max(1, len(both) // max(args.n, 1))
    sample = both[::step][: args.n]
    print(f"[抽样] 实际取 {len(sample)} 帧，步长 {step}")

    rows = []
    skipped = {"no_img": 0, "no_lab_ir": 0, "no_lab_rgb": 0}
    for stem in sample:
        ip = ir_idx.get(stem)
        if ip is None:
            skipped["no_img"] += 1
            continue
        ib = load_boxes(lab_ir / f"{stem}.txt")
        rb = load_boxes(lab_rgb / f"{stem}.txt")
        if not ib:
            skipped["no_lab_ir"] += 1
            continue
        if not rb:
            skipped["no_lab_rgb"] += 1
            continue
        im = load_gray(ip)
        if im is None:
            skipped["no_img"] += 1
            continue
        H, W = im.shape[:2]
        # 每帧取第一个框（both 帧基本恰 1 个目标）
        r_ir_ir = ratio_for(im, ib[0], W, H)      # IR 框 在 IR 图
        r_rgb_ir = ratio_for(im, rb[0], W, H)     # RGB 框 在 IR 图
        if r_ir_ir is None:
            continue
        rec = {
            "stem": stem,
            "W": W, "H": H,
            "ir_box": ib[0], "rgb_box": rb[0],
            "ratio_ir": r_ir_ir[0], "c_ir": r_ir_ir[1], "ctrl_ir": r_ir_ir[2],
            "ratio_rgb_on_ir": r_rgb_ir[0] if r_rgb_ir else None,
            "c_rgb_on_ir": r_rgb_ir[1] if r_rgb_ir else None,
        }
        t_ir = tightness(im, ib[0], W, H)
        if t_ir is not None:
            rec["tight_iou_ir"], rec["tight_arearatio_ir"] = t_ir[0], t_ir[1]
        t_rgbb = tightness(im, rb[0], W, H)
        if t_rgbb is not None:
            rec["tight_iou_rgb_on_ir"] = t_rgbb[0]
        # RGB 框在 RGB 图上（自洽性 sanity：RGB 标注应当高对比）
        if stem in rgb_idx:
            rgbi = load_gray(rgb_idx[stem])
            if rgbi is not None:
                r_rgb_rgb = ratio_for(rgbi, rb[0], rgbi.shape[1], rgbi.shape[0])
                rec["ratio_rgb_on_rgb"] = r_rgb_rgb[0] if r_rgb_rgb else None
                del rgbi
        rows.append(rec)
        del im

    print(f"[抽样] 有效 {len(rows)} 帧；跳过 {skipped}")

    def stat(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        if not v:
            return None
        v = np.asarray(v, dtype=np.float64)
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
            "frac_gt1": float((v > 1.0).mean()),
            "frac_gt12": float((v > 1.2).mean()),
            "frac_lt08": float((v < 0.8).mean()),
        }

    print("\n================ 对比/平移比 ratio ==================")
    print("ratio>1 = 真框比任何平移框都更有对比（框大概率落在目标上）")
    for key, name in (
        ("ratio_ir", "IR框@IR图"),
        ("ratio_rgb_on_ir", "RGB框@IR图"),
        ("ratio_rgb_on_rgb", "RGB框@RGB图(自洽 sanity)"),
    ):
        s = stat(key)
        if s is None:
            print(f"  {name:24s} 无数据")
            continue
        print(f"  {name:24s} n={s['n']:4d} 均值={s['mean']:.3f} 中位={s['median']:.3f} "
              f"p10={s['p10']:.3f} p90={s['p90']:.3f} | >1 占 {s['frac_gt1']:.1%} "
              f">1.2 占 {s['frac_gt12']:.1%} <0.8 占 {s['frac_lt08']:.1%}")

    # 亮度极性：框内 vs 图均值
    inner_over = []
    for r in rows:
        c = r.get("c_ir")
        if c is not None:
            inner_over.append(c)
    if inner_over:
        print(f"\n[绝对值] IR框内-环对比 c 均值={np.mean(inner_over):.2f} 中位={np.median(inner_over):.2f}")

    print("\n================ 框紧度（与热斑的重合） ================")
    ti = [r["tight_iou_ir"] for r in rows if r.get("tight_iou_ir") is not None]
    ta = [r["tight_arearatio_ir"] for r in rows if r.get("tight_arearatio_ir") is not None]
    tr = [r["tight_iou_rgb_on_ir"] for r in rows if r.get("tight_iou_rgb_on_ir") is not None]
    if ti:
        ti = np.asarray(ti)
        print(f"  IR框 vs 其内热斑  IoU: n={ti.size} 均值={ti.mean():.3f} 中位={np.median(ti):.3f} "
              f"| >0.3 占 {(ti > 0.3).mean():.1%} >0.5 占 {(ti > 0.5).mean():.1%} <0.1 占 {(ti < 0.1).mean():.1%}")
    if ta:
        ta = np.asarray(ta)
        print(f"  热斑面积 / IR框面积:      均值={ta.mean():.3f} 中位={np.median(ta):.3f} "
              f"| <0.3 占 {(ta < 0.3).mean():.1%}")
    if tr:
        tr = np.asarray(tr)
        print(f"  RGB框 vs IR图热斑  IoU: 均值={tr.mean():.3f} 中位={np.median(tr):.3f} "
              f"| >0.3 占 {(tr > 0.3).mean():.1%} <0.1 占 {(tr < 0.1).mean():.1%}")

    # ---------------- 可视化拼图 ----------------
    if cv2 is not None and args.montage_n > 0:
        mrows = []
        idx = np.linspace(0, max(len(sample) - 1, 0), min(args.montage_n, len(sample))).astype(int)
        for i in idx:
            stem = sample[int(i)]
            ip, rp = ir_idx.get(stem), rgb_idx.get(stem)
            if ip is None:
                continue
            im = load_gray(ip)
            if im is None:
                continue
            H, W = im.shape[:2]
            ib = load_boxes(lab_ir / f"{stem}.txt")
            rb = load_boxes(lab_rgb / f"{stem}.txt")
            if not ib or not rb:
                continue
            # 覆盖两框并集、外扩，并保证最小视野
            a = ib[0]
            b = rb[0]
            xs = [a[0] - a[2] / 2, a[0] + a[2] / 2, b[0] - b[2] / 2, b[0] + b[2] / 2]
            ys = [a[1] - a[3] / 2, a[1] + a[3] / 2, b[1] - b[3] / 2, b[1] + b[3] / 2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            span = max(x2 - x1, y2 - y1, 0.08)
            span *= 1.8
            x1, y1 = cx - span / 2, cy - span / 2
            x2, y2 = cx + span / 2, cy + span / 2
            px1, py1 = max(0, int(x1 * W)), max(0, int(y1 * H))
            px2, py2 = min(W, int(x2 * W)), min(H, int(y2 * H))
            if px2 - px1 < 8 or py2 - py1 < 8:
                continue
            crop = im[py1:py2, px1:px2].copy()
            cH, cW = crop.shape[:2]
            crop3 = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            def to_crop(bb):
                bx1, by1, bx2, by2 = box_px(bb, W, H)
                return [int(round((bx1 / W) * cW)), int(round((by1 / H) * cH)),
                        int(round((bx2 / W) * cW)), int(round((by2 / H) * cH))]

            for bb, col in ((a, (0, 255, 0)), (b, (255, 128, 0))):
                q = to_crop(bb)
                cv2.rectangle(crop3, (q[0], q[1]), (q[2], q[3]), col, 1)
            blob = bright_blob(im, W, H)
            if blob is not None:
                q = to_crop(blob)
                cv2.rectangle(crop3, (q[0], q[1]), (q[2], q[3]), (0, 0, 255), 1)

            rr_ = ratio_for(im, a, W, H)
            tag = f"r={rr_[0]:.2f}" if rr_ else "r=?"
            cell = cv2.resize(crop3, (args.cell, args.cell), interpolation=cv2.INTER_NEAREST)
            cv2.putText(cell, tag, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            mrows.append(cell)

        if mrows:
            cols = args.cols
            rr = int(np.ceil(len(mrows) / cols))
            canvas = np.zeros((rr * args.cell, cols * args.cell, 3), dtype=np.uint8)
            for k, cell in enumerate(mrows):
                r, c = k // cols, k % cols
                canvas[r * args.cell:(r + 1) * args.cell, c * args.cell:(c + 1) * args.cell] = cell
            p = out_dir / "ir_align_montage.jpg"
            cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print("\n[图] 拼图已保存: %s  (%dx%d)" % (p, canvas.shape[1], canvas.shape[0]))
            print("     绿=IR框 labels_ir   蓝=RGB框 labels   红=全图最亮连通块")

    # 落盘 json 便于复核
    (out_dir / "ir_align_stats.json").write_text(
        json.dumps({"rows": rows[:200], "stats_ratio_ir": stat("ratio_ir"),
                    "stats_ratio_rgb_on_ir": stat("ratio_rgb_on_ir"),
                    "stats_ratio_rgb_on_rgb": stat("ratio_rgb_on_rgb"),
                    "stats_tight_iou_ir": stat("tight_iou_ir"),
                    "skipped": skipped, "n_sample": len(sample)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[输出] {out_dir/'ir_align_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
