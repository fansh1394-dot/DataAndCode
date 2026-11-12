"""诊断：IR 标签框与红外目标亮斑的实际偏移（在 letterbox 之后的 640 画布上）。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/root/rivermind-data/uav_data_v2")
SPLIT = "train"
IMGS = DATA / "images" / SPLIT
IMGS_IR = DATA / "images_ir" / SPLIT
LBL_IR = DATA / "labels_ir" / SPLIT

names = sorted(p.name for p in IMGS.glob("*.jpg"))
rng = np.random.default_rng(0)
pick = rng.choice(len(names), size=min(600, len(names)), replace=False)

rows = []
for k in pick:
    stem = Path(names[k]).stem
    lp = LBL_IR / f"{stem}.txt"
    if not lp.exists():
        continue
    lines = [l.split() for l in lp.read_text().strip().splitlines() if l.strip()]
    if len(lines) != 1:
        continue
    c, x, y, w, h = [float(v) for v in lines[0][:5]]
    rgb = cv2.imread(str(IMGS / names[k]))
    irp = IMGS_IR / names[k]
    if rgb is None or not irp.exists():
        continue
    ir = cv2.imread(str(irp), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        continue
    H0, W0 = ir.shape[:2]
    if (rgb.shape[1], rgb.shape[0]) != (W0, H0):
        continue
    # letterbox: 等比缩放到 640 内并居中补边
    r = min(640 / H0, 640 / W0)
    nw, nh = int(round(W0 * r)), int(round(H0 * r))
    dw, dh = (640 - nw) / 2, (640 - nh) / 2
    irs = cv2.resize(ir, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640), 0, np.uint8)
    canvas[int(round(dh)):int(round(dh)) + nh, int(round(dw)):int(round(dw)) + nw] = irs

    # 标签（归一化, 原图坐标系）→ letterbox 画布
    cx, cy, bw, bh = x * W0, y * H0, w * W0, h * H0
    cx_l, cy_l = cx * r + dw, cy * r + dh
    bw_l, bh_l = bw * r, bh * r

    # 局部窗口内找亮斑
    pad = 60
    x1 = int(max(0, cx_l - bw_l / 2 - pad)); x2 = int(min(640, cx_l + bw_l / 2 + pad))
    y1 = int(max(0, cy_l - bh_l / 2 - pad)); y2 = int(min(640, cy_l + bh_l / 2 + pad))
    win = canvas[y1:y2, x1:x2].astype(np.float32)
    if win.size == 0:
        continue
    # 目标亮斑：局部最大值（目标在 IR 中是亮斑）
    idx = int(np.argmax(win))
    wy, wx = divmod(idx, win.shape[1])
    px, py = x1 + wx, y1 + wy
    peak = float(win[wy, wx])
    med = float(np.median(win))
    if peak - med < 40:  # 无明显亮斑，跳过
        continue
    # 质心：峰值 60% 以上像素的加权中心（亮度加权，抗单像素噪声）
    th = peak * 0.6
    m = win >= th
    yy, xx = np.nonzero(m)
    wts = win[yy, xx]
    cx_c = x1 + float((xx * wts).sum() / wts.sum())
    cy_c = y1 + float((yy * wts).sum() / wts.sum())
    rows.append((stem, cx_l, cy_l, bw_l, bh_l, cx_c, cy_c, cx_c - cx_l, cy_c - cy_l, peak - med,
             W0, H0, w, h))

if not rows:
    print("no samples")
    sys.exit(1)

dy = np.array([r[8] for r in rows]); dx = np.array([r[7] for r in rows])
bw = np.array([r[3] for r in rows]); bh = np.array([r[4] for r in rows])
print(f"n={len(rows)}")
print(f"dx: mean={dx.mean():.2f} median={np.median(dx):.2f} std={dx.std():.2f}")
print(f"dy: mean={dy.mean():.2f} median={np.median(dy):.2f} std={dy.std():.2f}")
print(f"|offset| mean={np.hypot(dx, dy).mean():.2f} px   within-box rate="
      f"{np.mean((np.abs(dx) <= bw / 2) & (np.abs(dy) <= bh / 2)):.3f}")
print(f"box w median={np.median(bw):.1f} h median={np.median(bh):.1f}")
print(f"dy > +5px (blob below box) ratio = {np.mean(dy > 5):.3f}")
q = np.percentile(np.hypot(dx, dy), [50, 75, 90, 95, 99])
print("|offset| p50/p75/p90/p95/p99 =", np.round(q, 1))
h0v = np.array([r[11] for r in rows], dtype=np.float32)
w0v = np.array([r[10] for r in rows], dtype=np.float32)
ok = np.hypot(dx, dy) < 15
print(f"--- 按原始分辨率分组（|offset|<15px 视为对准）---")
for tag, msk in (("h0<=512", h0v <= 512), ("h0>512", h0v > 512)):
    if msk.sum():
        print(f"{tag}: n={int(msk.sum())}  within-box={np.mean((np.abs(dx[msk]) <= bw[msk] / 2) & (np.abs(dy[msk]) <= bh[msk] / 2)):.3f}"
              f"  |off|mean={np.hypot(dx[msk], dy[msk]).mean():.1f}")
uniq, cnt = np.unique(np.stack([w0v.astype(int), h0v.astype(int)], 1), axis=0, return_counts=True)
top = np.argsort(-cnt)[:8]
print("原图尺寸分布 top:", [(tuple(uniq[i]), int(cnt[i])) for i in top])
out = Path("/root/rivermind-data/MutilModel_3485988898/YOLO26/logs/box_offset_diag.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "n": len(rows),
    "dx_mean": float(dx.mean()), "dx_std": float(dx.std()),
    "dy_mean": float(dy.mean()), "dy_std": float(dy.std()),
    "abs_mean": float(np.hypot(dx, dy).mean()),
    "within_box_rate": float(np.mean((np.abs(dx) <= bw / 2) & (np.abs(dy) <= bh / 2))),
    "box_w_median": float(np.median(bw)), "box_h_median": float(np.median(bh)),
    "samples": [{"stem": r[0], "cx": r[1], "cy": r[2], "bw": r[3], "bh": r[4],
                 "px": r[5], "py": r[6], "dx": r[7], "dy": r[8]} for r in rows[:50]],
}, indent=2))
print("saved", out)
