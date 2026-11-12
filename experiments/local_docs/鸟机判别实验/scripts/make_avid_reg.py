# -*- coding: utf-8 -*-
"""AVID IR 离线配准（相位相关，±12 px 截断）。

为什么必须先配准
----------------
AVID 的 RGB/IR 只是「同帧配对」，不是像素配准（论文：32.3% 全局位移 >5 px），
而 Mendeley 是像素对齐。SOURCES.md 硬约束第 1 条要求：只有已对齐数据才能进入
6 通道训练。因此 AVID 的 IR 先做逐帧全局平移配准，再与 Mendeley 混训。

口径
----
- 在 320×256 原生尺度上估计全局平移，用 Hann 窗抑制边界；
- 估计量超过 ±12 px 视为失败，记 0 位移（不硬扭）；
- 只做平移，不做旋转/缩放，残差在报告里声明；
- 输出 JPEG q95，文件名与 RGB 扁平名一致，供 bd_nc2 直接引用。

自检：先用合成位移验证 phaseCorrelate 的符号，再跑真实数据。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

AVID = Path("/root/rivermind-fs/AVID")
OUT = Path("/root/rivermind-fs/birddrone/avid_ir_reg")
MAX_SHIFT = 12.0


def _gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取 {p}")
    return img.astype(np.float32) / 255.0


def _hann(shape):
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def estimate_shift(rgb_g: np.ndarray, ir_g: np.ndarray):
    win = _hann(rgb_g.shape)
    (dx, dy), resp = cv2.phaseCorrelate(rgb_g * win, ir_g * win)
    return float(dx), float(dy), float(resp)


def self_test() -> None:
    """合成平移：src2 = shift(src1, (dx,dy))，验证还原方向。"""
    rng = np.random.default_rng(0)
    base = rng.random((256, 320)).astype(np.float32)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    moved = np.roll(base, (4, -6), axis=(0, 1))  # dy=4, dx=-6
    dx, dy, _ = estimate_shift(base, moved)
    print(f"[self-test] 真实位移 (dx=-6, dy=4) -> 估计 ({dx:+.2f}, {dy:+.2f})")
    assert abs(dx + 6) < 1.0 and abs(dy - 4) < 1.0, "符号约定不符，停止"


def warp_ir(ir_img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    return cv2.warpAffine(ir_img, m, (ir_img.shape[1], ir_img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def main() -> None:
    self_test()
    dxs, dys, resps, fails = [], [], [], 0
    for split in ("train", "val"):
        rgb_root = AVID / "images/rgb" / split
        ir_root = AVID / "images/ir" / split
        dst_root = OUT / split
        dst_root.mkdir(parents=True, exist_ok=True)
        n = 0
        for video_dir in sorted(p for p in rgb_root.iterdir() if p.is_dir()):
            video = video_dir.name
            for rgb_p in sorted(video_dir.iterdir()):
                if not rgb_p.is_file():
                    continue
                ir_p = ir_root / video / rgb_p.name
                if not ir_p.exists():
                    continue
                dst = dst_root / f"{video}__{rgb_p.stem}.jpg"
                if dst.exists():
                    n += 1
                    continue
                rgb_g = _gray(rgb_p)
                ir_g0 = _gray(ir_p)
                dx, dy, resp = estimate_shift(rgb_g, ir_g0)
                if abs(dx) > MAX_SHIFT or abs(dy) > MAX_SHIFT or not np.isfinite(dx + dy):
                    fails += 1
                    dx = dy = 0.0
                dxs.append(dx)
                dys.append(dy)
                resps.append(resp)
                ir_img = cv2.imread(str(ir_p), cv2.IMREAD_UNCHANGED)
                ir_reg = warp_ir(ir_img, dx, dy)
                cv2.imwrite(str(dst), ir_reg, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
            print(f"[reg] {split}/{video} -> {n} frames", flush=True)

    dxs = np.asarray(dxs)
    dys = np.asarray(dys)
    mag = np.hypot(dxs, dys)
    stats = {
        "frames": int(len(dxs)),
        "fail_over_12px": int(fails),
        "dx_mean": round(float(dxs.mean()), 3) if len(dxs) else None,
        "dy_mean": round(float(dys.mean()), 3) if len(dys) else None,
        "dx_std": round(float(dxs.std()), 3) if len(dxs) else None,
        "dy_std": round(float(dys.std()), 3) if len(dys) else None,
        "mag_median": round(float(np.median(mag)), 3) if len(mag) else None,
        "mag_p90": round(float(np.percentile(mag, 90)), 3) if len(mag) else None,
        "frac_mag_gt5": round(float((mag > 5).mean()), 4) if len(mag) else None,
        "resp_mean": round(float(np.mean(resps)), 4) if resps else None,
        "max_shift": MAX_SHIFT,
    }
    (OUT / "reg_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
