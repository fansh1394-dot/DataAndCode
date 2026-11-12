# -*- coding: utf-8 -*-
"""配准导出物内容级验收（一次性诊断，非正式产物）。

检查三件事：
1. data.yaml / 软链 / 标签文件数是否与 metadata 一致；
2. warp 后 IR 热斑是否落在 **RGB 标签框** 内（原图对照应落在 IR 标签框内）；
3. only_ir 帧的 labels/ 是否写入了非空框（否则该帧在 RGB 坐标系下变成背景）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/rivermind-data/uav_data_v2")
REG = Path("/root/rivermind-data/uav_data_v2_reg")
N_SAMPLE = 40


def read_boxes(path: Path):
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.split()
        if len(s) >= 5:
            out.append(tuple(float(x) for x in s[1:5]))
    return out


def peak_in_box(img, boxes, h_img, w_img):
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 6.0)
    cy, cx = np.unravel_index(int(np.argmax(g)), g.shape)
    px, py = cx / w_img, cy / h_img
    for b in boxes:
        if abs(px - b[0]) <= b[2] / 2 and abs(py - b[1]) <= b[3] / 2:
            return True, (round(px, 3), round(py, 3))
    return False, (round(px, 3), round(py, 3))


def main() -> int:
    print("=== 1. 结构检查 ===")
    print("  images  ->", REG.joinpath("images").resolve())
    print("  metadata->", REG.joinpath("metadata").resolve())
    print("  data.yaml:")
    print((REG / "data.yaml").read_text(encoding="utf-8"))

    meta = json.loads((REG / "_reg_meta.json").read_text(encoding="utf-8"))
    print(f"  mode={meta['mode']} oracle={meta['oracle']} "
          f"ok_frac={meta['acceptance']['frame_ok_frac_2px']*100:.2f}%")
    print(f"  scale_ratio w={meta['summary']['scale_ratio_w_mean']:.3f} "
          f"h={meta['summary']['scale_ratio_h_mean']:.3f}")

    n_img = sum(1 for p in (REG / "images_ir" / "test").iterdir() if p.is_file())
    n_lab = sum(1 for p in (REG / "labels" / "test").iterdir() if p.is_file())
    print(f"  test: images_ir={n_img} labels={n_lab}")
    if n_img != n_lab:
        print("  [失败] 图像与标签数不一致")
        return 2

    print("\n=== 2. 热斑位置检验（warp 后应落 RGB 框，原图应落 IR 框）===")
    frames = json.loads((REG / "metadata" / "test.json").read_text(encoding="utf-8"))["frames"]
    both = [s for s, v in frames.items() if v.get("kind") == "both"][:N_SAMPLE]
    ok_new, ok_old, n = 0, 0, 0
    for stem in both:
        rgb_boxes = read_boxes(REG / "labels" / "test" / f"{stem}.txt")
        ir_boxes = read_boxes(ROOT / "labels_ir" / "test" / f"{stem}.txt")
        if not rgb_boxes or not ir_boxes:
            continue
        new = cv2.imread(str(REG / "images_ir" / "test" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        old = cv2.imread(str(ROOT / "images_ir" / "test" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        if new is None or old is None:
            continue
        h, w = new.shape[:2]
        hit_new, p_new = peak_in_box(new, rgb_boxes, h, w)
        hit_old, p_old = peak_in_box(old, rgb_boxes, h, w)
        ok_new += hit_new
        ok_old += hit_old
        n += 1
        if n <= 5:
            print(f"  {stem}: warp后峰值{p_new} RGB框内={hit_new} | 原图峰值{p_old} RGB框内={hit_old}")
    print(f"  汇总 n={n}  warp 后落 RGB 框 {ok_new}/{n} = {ok_new/max(n,1)*100:.1f}%"
          f" | 原图落 RGB 框 {ok_old}/{n} = {ok_old/max(n,1)*100:.1f}%")

    print("\n=== 3. only_ir 帧标签检查 ===")
    only_ir = [s for s, v in frames.items() if v.get("kind") == "only_ir"][:200]
    nonempty = sum(1 for s in only_ir if read_boxes(REG / "labels" / "test" / f"{s}.txt"))
    print(f"  test only_ir 抽样 {len(only_ir)} 帧，labels/ 非空 {nonempty} 帧"
          f" = {nonempty/max(len(only_ir),1)*100:.1f}%")

    fail = []
    if ok_new / max(n, 1) < 0.85:
        fail.append("warp 后热斑未落到 RGB 框")
    if ok_old / max(n, 1) > 0.3:
        fail.append("原图热斑本就落 RGB 框（对齐前提不成立）")
    if nonempty / max(len(only_ir), 1) < 0.5:
        fail.append("only_ir 帧标签丢失")
    print("\n[结论]", "通过" if not fail else "失败：" + "；".join(fail))
    return 0 if not fail else 3


if __name__ == "__main__":
    sys.exit(main())
