# -*- coding: utf-8 -*-
"""C1：可视化对比图 —— 找出跨域失效的视觉原因。

生成四张拼版（每张左侧为 2x3 小图网格）：
  1. mendeley_hit.png    Mendeley 里**成功命中**的样本（看模型能认什么）
  2. mendeley_miss.png   Mendeley 里**漏检**的样本（看失败的是什么）
  3. mendeley_fp.png     Mendeley 空帧上的**误报**（看虚警长什么样）
  4. v2_reference.png    v2 域内正常检测样本（对照组）

每格标注：GT 框（绿）、检测框（红，标 conf）、尺寸桶。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
OUT = Path("/root/rivermind-fs/mendeley_eo_ir/_c1")
TILE = (320, 256)


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


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def side(b):
    return float(np.sqrt(max(b[2] - b[0], 1.0) * max(b[3] - b[1], 1.0)))


def tile(img, gts, dets, title):
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for g in gts:
        cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)
    for b, c in dets:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.putText(vis, f"{c:.2f}", (int(b[0]), max(12, int(b[1]) - 4)),
                    0, 0.5, (0, 0, 255), 1)
    # 若目标很小，裁一个以 GT 为中心的放大视图，保证人能看清
    if gts and max(side(g) for g in gts) < 48:
        cx = int(np.mean([(g[0] + g[2]) / 2 for g in gts]))
        cy = int(np.mean([(g[1] + g[3]) / 2 for g in gts]))
        r = 80
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r), min(h, cy + r)
        crop = vis[y0:y1, x0:x1]
        if crop.size:
            vis = crop
    vis = cv2.resize(vis, TILE)
    bar = np.zeros((22, TILE[0], 3), np.uint8)
    cv2.putText(bar, title[:46], (3, 15), 0, 0.45, (255, 255, 255), 1)
    return np.vstack([bar, vis])


def sheet(tiles, cols, path, title):
    if not tiles:
        print(f"  [跳过] {title} 无样本")
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    img = np.vstack(rows)
    head = np.zeros((30, img.shape[1], 3), np.uint8)
    cv2.putText(head, title, (6, 21), 0, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(path), np.vstack([head, img]))
    print(f"  写入 {path}  {img.shape}")


def main():
    proj = Path(os.environ["MM_PROJ"])
    sys.path.insert(0, str(proj))
    from ultralytics import YOLOMM

    W = proj / "ResTest/v2c_E1_rgbonly_s0/weights/best.pt"
    CONF = 0.5
    OUT.mkdir(parents=True, exist_ok=True)

    md = Path("/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/"
              "Labelled_UAV_Images/Labelled EO_IR Dataset/UAV_eo_Dataset")
    v2 = Path("/root/rivermind-data/uav_data_v2")

    def collect(img_dir, lab_dir, n_max):
        files = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        files = files[:n_max]
        model = YOLOMM(str(W))
        it = model.predict(source=[str(p) for p in files], imgsz=640, batch=16, device="0",
                           conf=0.001, iou=0.7, max_det=300, verbose=False, stream=True)
        recs = []
        for res, fp in zip(it, files):
            h, w = res.meta.get("ori_shape", (512, 640))[:2]
            gts = [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                   for cx, cy, bw, bh in read_yolo(lab_dir / f"{fp.stem}.txt")]
            dets = []
            if res.boxes is not None and len(res.boxes):
                arr = np.asarray(res.boxes)
                dets = [(tuple(arr[k, :4]), float(arr[k, 4])) for k in range(len(arr))]
            recs.append({"path": fp, "gts": gts, "dets": dets})
        return recs

    print("[C1] Mendeley ...")
    md_recs = collect(md / "images/test", md / "labels/test", 614)
    print("[C1] v2 ...")
    v2_recs = collect(v2 / "images/test", v2 / "labels/test", 1200)

    def classify(r):
        d = [x for x in r["dets"] if x[1] >= CONF]
        if not r["gts"]:
            return "empty", d
        hits, miss = 0, 0
        used = set()
        for g in r["gts"]:
            ok = False
            for k, (b, c) in enumerate(d):
                if k in used:
                    continue
                if iou(b, g) >= 0.5:
                    used.add(k)
                    ok = True
                    break
            hits += ok
            miss += (not ok)
        return ("allhit" if miss == 0 else "miss"), d

    hits, misses, fps, refs = [], [], [], []
    for r in md_recs:
        kind, d = classify(r)
        s = side(r["gts"][0]) if r["gts"] else 0
        b = ("<16px" if s < 16 else "16-32px" if s < 32 else
             "32-64px" if s < 64 else "64-128px" if s < 128 else ">=128px")
        img = cv2.imread(str(r["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if kind == "allhit" and len(hits) < 8:
            hits.append(tile(img, r["gts"], d, f"HIT {r['path'].stem} {b}"))
        elif kind == "miss" and len(misses) < 12:
            misses.append(tile(img, r["gts"], d, f"MISS {r['path'].stem} GT{b} det{len(d)}"))
        elif kind == "empty" and d and len(fps) < 8:
            fps.append(tile(img, [], d, f"FP {r['path'].stem} n={len(d)}"))
    for r in v2_recs:
        kind, d = classify(r)
        if kind == "allhit" and len(refs) < 8:
            img = cv2.imread(str(r["path"]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                refs.append(tile(img, r["gts"], d, f"v2 {r['path'].stem}"))

    print("\n=== 生成拼版 ===")
    sheet(hits, 4, OUT / "mendeley_hit.png", "Mendeley HIT (conf>=0.5)")
    sheet(misses, 4, OUT / "mendeley_miss.png", "Mendeley MISS (conf>=0.5)")
    sheet(fps, 4, OUT / "mendeley_fp.png", "Mendeley FP on empty frames")
    sheet(refs, 4, OUT / "v2_reference.png", "v2 in-domain reference")


if __name__ == "__main__":
    main()
