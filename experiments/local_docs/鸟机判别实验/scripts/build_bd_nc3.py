# -*- coding: utf-8 -*-
"""把 other 抠图按轨迹注入 AVID 训练视频，构建 nc=3 数据集 bd_nc3。

- 背景只用 AVID 视频；train/val 背景视频严格来自不同 split（无泄漏）
- 抠图按 8:2 分 train/val，互不重叠
- 只贴天空区域，且与已有 GT 框 IoU<0.1
- 合成帧写 class 2 标签；IR 用灰度复制并标记 fallback
- 输出 bd_nc3/{images,images_ir,labels,labels_ir}/... 与 data_nc3.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RAW = Path("/root/rivermind-fs/birddrone")
BASE = RAW / "bd_nc2"
OUT = RAW / "bd_nc3"
CUTS = RAW / "other_cutouts"
SPLIT = json.loads((RAW / "split.json").read_text())["sources"]["avid"]["splits"]
random.seed(0)
np.random.seed(0)


def sky_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m = (((V >= 140) & (S <= 85)) | ((V >= 150) & (H >= 95) & (H <= 125))).astype(np.uint8) * 255
    h = bgr.shape[0]
    m[int(h * 0.78):] = 0
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def yolo_to_xyxy(lines, w, h):
    out = []
    for ln in lines:
        t = ln.split()
        if len(t) < 5:
            continue
        c, cx, cy, bw, bh = int(float(t[0])), *[float(v) for v in t[1:5]]
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def pick_pos(sky, gt, w, h, cw, ch, tries=80):
    ys, xs = np.where(sky > 0)
    if len(xs) == 0:
        return None
    for _ in range(tries):
        i = random.randrange(len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        x1, y1 = cx - cw // 2, cy - ch // 2
        x2, y2 = x1 + cw, y1 + ch
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        box = (x1, y1, x2, y2)
        if any(iou(box, g[1:]) > 0.1 for g in gt):
            continue
        if sky[y1:y2, x1:x2].mean() < 200:
            continue
        return x1, y1
    return None


def load_cutout(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        return None
    return img


def compose(bgr, cut, x, y, cw, ch, angle=0.0, bright=1.0, blur=0):
    obj = cv2.resize(cut, (cw, ch), interpolation=cv2.INTER_AREA)
    if abs(angle) > 0.5:
        m = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
        obj = cv2.warpAffine(obj, m, (cw, ch), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    rgb = obj[:, :, :3].astype(np.float32) * bright
    a = obj[:, :, 3].astype(np.float32) / 255.0
    if blur > 1:
        k = cv2.getGaussianKernel(blur * 2 + 1, blur * 0.6)
        rgb = cv2.filter2D(rgb, -1, k @ k.T)
        a = cv2.filter2D(a, -1, k @ k.T)
    roi = bgr[y:y + ch, x:x + cw].astype(np.float32)
    roi = roi * (1 - a[..., None]) + rgb * a[..., None]
    bgr[y:y + ch, x:x + cw] = np.clip(roi, 0, 255).astype(np.uint8)
    return bgr


def build_clips(split, clips, frames_per_clip, cut_list, fname_key, n_objs=(1, 2)):
    img_dir = BASE / "images" / split
    lb_dir = BASE / "labels" / split
    vids = [v["video"] for v in SPLIT[split]]
    manifest = []
    made = 0
    for clip in range(clips):
        video = random.choice(vids)
        files = sorted(img_dir.glob(f"avid__{video}__*.jpg"))
        if len(files) < frames_per_clip + 1:
            continue
        start = random.randrange(0, len(files) - frames_per_clip)
        seq = files[start:start + frames_per_clip]
        bgr0 = cv2.imread(str(seq[0]))
        if bgr0 is None:
            continue
        h, w = bgr0.shape[:2]
        if (sky_mask(bgr0) > 0).mean() < 0.06:
            continue
        objs = []
        for _ in range(random.randint(*n_objs)):
            cut = random.choice(cut_list)
            im = load_cutout(CUTS / cut["file"])
            if im is None:
                continue
            side = int(round(np.exp(random.uniform(np.log(10), np.log(42)))))
            ar = im.shape[1] / im.shape[0]
            cw = max(8, int(side * (ar ** 0.5)))
            ch = max(8, int(side / (ar ** 0.5)))
            spd = random.uniform(0.2, 0.6) if cut["class"] == "balloon" else random.uniform(0.6, 2.0)
            objs.append({"cut": cut, "arr": im, "cw": cw, "ch": ch,
                         "vx": random.choice([-1, 1]) * spd,
                         "vy": random.uniform(-0.25, 0.25) * spd,
                         "phase": random.uniform(0, 6.28)})
        if not objs:
            continue
        # 初始位置（用首帧）
        gt0 = yolo_to_xyxy(lb_dir.joinpath(seq[0].stem + ".txt").read_text().splitlines(), w, h)
        sky0 = sky_mask(bgr0)
        kept = []
        for o in objs:
            p = pick_pos(sky0, gt0, w, h, o["cw"], o["ch"])
            if p is not None:
                o["x"], o["y"] = p
                kept.append(o)
        objs = kept
        if not objs:
            continue
        n_made = 0
        for t, fp in enumerate(seq):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            gt = yolo_to_xyxy(lb_dir.joinpath(fp.stem + ".txt").read_text().splitlines(), w, h)
            added = []
            for o in objs:
                x = int(round(o["x"] + o["vx"] * t))
                y = int(round(o["y"] + o["vy"] * t + 1.5 * np.sin(t * 0.5 + o["phase"])))
                if x < 0 or y < 0 or x + o["cw"] > w or y + o["ch"] > h:
                    continue
                if any(iou((x, y, x + o["cw"], y + o["ch"]), g[1:]) > 0.1 for g in gt):
                    continue
                if sky_mask(bgr)[y:y + o["ch"], x:x + o["cw"]].mean() < 180:
                    continue
                spd = (o["vx"] ** 2 + o["vy"] ** 2) ** 0.5
                bgr = compose(bgr, o["arr"], x, y, o["cw"], o["ch"],
                              angle=random.uniform(-6, 6),
                              bright=random.uniform(0.9, 1.1),
                              blur=int(min(3, spd)))
                added.append((x, y, o["cw"], o["ch"]))
            if not added:
                continue
            stem = f"{fp.stem}__syn{clip:03d}"
            cls2 = []
            for (x, y, cw, ch) in added:
                cls2.append(f"2 {(x + cw / 2) / w:.6f} {(y + ch / 2) / h:.6f} "
                            f"{cw / w:.6f} {ch / h:.6f}")
            base_lines = lb_dir.joinpath(fp.stem + ".txt").read_text().splitlines()
            (OUT / "labels" / split / f"{stem}.txt").write_text(
                "\n".join([ln for ln in base_lines if ln.strip()] + cls2) + "\n", encoding="utf-8")
            (OUT / "labels_ir" / split / f"{stem}.txt").write_text(
                "\n".join([ln for ln in base_lines if ln.strip()] + cls2) + "\n", encoding="utf-8")
            cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(OUT / "images_ir" / split / f"{stem}.jpg"),
                        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append({"split": split, "clip": clip, "frame": fp.name, "stem": stem,
                             "objects": [o["cut"]["class"] for o in objs],
                             "n_injected": len(added), "synthetic": True})
            n_made += 1
        made += n_made
        if clip % 25 == 0:
            print(f"[{split}] clip {clip}/{clips} frames {made}", flush=True)
    return manifest, made


def symlink_base(split, names):
    for sub in ("images", "images_ir", "labels", "labels_ir"):
        (OUT / sub / split).mkdir(parents=True, exist_ok=True)
    # 预建索引，避免每个样本都 glob 整个目录（O(n^2)）
    ir_map = {p.stem: p for p in (BASE / "images_ir" / split).iterdir()}
    lb_map = {p.stem: p for p in (BASE / "labels" / split).iterdir()}
    lbi_map = {p.stem: p for p in (BASE / "labels_ir" / split).iterdir()}
    for name in names:
        rel = Path(name)                      # 形如 images/train/men__xxx.png
        stem = rel.stem
        src = BASE / rel
        dst = OUT / rel
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
        ir = ir_map.get(stem)
        if ir is not None:
            d = OUT / "images_ir" / split / ir.name
            if not d.exists():
                os.symlink(ir, d)
        for m, sub in ((lb_map, "labels"), (lbi_map, "labels_ir")):
            s = m.get(stem)
            if s is not None:
                d = OUT / sub / split / s.name
                if not d.exists():
                    os.symlink(s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-clips", type=int, default=280)
    ap.add_argument("--val-clips", type=int, default=20)
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args()

    idx = json.loads((CUTS / "index.json").read_text())
    random.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.2))
    val_cuts, train_cuts = idx[:n_val], idx[n_val:]
    print("cutouts train/val", len(train_cuts), len(val_cuts))

    for split in ("train", "val"):
        for sub in ("images", "images_ir", "labels", "labels_ir"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)

    base_train = [x.strip() for x in (BASE / "train_3to1.txt").read_text().splitlines() if x.strip()]
    base_val = [x.strip() for x in (BASE / "val_bal.txt").read_text().splitlines() if x.strip()]
    symlink_base("train", base_train)
    symlink_base("val", base_val)

    m1, n1 = build_clips("train", a.train_clips, a.frames, train_cuts, "train")
    m2, n2 = build_clips("val", a.val_clips, a.frames, val_cuts, "val")
    (OUT / "synth_manifest.json").write_text(
        json.dumps(m1 + m2, ensure_ascii=False), encoding="utf-8")

    syn_train = sorted((OUT / "images/train").glob("*__syn*.jpg"))
    syn_val = sorted((OUT / "images/val").glob("*__syn*.jpg"))
    (OUT / "train_nc3.txt").write_text(
        "\n".join(base_train + [f"images/train/{p.name}" for p in syn_train]) + "\n",
        encoding="utf-8")
    (OUT / "val_nc3.txt").write_text(
        "\n".join(base_val + [f"images/val/{p.name}" for p in syn_val]) + "\n",
        encoding="utf-8")

    y = f"""path: {OUT}
train: train_nc3.txt
val: val_nc3.txt
test: images/test
modality:
  rgb: images
  ir: images_ir
modality_used:
- rgb
- ir
labels:
  rgb: labels
  ir: labels_ir
pair_policy: any
align: false
channels: 3
Xch: 3
nc: 3
names:
  0: bird
  1: drone
  2: other
"""
    (OUT / "data_nc3.yaml").write_text(y, encoding="utf-8")
    print("base train/val", len(base_train), len(base_val),
          "synth", n1, n2, "total", len(base_train) + n1)


if __name__ == "__main__":
    main()
