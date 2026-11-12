# -*- coding: utf-8 -*-
"""生成 other 类抠图（RGBA）：气球用多边形掩膜，风筝/飞机用 GrabCut。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

RAW = Path("/root/rivermind-fs/birddrone/other_raw")
OUT = Path("/root/rivermind-fs/birddrone/other_cutouts")
OUT.mkdir(parents=True, exist_ok=True)


def feather(alpha: np.ndarray, r: float = 0.8) -> np.ndarray:
    return cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), r)


def balloon_cutouts():
    """按 (width,height) 把 JSON 标注重定位到 party_*.jpg。"""
    records = []
    for split in ("train", "val"):
        j = json.loads((RAW / "balloon" / f"custom_{split}.json").read_text())
        files = sorted((RAW / "balloon").glob(f"party_{split}_*.jpg"))
        sizes = {}
        for i, p in enumerate(files):
            with Image.open(p) as im:
                sizes.setdefault(im.size, []).append(i)
        anns_by_img = {}
        for a in j["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        used = set()
        for im in j["images"]:
            cands = [i for i in sizes.get((im["width"], im["height"]), []) if i not in used]
            if not cands:
                continue
            anns = anns_by_img.get(im["id"], [])
            pick = None
            for i in cands:
                ok = all(
                    0 <= a["bbox"][0] and 0 <= a["bbox"][1]
                    and a["bbox"][0] + a["bbox"][2] <= im["width"] + 1
                    and a["bbox"][1] + a["bbox"][3] <= im["height"] + 1
                    for a in anns)
                if ok:
                    pick = i
                    break
            if pick is None:
                pick = cands[0]
            used.add(pick)
            src = files[pick]
            with Image.open(src) as im_obj:
                rgb = np.array(im_obj.convert("RGB"))
            for a in anns:
                x, y, w, h = [int(round(v)) for v in a["bbox"]]
                x, y = max(x, 0), max(y, 0)
                w, h = min(w, rgb.shape[1] - x), min(h, rgb.shape[0] - y)
                if w < 8 or h < 8:
                    continue
                mask = Image.new("L", (rgb.shape[1], rgb.shape[0]), 0)
                dr = ImageDraw.Draw(mask)
                seg = a.get("segmentation") or []
                if seg and isinstance(seg[0], list):
                    for poly in seg:
                        pts = [(poly[k], poly[k + 1]) for k in range(0, len(poly) - 1, 2)]
                        if len(pts) >= 3:
                            dr.polygon(pts, fill=255)
                else:
                    dr.rectangle([x, y, x + w, y + h], fill=255)
                m = np.array(mask)[y:y + h, x:x + w]
                if m.max() == 0:
                    continue
                crop = rgb[y:y + h, x:x + w]
                rgba = np.dstack([crop, m])
                name = f"balloon_{split}_{im['id']:03d}_{a['id']:03d}.png"
                Image.fromarray(rgba, "RGBA").save(OUT / name)
                records.append({"file": name, "class": "balloon", "source": src.name,
                                "w": w, "h": h})
    return records


def box_cutouts(cls: str, cid: int):
    records = []
    d = RAW / "coco" / cls
    for img_p in sorted(d.glob("*.jpg")):
        lb = img_p.with_suffix(".txt")
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        for k, ln in enumerate(lb.read_text().splitlines()):
            t = ln.split()
            if len(t) < 5 or int(float(t[0])) != cid:
                continue
            cx, cy, bw, bh = [float(v) for v in t[1:5]]
            x1 = int(max((cx - bw / 2) * w - 2, 0))
            y1 = int(max((cy - bh / 2) * h - 2, 0))
            x2 = int(min((cx + bw / 2) * w + 2, w))
            y2 = int(min((cy + bh / 2) * h + 2, h))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            roi = bgr[y1:y2, x1:x2]
            mask = np.zeros(roi.shape[:2], np.uint8)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            rect = (2, 2, roi.shape[1] - 4, roi.shape[0] - 4)
            try:
                cv2.grabCut(roi, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
                fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            except Exception:
                fg = np.zeros(roi.shape[:2], np.uint8)
            frac = float((fg > 0).mean())
            if frac < 0.05 or frac > 0.97:
                fg = np.zeros(roi.shape[:2], np.uint8)
                cv2.ellipse(fg, (roi.shape[1] // 2, roi.shape[0] // 2),
                            (max(roi.shape[1] // 2 - 1, 2), max(roi.shape[0] // 2 - 1, 2)),
                            0, 0, 360, 255, -1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # 填洞 + alpha 归一化：GrabCut 常给出 PR_FGD(128)，不归一化会显得半透明
            cnts, _ = cv2.findContours((fg > 127).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fg = np.zeros_like(fg)
            if cnts:
                cv2.drawContours(fg, cnts, -1, 255, -1)
            if fg.max() == 0 or (fg > 0).mean() < 0.04 or (fg > 0).mean() > 0.98:
                continue
            fg = feather(fg)
            rgba = np.dstack([cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), fg.astype(np.uint8)])
            name = f"{cls}_{img_p.stem}_{k:02d}.png"
            Image.fromarray(rgba, "RGBA").save(OUT / name)
            records.append({"file": name, "class": cls, "source": img_p.name,
                            "w": int(x2 - x1), "h": int(y2 - y1)})
    return records


def main():
    recs = balloon_cutouts() + box_cutouts("kite", 33) + box_cutouts("airplane", 4)
    (OUT / "index.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    from collections import Counter
    print("cutouts", Counter(r["class"] for r in recs))
    print("total", len(recs))


if __name__ == "__main__":
    main()
