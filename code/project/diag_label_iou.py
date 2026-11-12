# -*- coding: utf-8 -*-
"""量化 RGB 与 IR 两套标注在同一帧内的几何不相容程度（纯 CPU，不占显存）。

为什么需要这个数
----------------
交叉口径探针已证明：同一批预测在 RGB 标签上得 0.65、在 IR 标签上得 0.0038
（反之 0.0038 / 0.6084）。这说明两套标注「几何上近乎互斥」。本脚本给出可写进
论文的量化证据：在 both 帧上把 RGB 框与 IR 框做一对一匹配后，统计 IoU 分布。

输出：
  * metadata/val.json 的字段结构（确认 valid_* 判定依据）
  * both / only_rgb / only_ir 帧数
  * both 帧上的框数、匹配对数、IoU 均值/中位数、IoU<0.5 的占比
  * 归一化中心偏移 |dx|、|dy| 的均值与分位数
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path("/root/rivermind-data/uav_data_v2")


def load_boxes(p: Path) -> list[tuple[float, float, float, float]]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        try:
            _, cx, cy, w, h = (float(x) for x in s[:5])
        except ValueError:
            continue
        out.append((cx, cy, w, h))
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 1e-12 else 0.0


def main() -> None:
    meta_p = ROOT / "metadata" / "val.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    frames = meta.get("frames", {})
    print(f"[metadata] {meta_p}  version={meta.get('version')} split={meta.get('split')} 帧数={len(frames)}")
    first = next(iter(frames.items()))
    print(f"[metadata] 示例条目 {first[0]}: {json.dumps(first[1], ensure_ascii=False)[:400]}")

    kinds = {}
    for _stem, info in frames.items():
        k = str(info.get("kind") or info.get("frame_kind") or "?")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"[metadata] kind 分布 = {kinds}")

    def flags(info):
        vr = info.get("valid_rgb")
        vi = info.get("valid_ir")
        if vr is None and vi is None:
            k = str(info.get("kind") or "")
            return k == "both", k == "both"
        return bool(vr), bool(vi)

    both = [(s, i) for s, i in frames.items() if flags(i) == (True, True)]
    only_rgb = [s for s, i in frames.items() if flags(i) == (True, False)]
    only_ir = [s for s, i in frames.items() if flags(i) == (False, True)]
    print(f"[帧] both={len(both)}  only_rgb={len(only_rgb)}  only_ir={len(only_ir)}")

    lab_rgb = ROOT / "labels" / "val"
    lab_ir = ROOT / "labels_ir" / "val"
    ious, dxs, dys, npair, ntot = [], [], [], 0, 0
    no_ir_file = 0
    for stem, _ in both:
        rb = load_boxes(lab_rgb / f"{stem}.txt")
        ib = load_boxes(lab_ir / f"{stem}.txt")
        if not (lab_ir / f"{stem}.txt").exists():
            no_ir_file += 1
        ntot += len(rb)
        if not rb or not ib:
            continue
        # 贪心一对一：按中心距离最近配对
        used = set()
        for a in rb:
            best, bestd = None, 1e9
            for j, b in enumerate(ib):
                if j in used:
                    continue
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d < bestd:
                    best, bestd = j, d
            if best is None:
                continue
            used.add(best)
            b = ib[best]
            ious.append(iou(a, b))
            dxs.append(abs(a[0] - b[0]))
            dys.append(abs(a[1] - b[1]))
            npair += 1

    print(f"[标签] both 帧 RGB 框总数={ntot}  IR 标签文件缺失={no_ir_file}")
    if ious:
        q = lambda v, p: sorted(v)[min(len(v) - 1, int(len(v) * p))]
        print(f"[匹配] 配对数={npair}  覆盖率={npair / max(ntot, 1):.1%}")
        print(f"[IoU ] 均值={st.mean(ious):.4f}  中位={st.median(ious):.4f}  "
              f"p10={q(ious, 0.1):.4f} p90={q(ious, 0.9):.4f}  最大={max(ious):.4f}")
        for thr in (0.1, 0.3, 0.5, 0.75):
            frac = sum(1 for x in ious if x < thr) / len(ious)
            print(f"[IoU ] IoU < {thr:.2f} 占比 = {frac:.1%}")
        print(f"[偏移] |dx| 均值={st.mean(dxs):.4f} 中位={st.median(dxs):.4f}  p90={q(dxs, 0.9):.4f} max={max(dxs):.4f}")
        print(f"[偏移] |dy| 均值={st.mean(dys):.4f} 中位={st.median(dys):.4f}  p90={q(dys, 0.9):.4f} max={max(dys):.4f}")


if __name__ == "__main__":
    main()
