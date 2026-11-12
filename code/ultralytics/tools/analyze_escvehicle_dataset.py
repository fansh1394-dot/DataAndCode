#!/usr/bin/env python
"""ESCVehicle_yolomm 数据集全自动体检脚本。

纯只读、全自动统计，输出 JSON + Markdown + 投稿级 PNG/SVG 图。
不训练、不修改任何模型代码、不改动数据集。

用法:
    /root/miniconda3/envs/yolon/bin/python ultralytics/tools/analyze_escvehicle_dataset.py \
        --root /root/datasets/ESCVehicle_yolomm \
        --out ResDatasetCheck
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 项目复用
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]  # ultralytics/tools → ultralyticsmm
sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics.utils.ops import xyxyxyxy2xywhr  # noqa: E402
from ultralytics.tools.convert_escvehicle_yolomm import (  # noqa: E402
    CLASS_NAMES,
    _validate_label_file,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SPLITS = ("train", "val", "test")
IMAGE_SIZE = 704  # 数据集所有图像统一 704x704
NUM_CLASSES = len(CLASS_NAMES)

# 已知非异常空标签（infrared 模态在 train 有 2 张空标签）
KNOWN_EMPTY_LABELS = {
    "label-ir/train/434.txt",
    "label-ir/train/435.txt",
}

# COCO 尺度分桶 (像素²) — 复用 coco_eval_bbox_mm.py:18 口径
SCALE_BUCKETS = {
    "small": (0.0, 32.0 * 32.0),           # ≤ 1024
    "medium": (32.0 * 32.0, 96.0 * 96.0),  # (1024, 9216]
    "large": (96.0 * 96.0, float("inf")),   # > 9216
}

# ---------------------------------------------------------------------------
# nature-figure 图表规范
# ---------------------------------------------------------------------------
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral_light": "#CFCECE", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "neutral_black": "#272727",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [
    PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
    PALETTE["teal"], PALETTE["violet"], PALETTE["neutral_light"],
]


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def save_pub_figure(fig: plt.Figure, path_no_ext: str, dpi: int = 300) -> list[str]:
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    saved = []
    for ext in ("svg", "png"):
        p = f"{path_no_ext}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def compute_gini(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() - (n + 1) * arr.sum()) / (n * arr.sum()))


def quantiles(arr: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict:
    if arr.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def to_python(obj):
    """递归把 numpy 类型转 Python 类型，便于 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# 数据加载（共享，供 M1/M2/M3/M4/M5/M6 复用）
# ---------------------------------------------------------------------------
def load_all_labels(root: Path) -> dict:
    """
    加载所有标签到内存。

    Returns
    -------
    data[modal][split] = list of (stem, boxes_ndarray)
        boxes_ndarray: shape (N, 9), 列 = [cls, x1,y1,...,x4,y4]，归一化 [0,1]
    """
    data = {}
    for modal, modal_key in (("labels", "rgb"), ("label-ir", "ir")):
        data[modal_key] = {}
        for split in SPLITS:
            d = root / modal / split
            split_data = []
            for txt in sorted(d.glob("*.txt")):
                if txt.stat().st_size == 0:
                    split_data.append((txt.stem, np.empty((0, 9), dtype=np.float64)))
                    continue
                lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
                if not lines:
                    split_data.append((txt.stem, np.empty((0, 9), dtype=np.float64)))
                    continue
                arr = np.array([[float(x) for x in l.split()] for l in lines], dtype=np.float64)
                split_data.append((txt.stem, arr))
            data[modal_key][split] = split_data
    return data


def iter_all_boxes(data: dict, modalities=("rgb", "ir")):
    """遍历所有框，yield (modal, split, stem, boxes_arr)。"""
    for modal in modalities:
        for split in SPLITS:
            for stem, arr in data[modal][split]:
                yield modal, split, stem, arr


# ---------------------------------------------------------------------------
# M0: 完整性与健康性
# ---------------------------------------------------------------------------
def check_integrity(root: Path) -> dict:
    print("  [M0] 完整性检查... ", end="", flush=True)
    t0 = time.time()
    result = {
        "directory_stem_consistency": {},
        "empty_labels": [],
        "known_empty_labels": list(KNOWN_EMPTY_LABELS),
        "format_violations": [],
        "cache_files": {},
        "root_anomalies": [],
        "cross_validation": {},
        "checks": [],  # 红绿灯清单 [(status, message)]
    }

    # 四目录 stem 一致性
    for split in SPLITS:
        stem_sets = {}
        for subdir in ("images", "images_ir", "labels", "label-ir"):
            d = root / subdir / split
            suffix = ".jpg" if "image" in subdir else ".txt"
            if not d.is_dir():
                result["checks"].append(("FAIL", f"缺少目录: {d}"))
                continue
            stem_sets[subdir] = {p.stem for p in d.glob(f"*{suffix}")}
        ref = stem_sets.get("images", set())
        for name, stems in stem_sets.items():
            if name == "images":
                continue
            if stems != ref:
                diff = stems.symmetric_difference(ref)
                result["directory_stem_consistency"][f"{split}/{name}"] = {
                    "consistent": False, "diff_count": len(diff),
                    "examples": sorted(diff)[:5],
                }
                result["checks"].append(("FAIL", f"{split}/{name} stem 不一致 ({len(diff)} 个)"))
            else:
                result["directory_stem_consistency"][f"{split}/{name}"] = {"consistent": True}

    # 空标签 + 格式校验
    empty_labels = []
    violations = []
    total_checked = 0
    for modal in ("labels", "label-ir"):
        for split in SPLITS:
            d = root / modal / split
            for txt in sorted(d.glob("*.txt")):
                rel = f"{modal}/{split}/{txt.name}"
                if txt.stat().st_size == 0:
                    empty_labels.append(rel)
                    is_known = rel in KNOWN_EMPTY_LABELS
                    result["checks"].append((
                        "INFO" if is_known else "WARN",
                        f"空标签: {rel}" + (" (已知非异常)" if is_known else ""),
                    ))
                    continue
                try:
                    _validate_label_file(txt)
                    total_checked += 1
                except Exception as exc:
                    violations.append({"file": rel, "error": str(exc)})
                    result["checks"].append(("FAIL", f"标签格式违规: {rel}: {exc}"))

    result["empty_labels"] = empty_labels
    result["format_violations"] = violations
    result["checks"].append(("PASS", f"成功校验 {total_checked} 个非空标签文件"))

    # cache 文件
    for modal in ("labels", "label-ir"):
        caches = list((root / modal).glob("*.cache"))
        result["cache_files"][modal] = [str(p.relative_to(root)) for p in caches]

    # 根目录异物
    expected_entries = {
        "images", "images_ir", "labels", "label-ir",
        "escvehicle_yolomm.yaml", "conversion_report.json",
    }
    actual_entries = {p.name for p in root.iterdir()}
    extra = actual_entries - expected_entries
    if extra:
        result["root_anomalies"] = sorted(extra)
        result["checks"].append(("WARN", f"根目录异物: {sorted(extra)}"))

    # 交叉验证 conversion_report.json
    report_path = root / "conversion_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            conv = json.load(f)
        cross = {}
        for modal, key in (("labels", "visible"), ("label-ir", "infrared")):
            actual_counts = Counter()
            actual_total = 0
            for split in SPLITS:
                for txt in (root / modal / split).glob("*.txt"):
                    if txt.stat().st_size == 0:
                        continue
                    for line in txt.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        cls_id = int(line.split()[0])
                        if 0 <= cls_id < NUM_CLASSES:
                            actual_counts[CLASS_NAMES[cls_id]] += 1
                            actual_total += 1
            exp_counts = conv["labels"][key]["class_counts"]
            exp_total = conv["labels"][key]["written_objects"]
            class_diffs = {}
            for cls in CLASS_NAMES:
                act = actual_counts.get(cls, 0)
                exp = exp_counts.get(cls, 0)
                if act != exp:
                    class_diffs[cls] = {"expected": exp, "actual": act}
            cross[key] = {
                "class_diffs": class_diffs,
                "total_expected": exp_total,
                "total_actual": actual_total,
                "total_ok": exp_total == actual_total,
            }
        result["cross_validation"] = cross
        all_ok = all(v["total_ok"] and not v["class_diffs"] for v in cross.values())
        result["checks"].append(("PASS" if all_ok else "FAIL",
                                  "与 conversion_report.json 交叉验证: " + ("一致" if all_ok else "存在差异")))
    else:
        result["checks"].append(("WARN", "conversion_report.json 不存在，跳过交叉验证"))

    dt = time.time() - t0
    result["elapsed_seconds"] = round(dt, 3)
    print(f"完成 ({dt:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# M1: 类别分布与长尾
# ---------------------------------------------------------------------------
def analyze_class_distribution(data: dict) -> dict:
    print("  [M1] 类别分布与长尾... ", end="", flush=True)
    t0 = time.time()
    result = {}

    for modal in ("rgb", "ir"):
        modal_result = {"per_split": {}}
        for split in SPLITS:
            counts = Counter()
            for stem, arr in data[modal][split]:
                if arr.size == 0:
                    continue
                for cls_id in arr[:, 0].astype(int):
                    counts[CLASS_NAMES[cls_id]] += 1

            total = sum(counts.values())
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
            cumulative = 0.0
            top_k = []
            for cls, cnt in sorted_counts:
                cumulative += cnt
                top_k.append({
                    "class": cls, "count": int(cnt),
                    "percent": round(cnt / total * 100, 3) if total else 0,
                    "cumulative_percent": round(cumulative / total * 100, 3) if total else 0,
                })
            max_cnt = sorted_counts[0][1] if sorted_counts else 0
            min_cnt = sorted_counts[-1][1] if sorted_counts else 0

            modal_result["per_split"][split] = {
                "total": int(total),
                "per_class": {k: int(v) for k, v in counts.items()},
                "top_k": top_k,
                "imbalance_ratio": round(max_cnt / min_cnt, 2) if min_cnt > 0 else None,
                "gini": round(compute_gini([c for _, c in sorted_counts]), 4),
            }

        # 总汇
        total_counts = Counter()
        for sp in modal_result["per_split"].values():
            total_counts.update(sp["per_class"])
        total = sum(total_counts.values())
        sorted_counts = sorted(total_counts.items(), key=lambda x: -x[1])
        max_cnt = sorted_counts[0][1] if sorted_counts else 0
        min_cnt = sorted_counts[-1][1] if sorted_counts else 0
        modal_result["total"] = {
            "total": int(total),
            "per_class": {k: int(v) for k, v in total_counts.items()},
            "imbalance_ratio": round(max_cnt / min_cnt, 2) if min_cnt > 0 else None,
            "gini": round(compute_gini([c for _, c in sorted_counts]), 4),
        }
        result[modal] = modal_result

    dt = time.time() - t0
    print(f"完成 ({dt:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# 共享几何计算（M2/M3/M5 复用）
# ---------------------------------------------------------------------------
def compute_all_xywhr(data: dict, modalities=("rgb", "ir")):
    """
    收集所有框的几何信息。

    Returns
    -------
    dict with keys:
        cls_ids: np.ndarray (N,) int
        corners_px: np.ndarray (N, 8) float32, 像素角点
        xywhr: np.ndarray (N, 5) [cx, cy, w, h, theta_rad]
        modal: np.ndarray (N,) str (每个框所属 modal)
    """
    all_cls = []
    all_corners = []
    all_modal = []
    for modal, split, stem, arr in iter_all_boxes(data, modalities):
        if arr.size == 0:
            continue
        n = len(arr)
        all_cls.append(arr[:, 0].astype(np.int32))
        all_corners.append((arr[:, 1:] * IMAGE_SIZE).astype(np.float32))
        all_modal.append(np.full(n, modal, dtype=object))

    if not all_cls:
        empty = {
            "cls_ids": np.empty(0, dtype=np.int32),
            "corners_px": np.empty((0, 8), dtype=np.float32),
            "xywhr": np.empty((0, 5), dtype=np.float32),
            "modal": np.empty(0, dtype=object),
        }
        return empty

    cls_ids = np.concatenate(all_cls)
    corners_px = np.concatenate(all_corners)
    modal_arr = np.concatenate(all_modal)
    xywhr = xyxyxyxy2xywhr(corners_px).astype(np.float32)  # (N, 5)

    return {
        "cls_ids": cls_ids,
        "corners_px": corners_px,
        "xywhr": xywhr,
        "modal": modal_arr,
    }


# ---------------------------------------------------------------------------
# M2: OBB 几何
# ---------------------------------------------------------------------------
def analyze_geometry(geo: dict) -> dict:
    print("  [M2] OBB 几何... ", end="", flush=True)
    t0 = time.time()

    cls_ids = geo["cls_ids"]
    xywhr = geo["xywhr"]
    modals = geo["modal"]

    if cls_ids.size == 0:
        return {"per_class": {}, "overall": {}, "per_modal": {}}

    w = xywhr[:, 2]; h = xywhr[:, 3]
    ar = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1e-6)
    area = w * h
    theta_deg = xywhr[:, 4] * 180.0 / np.pi
    long_side = np.maximum(w, h)
    short_side = np.minimum(w, h)

    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = cls_ids == cls_id
        if not mask.any():
            continue
        per_class[cls_name] = {
            "count": int(mask.sum()),
            "aspect_ratio": {
                **quantiles(ar[mask]),
                "mean": float(ar[mask].mean()),
                "near_square_ratio": round(float((ar[mask] < 1.2).mean()), 4),
                "elongated_ratio": round(float((ar[mask] > 3.0).mean()), 4),
            },
            "angle_deg": {
                "mean": float(theta_deg[mask].mean()),
                "median": float(np.median(theta_deg[mask])),
                "std": float(theta_deg[mask].std()),
            },
            "size_px": {
                "long_side": {"mean": float(long_side[mask].mean()), "median": float(np.median(long_side[mask]))},
                "short_side": {"mean": float(short_side[mask].mean()), "median": float(np.median(short_side[mask]))},
            },
            "area_px2": {
                "mean": float(area[mask].mean()),
                "median": float(np.median(area[mask])),
                "min": float(area[mask].min()),
                "max": float(area[mask].max()),
            },
        }

    overall = {
        "count": int(cls_ids.size),
        "aspect_ratio": {**quantiles(ar), "mean": float(ar.mean())},
        "angle_deg": {"mean": float(theta_deg.mean()), "median": float(np.median(theta_deg))},
        "area_px2": {"mean": float(area.mean()), "median": float(np.median(area))},
    }

    per_modal = {}
    for modal in ("rgb", "ir"):
        m = modals == modal
        if not m.any():
            continue
        per_modal[modal] = {
            "count": int(m.sum()),
            "aspect_ratio": {"mean": float(ar[m].mean()), "median": float(np.median(ar[m]))},
            "angle_deg": {"mean": float(theta_deg[m].mean()), "median": float(np.median(theta_deg[m]))},
        }

    dt = time.time() - t0
    print(f"完成 ({dt:.1f}s, {cls_ids.size} 框)")
    return {"per_class": per_class, "overall": overall, "per_modal": per_modal, "elapsed_seconds": round(dt, 3)}


# ---------------------------------------------------------------------------
# M3: 尺度分布（COCO 口径）
# ---------------------------------------------------------------------------
def analyze_scale(geo: dict) -> dict:
    print("  [M3] 尺度分布... ", end="", flush=True)
    t0 = time.time()

    cls_ids = geo["cls_ids"]
    xywhr = geo["xywhr"]
    modals = geo["modal"]

    if cls_ids.size == 0:
        return {"per_class": {}, "overall": {}, "per_modal": {}}

    w = xywhr[:, 2]; h = xywhr[:, 3]
    area = w * h

    def bucket_stats(area_arr):
        stats = {}
        for label, (lo, hi) in SCALE_BUCKETS.items():
            mask = (area_arr > lo) & (area_arr <= hi)
            stats[label] = {"count": int(mask.sum()), "ratio": round(float(mask.mean()), 4)}
        # 极小目标
        tiny = area_arr < 64.0
        stats["tiny_lt_8x8"] = {"count": int(tiny.sum()), "ratio": round(float(tiny.mean()), 4)}
        return stats

    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = cls_ids == cls_id
        if not mask.any():
            continue
        per_class[cls_name] = bucket_stats(area[mask])

    overall = bucket_stats(area)

    per_modal = {}
    for modal in ("rgb", "ir"):
        m = modals == modal
        if not m.any():
            continue
        per_modal[modal] = bucket_stats(area[m])

    dt = time.time() - t0
    print(f"完成 ({dt:.1f}s)")
    return {"per_class": per_class, "overall": overall, "per_modal": per_modal, "elapsed_seconds": round(dt, 3)}


# ---------------------------------------------------------------------------
# M4: 每图密集度 + IoU 重叠
# ---------------------------------------------------------------------------
def analyze_density(data: dict) -> dict:
    print("  [M4] 每图密集度... ", end="", flush=True)
    t0 = time.time()
    result = {}

    for modal in ("rgb", "ir"):
        counts_per_image = []
        max_iou_per_image = []

        for split in SPLITS:
            for stem, arr in data[modal][split]:
                n = len(arr)
                counts_per_image.append(n)
                if n < 2:
                    continue
                # 水平外接框 IoU
                corners = (arr[:, 1:] * IMAGE_SIZE).astype(np.float32)
                pts = corners.reshape(n, 4, 2)
                x1 = pts[:, :, 0].min(axis=1)
                y1 = pts[:, :, 1].min(axis=1)
                x2 = pts[:, :, 0].max(axis=1)
                y2 = pts[:, :, 1].max(axis=1)
                areas = (x2 - x1) * (y2 - y1)
                xx1 = np.maximum(x1[:, None], x1[None, :])
                yy1 = np.maximum(y1[:, None], y1[None, :])
                xx2 = np.minimum(x2[:, None], x2[None, :])
                yy2 = np.minimum(y2[:, None], y2[None, :])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                union = areas[:, None] + areas[None, :] - inter
                iou = inter / (union + 1e-10)
                np.fill_diagonal(iou, 0.0)
                # 对大图（n>100），只采样避免 O(n²) 内存过大
                if n > 100:
                    idx = np.random.choice(n, 100, replace=False)
                    iou_sub = iou[np.ix_(idx, idx)]
                    max_iou_per_image.append(float(iou_sub.max()))
                else:
                    max_iou_per_image.append(float(iou.max()))

        counts_arr = np.array(counts_per_image, dtype=np.int64)
        iou_arr = np.array(max_iou_per_image, dtype=np.float32) if max_iou_per_image else np.empty(0)

        result[modal] = {
            "per_image_count": {
                "total_images": int(len(counts_arr)),
                "total_boxes": int(counts_arr.sum()),
                "min": int(counts_arr.min()),
                "mean": round(float(counts_arr.mean()), 2),
                "median": float(np.median(counts_arr)),
                "p90": float(np.percentile(counts_arr, 90)),
                "p95": float(np.percentile(counts_arr, 95)),
                "max": int(counts_arr.max()),
                "empty_ratio": round(float((counts_arr == 0).mean()), 4),
                "dense_ratio_gt_50": round(float((counts_arr > 50).mean()), 4),
            },
            "max_iou_within_image": (
                {
                    "count": int(len(iou_arr)),
                    "mean": round(float(iou_arr.mean()), 4),
                    "median": round(float(np.median(iou_arr)), 4),
                    "p95": round(float(np.percentile(iou_arr, 95)), 4),
                    "max": round(float(iou_arr.max()), 4),
                } if len(iou_arr) > 0 else None
            ),
        }

    dt = time.time() - t0
    print(f"完成 ({dt:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# M5: 空间位置
# ---------------------------------------------------------------------------
def analyze_spatial(geo: dict) -> dict:
    print("  [M5] 空间位置... ", end="", flush=True)
    t0 = time.time()
    GRID = 16
    result = {}

    xywhr = geo["xywhr"]
    modals = geo["modal"]
    if xywhr.size == 0:
        return result

    # 质心归一化
    cx_norm = xywhr[:, 0] / IMAGE_SIZE
    cy_norm = xywhr[:, 1] / IMAGE_SIZE

    for modal in ("rgb", "ir"):
        m = modals == modal
        if not m.any():
            continue
        cx_m = cx_norm[m]
        cy_m = cy_norm[m]

        # clip to [0,1] 以防 minAreaRect 给出越界中心
        cx_m = np.clip(cx_m, 0, 1)
        cy_m = np.clip(cy_m, 0, 1)

        hist, _, _ = np.histogram2d(cy_m, cx_m, bins=GRID, range=[[0, 1], [0, 1]])

        quadrants = {
            "top_left": round(float(((cx_m < 0.5) & (cy_m < 0.5)).mean()), 4),
            "top_right": round(float(((cx_m >= 0.5) & (cy_m < 0.5)).mean()), 4),
            "bottom_left": round(float(((cx_m < 0.5) & (cy_m >= 0.5)).mean()), 4),
            "bottom_right": round(float(((cx_m >= 0.5) & (cy_m >= 0.5)).mean()), 4),
        }

        result[modal] = {
            "histogram_2d": hist.tolist(),
            "grid_size": GRID,
            "quadrant_ratios": quadrants,
            "total_boxes": int(m.sum()),
        }

    dt = time.time() - t0
    print(f"完成 ({dt:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# M6: 跨模态标签一致性
# ---------------------------------------------------------------------------
def analyze_cross_label_consistency(data: dict) -> dict:
    print("  [M6] 跨模态标签一致性... ", end="", flush=True)
    t0 = time.time()
    result = {"per_split": {}}

    all_count_diffs = []
    all_class_diffs = Counter()

    for split in SPLITS:
        rgb_by_stem = {stem: arr for stem, arr in data["rgb"][split]}
        ir_by_stem = {stem: arr for stem, arr in data["ir"][split]}
        common = sorted(set(rgb_by_stem) & set(ir_by_stem))
        only_rgb = set(rgb_by_stem) - set(ir_by_stem)
        only_ir = set(ir_by_stem) - set(rgb_by_stem)

        count_diffs = []
        class_diffs = Counter()
        empty_diffs = []

        for stem in common:
            rgb_arr = rgb_by_stem[stem]
            ir_arr = ir_by_stem[stem]
            n_rgb = len(rgb_arr)
            n_ir = len(ir_arr)
            count_diffs.append(n_ir - n_rgb)

            rgb_cls = Counter(rgb_arr[:, 0].astype(int).tolist()) if n_rgb > 0 else Counter()
            ir_cls = Counter(ir_arr[:, 0].astype(int).tolist()) if n_ir > 0 else Counter()
            for cls_id in range(NUM_CLASSES):
                diff = ir_cls.get(cls_id, 0) - rgb_cls.get(cls_id, 0)
                if diff != 0:
                    class_diffs[CLASS_NAMES[cls_id]] += diff

            if (n_rgb == 0) != (n_ir == 0):
                empty_diffs.append({
                    "stem": stem,
                    "rgb_empty": n_rgb == 0,
                    "ir_empty": n_ir == 0,
                })

        count_arr = np.array(count_diffs, dtype=np.int64) if count_diffs else np.empty(0, dtype=np.int64)

        result["per_split"][split] = {
            "common_stems": len(common),
            "only_rgb": len(only_rgb),
            "only_ir": len(only_ir),
            "count_diff_ir_minus_rgb": {
                "mean": round(float(count_arr.mean()), 3) if len(count_arr) else 0,
                "median": int(np.median(count_arr)) if len(count_arr) else 0,
                "min": int(count_arr.min()) if len(count_arr) else 0,
                "max": int(count_arr.max()) if len(count_arr) else 0,
                "std": round(float(count_arr.std()), 3) if len(count_arr) else 0,
                "positive_ratio": round(float((count_arr > 0).mean()), 4) if len(count_arr) else 0,
                "zero_ratio": round(float((count_arr == 0).mean()), 4) if len(count_arr) else 0,
            },
            "class_diff_ir_minus_rgb": {k: int(v) for k, v in class_diffs.items()},
            "empty_label_diffs": empty_diffs[:20],
        }

        all_count_diffs.extend(count_diffs)
        all_class_diffs.update(class_diffs)

    all_arr = np.array(all_count_diffs, dtype=np.int64) if all_count_diffs else np.empty(0, dtype=np.int64)
    result["overall"] = {
        "count_diff_ir_minus_rgb": {
            "mean": round(float(all_arr.mean()), 3) if len(all_arr) else 0,
            "median": int(np.median(all_arr)) if len(all_arr) else 0,
            "min": int(all_arr.min()) if len(all_arr) else 0,
            "max": int(all_arr.max()) if len(all_arr) else 0,
            "positive_ratio": round(float((all_arr > 0).mean()), 4) if len(all_arr) else 0,
            "zero_ratio": round(float((all_arr == 0).mean()), 4) if len(all_arr) else 0,
        },
        "class_diff_ir_minus_rgb": {k: int(v) for k, v in all_class_diffs.items()},
    }

    dt = time.time() - t0
    result["elapsed_seconds"] = round(dt, 3)
    print(f"完成 ({dt:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# M7: 跨模态像素域差（B 类，多进程采样）
# ---------------------------------------------------------------------------
def _process_one_pair(root: Path, split: str, stem: str) -> dict | None:
    """处理一对 RGB-IR 图像，返回统计指标（在 worker 进程中执行）。"""
    try:
        rgb_path = root / "images" / split / f"{stem}.jpg"
        ir_path = root / "images_ir" / split / f"{stem}.jpg"
        if not rgb_path.exists() or not ir_path.exists():
            return None

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        ir = np.array(Image.open(ir_path).convert("RGB"))

        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 皮尔逊相关
        flat_r = rgb_gray.flatten()
        flat_i = ir_gray.flatten()
        if flat_r.std() < 1e-6 or flat_i.std() < 1e-6:
            corr = 0.0
        else:
            corr = float(np.corrcoef(flat_r, flat_i)[0, 1])

        # Sobel 边缘 IoU（top 10% 梯度作为边缘）
        sobel_r = cv2.Sobel(rgb_gray, cv2.CV_64F, 1, 1, ksize=3)
        sobel_i = cv2.Sobel(ir_gray, cv2.CV_64F, 1, 1, ksize=3)
        mag_r = np.abs(sobel_r)
        mag_i = np.abs(sobel_i)
        thr_r = np.percentile(mag_r, 90)
        thr_i = np.percentile(mag_i, 90)
        edge_r = mag_r >= thr_r
        edge_i = mag_i >= thr_i
        inter = int((edge_r & edge_i).sum())
        union = int((edge_r | edge_i).sum())
        edge_iou = inter / union if union > 0 else 0.0

        # 图像熵（灰度直方图信息熵）
        def entropy(gray: np.ndarray) -> float:
            hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256])
            hist = hist.flatten().astype(np.float64)
            s = hist.sum()
            if s == 0:
                return 0.0
            p = hist / s
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))

        ent_r = entropy(rgb_gray)
        ent_i = entropy(ir_gray)

        # 亮度/对比度
        return {
            "correlation": corr,
            "edge_iou": edge_iou,
            "entropy_rgb": ent_r,
            "entropy_ir": ent_i,
            "mean_rgb": float(rgb_gray.mean()),
            "mean_ir": float(ir_gray.mean()),
            "std_rgb": float(rgb_gray.std()),
            "std_ir": float(ir_gray.std()),
        }
    except Exception:
        return None


def analyze_pixel_domain_gap(
    root: Path,
    sample_per_split: int = 2000,
    full: bool = False,
    workers: int | None = None,
) -> dict:
    print("  [M7] 跨模态像素域差... ", end="", flush=True)
    t0 = time.time()

    # 采样
    samples = []
    for split in SPLITS:
        stems = sorted(p.stem for p in (root / "images" / split).glob("*.jpg"))
        if not full and len(stems) > sample_per_split:
            rng = np.random.default_rng(42)
            stems = list(rng.choice(stems, sample_per_split, replace=False))
        for s in stems:
            samples.append((split, s))

    workers = workers or max(1, cpu_count() - 2)
    print(f"(采样 {len(samples)} 对, workers={workers}) ", end="", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_process_one_pair, root, sp, st) for sp, st in samples]
        for f in futs:
            r = f.result()
            if r is not None:
                results.append(r)

    if not results:
        return {"sample_count": 0, "error": "no valid pairs processed"}

    corr = np.array([r["correlation"] for r in results], dtype=np.float32)
    edge_iou = np.array([r["edge_iou"] for r in results], dtype=np.float32)
    ent_r = np.array([r["entropy_rgb"] for r in results], dtype=np.float32)
    ent_i = np.array([r["entropy_ir"] for r in results], dtype=np.float32)
    mean_r = np.array([r["mean_rgb"] for r in results], dtype=np.float32)
    mean_i = np.array([r["mean_ir"] for r in results], dtype=np.float32)
    std_r = np.array([r["std_rgb"] for r in results], dtype=np.float32)
    std_i = np.array([r["std_ir"] for r in results], dtype=np.float32)

    dt = time.time() - t0
    print(f"完成 ({dt:.1f}s)")
    return {
        "sample_count": len(results),
        "pixel_correlation_rgb_ir": {
            "mean": round(float(corr.mean()), 4),
            "median": round(float(np.median(corr)), 4),
            "std": round(float(corr.std()), 4),
            "min": round(float(corr.min()), 4),
            "max": round(float(corr.max()), 4),
            "p5": round(float(np.percentile(corr, 5)), 4),
            "p95": round(float(np.percentile(corr, 95)), 4),
        },
        "edge_iou": {
            "mean": round(float(edge_iou.mean()), 4),
            "median": round(float(np.median(edge_iou)), 4),
            "p5": round(float(np.percentile(edge_iou, 5)), 4),
            "p95": round(float(np.percentile(edge_iou, 95)), 4),
        },
        "entropy": {
            "rgb": {"mean": round(float(ent_r.mean()), 3), "median": round(float(np.median(ent_r)), 3)},
            "ir":  {"mean": round(float(ent_i.mean()), 3), "median": round(float(np.median(ent_i)), 3)},
        },
        "brightness_mean": {
            "rgb": round(float(mean_r.mean()), 2),
            "ir":  round(float(mean_i.mean()), 2),
        },
        "contrast_std": {
            "rgb": round(float(std_r.mean()), 2),
            "ir":  round(float(std_i.mean()), 2),
        },
        # 分布直方图数据（供出图）
        "_histograms": {
            "correlation": {"values": corr.tolist(), "bins": 50},
            "edge_iou": {"values": edge_iou.tolist(), "bins": 50},
            "entropy_rgb": ent_r.tolist(),
            "entropy_ir": ent_i.tolist(),
        },
        "elapsed_seconds": round(dt, 3),
    }


# ---------------------------------------------------------------------------
# 输出: JSON 报告
# ---------------------------------------------------------------------------
def write_json_report(report: dict, out_dir: Path) -> str:
    path = out_dir / "escvehicle_health_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(report), f, ensure_ascii=False, indent=2, default=str)
    return str(path)


# ---------------------------------------------------------------------------
# 输出: Markdown 摘要
# ---------------------------------------------------------------------------
def write_markdown_summary(report: dict, out_dir: Path) -> str:
    md = []
    md.append("# ESCVehicle_yolomm 数据集体检报告\n")
    md.append(f"**数据集**: `{report['meta']['root']}`  ")
    md.append(f"**生成时间**: {report['meta']['timestamp']}  ")
    md.append(f"**耗时**: {report['meta'].get('elapsed_seconds', 'N/A')}s  ")
    md.append(f"**总样本**: {report['meta']['total_images']} 张 (train {report['meta']['split_counts']['train']} / val {report['meta']['split_counts']['val']} / test {report['meta']['split_counts']['test']})\n")

    # 健康检查
    md.append("## 1. 健康检查\n")
    checks = report["integrity"].get("checks", [])
    for status, msg in checks:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(status, "")
        md.append(f"- {icon} **[{status}]** {msg}")

    if report["integrity"]["format_violations"]:
        md.append("\n**格式违规明细 (前 10 条)**:")
        for v in report["integrity"]["format_violations"][:10]:
            md.append(f"- `{v['file']}`: {v['error']}")

    if report["integrity"]["root_anomalies"]:
        md.append(f"\n**根目录异物**: {report['integrity']['root_anomalies']}")

    # 类别分布
    md.append("\n## 2. 类别分布与长尾\n")
    for modal in ("rgb", "ir"):
        total_data = report["class_distribution"][modal]["total"]
        md.append(f"### {modal.upper()} (总计 {total_data['total']} 个)\n")
        md.append(f"- 不平衡比 (max/min): **{total_data['imbalance_ratio']}**")
        md.append(f"- 基尼系数: **{total_data['gini']}**")
        md.append("\n| 类别 | 数量 | 占比 | 累积占比 |")
        md.append("|---|---:|---:|---:|")
        for row in total_data.get("per_class_top_k", []):
            md.append(f"| {row['class']} | {row['count']} | {row['percent']}% | {row['cumulative_percent']}% |")

    # OBB 几何
    md.append("\n## 3. OBB 几何\n")
    geo = report["geometry"]["per_class"]
    md.append("| 类别 | 数量 | AR P50 | AR P95 | 近方(AR<1.2) | 瘦长(AR>3) | 角度 P50(°) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for cls, d in geo.items():
        ar = d["aspect_ratio"]
        md.append(f"| {cls} | {d['count']} | {ar['p50']:.2f} | {ar['p95']:.2f} | {ar['near_square_ratio']:.1%} | {ar['elongated_ratio']:.1%} | {d['angle_deg']['median']:.1f} |")

    # 尺度
    md.append("\n## 4. 目标尺度分布 (COCO 口径)\n")
    md.append("| 类别 | small (≤32²) | medium (32²~96²) | large (>96²) | tiny (<8²) |")
    md.append("|---|---:|---:|---:|---:|")
    for cls, d in report["scale"]["per_class"].items():
        md.append(f"| {cls} | {d['small']['ratio']:.1%} | {d['medium']['ratio']:.1%} | {d['large']['ratio']:.1%} | {d['tiny_lt_8x8']['ratio']:.1%} |")

    overall_s = report["scale"]["overall"]
    md.append(f"\n**整体**: small {overall_s['small']['ratio']:.1%} / medium {overall_s['medium']['ratio']:.1%} / large {overall_s['large']['ratio']:.1%} / tiny {overall_s['tiny_lt_8x8']['ratio']:.1%}")

    # 密集度
    md.append("\n## 5. 每图密集度\n")
    for modal in ("rgb", "ir"):
        d = report["density"][modal]["per_image_count"]
        md.append(f"### {modal.upper()}")
        md.append(f"- 总图数: {d['total_images']}, 总目标: {d['total_boxes']}")
        md.append(f"- 每图目标数: min={d['min']}, mean={d['mean']}, median={d['median']}, P95={d['p95']}, max={d['max']}")
        md.append(f"- 空图占比: {d['empty_ratio']:.1%}, 密集(>50)占比: {d['dense_ratio_gt_50']:.1%}")
        iou_d = report["density"][modal].get("max_iou_within_image")
        if iou_d:
            md.append(f"- 图内最大 IoU (采样): mean={iou_d['mean']}, median={iou_d['median']}, P95={iou_d['p95']}")

    # 跨模态一致性
    md.append("\n## 6. 跨模态标签一致性 (RGB vs IR)\n")
    overall_c = report["cross_label_consistency"]["overall"]
    cdiff = overall_c["count_diff_ir_minus_rgb"]
    md.append(f"- IR 比 RGB 多的平均目标数: **{cdiff['mean']:+.3f}**")
    md.append(f"- IR 比 RGB 多的占比: {cdiff['positive_ratio']:.1%}, 完全相等: {cdiff['zero_ratio']:.1%}")
    md.append("\n**每类 IR-RGB 差异 (正=IR 多)**:")
    md.append("| 类别 | IR-RGB 差 |")
    md.append("|---|---:|")
    for cls, diff in sorted(overall_c.get("class_diff_ir_minus_rgb", {}).items(), key=lambda x: -abs(x[1])):
        md.append(f"| {cls} | {diff:+d} |")

    # 跨模态像素域差
    if "pixel_domain_gap" in report and report["pixel_domain_gap"].get("sample_count", 0) > 0:
        pdg = report["pixel_domain_gap"]
        md.append("\n## 7. 跨模态像素域差\n")
        md.append(f"- 采样: {pdg['sample_count']} 对")
        md.append(f"- RGB↔IR 皮尔逊相关: mean={pdg['pixel_correlation_rgb_ir']['mean']}, median={pdg['pixel_correlation_rgb_ir']['median']}")
        md.append(f"- Sobel 边缘 IoU: mean={pdg['edge_iou']['mean']}, median={pdg['edge_iou']['median']}")
        md.append(f"- 熵 RGB/IR: {pdg['entropy']['rgb']['mean']} / {pdg['entropy']['ir']['mean']}")
        md.append(f"- 亮度 mean RGB/IR: {pdg['brightness_mean']['rgb']} / {pdg['brightness_mean']['ir']}")
        md.append(f"- 对比度 std RGB/IR: {pdg['contrast_std']['rgb']} / {pdg['contrast_std']['ir']}")

    # 生成图片清单
    if report.get("figures"):
        md.append("\n## 生成图片\n")
        for name, paths in report["figures"].items():
            md.append(f"- **{name}**: {', '.join(f'`{p}`' for p in paths)}")

    md_path = out_dir / "escvehicle_health_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return str(md_path)


# ---------------------------------------------------------------------------
# 出图 (nature-figure 规范)
# ---------------------------------------------------------------------------
def generate_figures(report: dict, out_dir: Path) -> dict:
    print("  [figures] 生成投稿级图... ", end="", flush=True)
    t0 = time.time()
    apply_publication_style()
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # --- 图 1: 类别分布 ---
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = list(CLASS_NAMES)
    rgb_counts = [report["class_distribution"]["rgb"]["total"]["per_class"].get(c, 0) for c in classes]
    ir_counts = [report["class_distribution"]["ir"]["total"]["per_class"].get(c, 0) for c in classes]
    x = np.arange(len(classes))
    w = 0.35
    ax.bar(x - w / 2, rgb_counts, width=w, color=DEFAULT_COLORS[0], label="RGB", edgecolor="black", linewidth=0.8)
    ax.bar(x + w / 2, ir_counts, width=w, color=DEFAULT_COLORS[1], label="IR", edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Instance count (log)")
    ax.set_title("Class distribution (RGB vs IR)")
    ax.legend(loc="upper right")
    for xi, (rc, ic) in enumerate(zip(rgb_counts, ir_counts)):
        ax.text(xi - w / 2, rc * 1.15, str(rc), ha="center", va="bottom", fontsize=6)
        ax.text(xi + w / 2, ic * 1.15, str(ic), ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    paths["fig_class_distribution"] = save_pub_figure(fig, str(figs_dir / "fig_class_distribution"))

    # --- 图 2: OBB 几何（长宽比 + 角度）---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # 长宽比：每类箱线图
    # 需要原始数据，从 report 拿不到原始分布，用统计值替代
    # 实际出图时从 _geometry_raw 取；这里我们简化为柱状图 P50 + error bar (P25-P75)
    ar_p50 = [report["geometry"]["per_class"][c]["aspect_ratio"]["p50"] for c in classes if c in report["geometry"]["per_class"]]
    ar_p25 = [report["geometry"]["per_class"][c]["aspect_ratio"]["p25"] for c in classes if c in report["geometry"]["per_class"]]
    ar_p75 = [report["geometry"]["per_class"][c]["aspect_ratio"]["p75"] for c in classes if c in report["geometry"]["per_class"]]
    classes_present = [c for c in classes if c in report["geometry"]["per_class"]]
    if classes_present:
        x = np.arange(len(classes_present))
        ax1.bar(x, ar_p50, color=DEFAULT_COLORS[0], edgecolor="black", linewidth=0.8)
        ax1.errorbar(x, ar_p50, yerr=[np.array(ar_p50) - np.array(ar_p25), np.array(ar_p75) - np.array(ar_p50)],
                     fmt="none", ecolor="black", capsize=3)
        ax1.axhline(1.0, color=DEFAULT_COLORS[2], linestyle="--", linewidth=1, alpha=0.6, label="square (AR=1)")
        ax1.axhline(3.0, color=DEFAULT_COLORS[3], linestyle="--", linewidth=1, alpha=0.6, label="elongated (AR=3)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes_present, rotation=30, ha="right")
        ax1.set_ylabel("Aspect ratio")
        ax1.set_title("OBB aspect ratio (P50, error=P25-P75)")
        ax1.legend(loc="upper left", fontsize=7)

    # 角度分布（每类角度中位数柱状图）
    angle_p50 = [report["geometry"]["per_class"][c]["angle_deg"]["median"] for c in classes if c in report["geometry"]["per_class"]]
    if classes_present:
        x = np.arange(len(classes_present))
        ax2.bar(x, angle_p50, color=DEFAULT_COLORS[1], edgecolor="black", linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes_present, rotation=30, ha="right")
        ax2.set_ylabel("Angle median (°)")
        ax2.set_title("OBB rotation angle (median per class)")
        ax2.set_ylim(0, 90)
    fig.tight_layout()
    paths["fig_geometry"] = save_pub_figure(fig, str(figs_dir / "fig_geometry"))

    # --- 图 3: 尺度分布（堆叠）---
    fig, ax = plt.subplots(figsize=(7, 4))
    small_v = [report["scale"]["per_class"].get(c, {}).get("small", {}).get("ratio", 0) for c in classes]
    medium_v = [report["scale"]["per_class"].get(c, {}).get("medium", {}).get("ratio", 0) for c in classes]
    large_v = [report["scale"]["per_class"].get(c, {}).get("large", {}).get("ratio", 0) for c in classes]
    x = np.arange(len(classes))
    ax.bar(x, small_v, color=DEFAULT_COLORS[2], label="small (≤32²)", edgecolor="black", linewidth=0.8)
    ax.bar(x, medium_v, bottom=small_v, color=DEFAULT_COLORS[3], label="medium", edgecolor="black", linewidth=0.8)
    ax.bar(x, large_v, bottom=[s + m for s, m in zip(small_v, medium_v)],
           color=DEFAULT_COLORS[0], label="large (>96²)", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Object scale distribution (COCO buckets)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    paths["fig_scale"] = save_pub_figure(fig, str(figs_dir / "fig_scale"))

    # --- 图 4: 每图目标数分布 ---
    fig, ax = plt.subplots(figsize=(6, 4))
    # 用统计量 (min/median/P90/P95/max) 画 RGB vs IR 对比柱状图
    stats = ["min", "median", "p90", "p95", "max"]
    x = np.arange(len(stats))
    w = 0.35
    rgb_v = [report["density"]["rgb"]["per_image_count"][s] for s in stats]
    ir_v = [report["density"]["ir"]["per_image_count"][s] for s in stats]
    ax.bar(x - w / 2, rgb_v, width=w, color=DEFAULT_COLORS[0], label="RGB", edgecolor="black", linewidth=0.8)
    ax.bar(x + w / 2, ir_v, width=w, color=DEFAULT_COLORS[1], label="IR", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylabel("Objects per image")
    ax.set_title("Per-image object count (RGB vs IR)")
    ax.legend()
    fig.tight_layout()
    paths["fig_density"] = save_pub_figure(fig, str(figs_dir / "fig_density"))

    # --- 图 5: 空间位置热力图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for ax, modal, title in [(ax1, "rgb", "RGB"), (ax2, "ir", "IR")]:
        hist = np.array(report["spatial"][modal]["histogram_2d"])
        im = ax.imshow(hist, cmap="magma", aspect="equal", origin="upper")
        ax.set_title(f"{title} object centers")
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    paths["fig_spatial"] = save_pub_figure(fig, str(figs_dir / "fig_spatial"))

    # --- 图 6: 跨模态像素域差 ---
    if "pixel_domain_gap" in report and "_histograms" in report["pixel_domain_gap"]:
        h = report["pixel_domain_gap"]["_histograms"]
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        # 相关系数分布
        ax = axes[0, 0]
        ax.hist(h["correlation"]["values"], bins=h["correlation"]["bins"],
                color=DEFAULT_COLORS[0], edgecolor="black", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("Pearson r (RGB ↔ IR)")
        ax.set_ylabel("Count")
        ax.set_title("Pixel correlation")
        ax.axvline(report["pixel_domain_gap"]["pixel_correlation_rgb_ir"]["mean"],
                   color=DEFAULT_COLORS[2], linestyle="--", linewidth=1.5, label="mean")
        ax.legend(fontsize=7)

        # 边缘 IoU 分布
        ax = axes[0, 1]
        ax.hist(h["edge_iou"]["values"], bins=h["edge_iou"]["bins"],
                color=DEFAULT_COLORS[1], edgecolor="black", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("Sobel edge IoU")
        ax.set_ylabel("Count")
        ax.set_title("Edge overlap (RGB ↔ IR)")

        # 熵对比
        ax = axes[1, 0]
        ax.hist(h["entropy_rgb"], bins=40, color=DEFAULT_COLORS[0], alpha=0.6, label="RGB", edgecolor="black", linewidth=0.6)
        ax.hist(h["entropy_ir"], bins=40, color=DEFAULT_COLORS[1], alpha=0.6, label="IR", edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Entropy (bits)")
        ax.set_ylabel("Count")
        ax.set_title("Image entropy")
        ax.legend(fontsize=7)

        # 亮度对比
        ax = axes[1, 1]
        # 无原始数据，用表格代替：显示 report 中的 summary
        ax.axis("off")
        pdg = report["pixel_domain_gap"]
        summary_text = (
            f"Pixel correlation: {pdg['pixel_correlation_rgb_ir']['mean']:.3f} ± {pdg['pixel_correlation_rgb_ir']['std']:.3f}\n"
            f"Edge IoU: {pdg['edge_iou']['mean']:.3f}\n"
            f"Entropy RGB: {pdg['entropy']['rgb']['mean']:.2f} bits\n"
            f"Entropy IR:  {pdg['entropy']['ir']['mean']:.2f} bits\n"
            f"Brightness RGB: {pdg['brightness_mean']['rgb']:.1f}\n"
            f"Brightness IR:  {pdg['brightness_mean']['ir']:.1f}\n"
            f"Contrast std RGB: {pdg['contrast_std']['rgb']:.1f}\n"
            f"Contrast std IR:  {pdg['contrast_std']['ir']:.1f}"
        )
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="center", fontfamily="monospace")
        ax.set_title("Cross-modal summary")

        fig.suptitle("Cross-modal pixel domain gap (RGB vs IR)", fontsize=10, y=1.02)
        fig.tight_layout()
        paths["fig_cross_modal"] = save_pub_figure(fig, str(figs_dir / "fig_cross_modal"))

    dt = time.time() - t0
    print(f"完成 ({dt:.1f}s)")
    return paths


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="ESCVehicle_yolomm 数据集全自动体检")
    parser.add_argument("--root", type=Path, default=Path("/root/datasets/ESCVehicle_yolomm"),
                        help="数据集根目录")
    parser.add_argument("--out", type=Path, default=Path("ResDatasetCheck"),
                        help="输出目录")
    parser.add_argument("--pixel-sample", type=int, default=2000,
                        help="M7 每 split 采样数（默认 2000）")
    parser.add_argument("--full-pixel", action="store_true",
                        help="M7 全量读图（较慢）")
    parser.add_argument("--no-figures", action="store_true",
                        help="跳过出图")
    parser.add_argument("--no-pixel", action="store_true",
                        help="跳过 M7（跨模态像素域差）")
    parser.add_argument("--workers", type=int, default=None,
                        help="M7 多进程 workers 数")
    args = parser.parse_args()

    root: Path = args.root
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        print(f"ERROR: 数据集根目录不存在: {root}", file=sys.stderr)
        return 1

    t_total = time.time()
    print(f"\n=== ESCVehicle_yolomm 数据集体检 ===")
    print(f"数据集: {root}")
    print(f"输出: {out_dir}\n")

    report = {
        "meta": {
            "root": str(root),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split_counts": {
                "train": len(list((root / "images" / "train").glob("*.jpg"))),
                "val": len(list((root / "images" / "val").glob("*.jpg"))),
                "test": len(list((root / "images" / "test").glob("*.jpg"))),
            },
        }
    }
    report["meta"]["total_images"] = sum(report["meta"]["split_counts"].values())

    print("--- 标签层分析 (M0–M6) ---")
    report["integrity"] = check_integrity(root)
    data = load_all_labels(root)
    report["class_distribution"] = analyze_class_distribution(data)
    # 把 top_k 提升到 total 层方便 MD 输出
    for modal in ("rgb", "ir"):
        for split in SPLITS:
            report["class_distribution"][modal]["per_split"][split]["per_class_top_k"] = \
                report["class_distribution"][modal]["per_split"][split].pop("top_k", [])
        # total 没有 top_k 字段，从 per_split 汇总重建
        total_counts = report["class_distribution"][modal]["total"]["per_class"]
        sorted_counts = sorted(total_counts.items(), key=lambda x: -x[1])
        total = sum(total_counts.values())
        cum = 0
        top_k = []
        for cls, cnt in sorted_counts:
            cum += cnt
            top_k.append({
                "class": cls, "count": cnt,
                "percent": round(cnt / total * 100, 3) if total else 0,
                "cumulative_percent": round(cum / total * 100, 3) if total else 0,
            })
        report["class_distribution"][modal]["total"]["per_class_top_k"] = top_k

    geo = compute_all_xywhr(data)
    report["geometry"] = analyze_geometry(geo)
    report["scale"] = analyze_scale(geo)
    report["density"] = analyze_density(data)
    report["spatial"] = analyze_spatial(geo)
    report["cross_label_consistency"] = analyze_cross_label_consistency(data)

    # M7 独立（需读图像）
    if not args.no_pixel:
        print("\n--- 像素层分析 (M7) ---")
        report["pixel_domain_gap"] = analyze_pixel_domain_gap(
            root, sample_per_split=args.pixel_sample,
            full=args.full_pixel, workers=args.workers,
        )

    print("\n--- 输出报告 ---")
    json_path = write_json_report(report, out_dir)
    print(f"  JSON: {json_path}")
    md_path = write_markdown_summary(report, out_dir)
    print(f"  Markdown: {md_path}")

    if not args.no_figures:
        fig_paths = generate_figures(report, out_dir)
        report["figures"] = fig_paths
        # 更新 JSON/MD 以包含图片路径
        write_json_report(report, out_dir)
        write_markdown_summary(report, out_dir)

    total_time = time.time() - t_total
    report["meta"]["elapsed_seconds"] = round(total_time, 1)
    write_json_report(report, out_dir)
    write_markdown_summary(report, out_dir)

    print(f"\n✅ 体检完成，总耗时 {total_time:.1f}s")
    print(f"   JSON: {json_path}")
    print(f"   MD:   {md_path}")
    if not args.no_figures:
        print(f"   图:   {out_dir / 'figs'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
