from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _extract_prefixed(mapping: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Extract key/value pairs from mapping whose key starts with prefix."""
    return {k: v for k, v in mapping.items() if isinstance(k, str) and k.startswith(prefix)}


@dataclass(frozen=True)
class COCOValResults:
    """COCO 扩展验证结果（扁平字典 + 可选保存目录）。"""

    metrics: Dict[str, Any]
    save_dir: Optional[Path] = None

    @classmethod
    def from_stats_dict(cls, stats: Dict[str, Any], *, save_dir: Optional[Path] = None) -> "COCOValResults":
        # 兼容多种前缀：
        # - metrics/coco/*（检测 COCO）
        # - metrics/coco_mask/*、metrics/coco_box/*（分割 COCO）
        return cls(metrics=_extract_prefixed(stats, "metrics/coco"), save_dir=save_dir)


@dataclass(frozen=True)
class MMValResults:
    """
    统一的验证结果容器：
    - standard: 标准验证器 metrics 对象（DetMetrics/SegmentMetrics/...）
    - coco: 可选 COCO 扩展结果
    - results_dict: 扁平 dict（用于日志/导出）
    """

    standard: Any
    coco: Optional[COCOValResults]
    results_dict: Dict[str, Any]
    save_dir: Optional[Path] = None

    def __getattr__(self, name: str) -> Any:
        # 代理到标准 metrics，保持用户旧代码兼容性；不吞异常（Fail-Fast）
        return getattr(self.standard, name)


# ---- Validation metrics CSV writer -------------------------------------------------
# 将标准验证标量指标落盘为 CSV，供 valMM/valMMseg/valMMpose/valMMobb 复用。
# 与 COCO 验证器自有的 _save_csv_results()（多文件 COCO 报告）解耦：本函数只处理标准指标。

import csv as _csv

# 任务 -> [(分支属性名, 类名后缀)] 映射；detect/obb 仅 box，segment 加 mask，pose 加 keypoint
_VAL_CSV_BRANCHES = {
    "detect": [("box", "")],
    "obb": [("box", "")],
    "segment": [("box", "_Box"), ("seg", "_Mask")],
    "pose": [("box", "_Box"), ("pose", "_Keypoint")],
}


def save_validation_csv(results_dict, metrics_obj, save_dir, task="detect", class_names=None):
    """
    保存标准验证指标到 2 个 CSV 文件（覆盖写入）。

    生成文件：
        - metrics.csv          : 总体摘要（单行，所有 metrics/* 与 fitness 字段）
        - metrics_per_class.csv: 每类明细（Class,Precision,Recall,F1,mAP50,mAP50-95）

    Args:
        results_dict (dict): 扁平指标字典，来自 validator.metrics.results_dict
        metrics_obj: 标准 metrics 对象（DetMetrics/SegmentMetrics/...），需含 .box 分支
        save_dir (str | Path): 输出目录
        task (str): 任务类型 detect/segment/pose/obb
        class_names (dict | None): {class_index: class_name} 映射
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    class_names = class_names or {}

    _save_overall_csv(results_dict, save_dir / "metrics.csv")
    _save_per_class_csv(metrics_obj, save_dir / "metrics_per_class.csv", task, class_names)


def _save_overall_csv(results_dict, csv_path):
    """写入总体指标（单行），仅保留 metrics/* 与 fitness 字段。"""
    if not results_dict:
        return
    filtered = {
        k: v
        for k, v in results_dict.items()
        if isinstance(k, str) and (k.startswith("metrics/") or k == "fitness")
    }
    if not filtered:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(filtered.keys()))
        writer.writeheader()
        writer.writerow({k: _format_metric(v) for k, v in filtered.items()})


def _save_per_class_csv(metrics_obj, csv_path, task, class_names):
    """写入每类指标；按 task 选择分支（box/seg/pose），分支名以类名后缀区分。"""
    branches = _VAL_CSV_BRANCHES.get(task, [("box", "")])
    rows = []
    for attr, suffix in branches:
        branch = getattr(metrics_obj, attr, None)
        rows.extend(_extract_branch_rows(branch, class_names, suffix))
    if not rows:
        return
    fieldnames = ["Class", "Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_branch_rows(branch, class_names, suffix):
    """从一个 Metric 分支（box/seg/pose）提取每类指标行。class_result(i) -> (p, r, ap50, ap)。"""
    rows = []
    if branch is None or not hasattr(branch, "ap_class_index") or not len(branch.ap_class_index):
        return rows
    ap_class_index = branch.ap_class_index
    f1_arr = getattr(branch, "f1", [])
    for i, c in enumerate(ap_class_index):
        c = int(c)
        name = class_names.get(c, f"class_{c}")
        p, r, ap50, ap = branch.class_result(i)
        f1 = float(f1_arr[i]) if i < len(f1_arr) else 0.0
        rows.append(
            {
                "Class": f"{name}{suffix}",
                "Precision": _format_metric(p),
                "Recall": _format_metric(r),
                "F1": _format_metric(f1),
                "mAP50": _format_metric(ap50),
                "mAP50-95": _format_metric(ap),
            }
        )
    return rows


def _format_metric(v):
    """指标值格式化为字符串（%.6g，与 trainer save_metrics 一致）。"""
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)
