"""双模态 LabelMe OBB → YOLOMM 数据集转换（CLI）。

通用契约：
- split 清单为 JSON list，每条含 ``img``（不带目录的 ``.jpg``）与 ``split``
- 源图像/标签按相对目录 + 同名 stem 配对
- 标签为 LabelMe JSON，每个 shape 恰好 4 个角点
- 输出为 YOLOMM 目录 + data yaml + conversion_report.json
- 图像用硬链接，发布使用 renameat2(RENAME_NOREPLACE)，失败回滚 staging

用法：
    /root/miniconda3/envs/yolon/bin/python -m ultralytics.tools.mm_obb_convert \\
        --preset escvehicle --source /data/ESCVehicle --output /data/ESCVehicle_yolomm

    /root/miniconda3/envs/yolon/bin/python -m ultralytics.tools.mm_obb_convert \\
        --source /data/raw --output /data/yolomm \\
        --classes car suv truck --x-name ir
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
import shutil
import struct
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

MAX_REPORT_EXAMPLES = 20
DEFAULT_SPLITS = ("train", "val", "test")


class ConversionError(RuntimeError):
    """输入、标签或发布过程不满足转换契约。"""


@dataclass(frozen=True)
class DualModalLabelMeSpec:
    """双模态 LabelMe OBB 转换规格。"""

    class_names: tuple[str, ...]
    splits: tuple[str, ...] = DEFAULT_SPLITS
    split_file: str = "data.json"
    rgb_image_dir: str = "visible/images"
    x_image_dir: str = "infrared/images"
    rgb_label_dir: str = "visible/labels"
    x_label_dir: str = "infrared/labels"
    rgb_image_out: str = "images"
    x_image_out: str = "images_ir"
    rgb_label_out: str = "labels"
    x_label_out: str = "label-ir"
    yaml_name: str = "data.yaml"
    rgb_modality: str = "rgb"
    x_modality: str = "ir"
    expected_counts: Mapping[str, int] | None = None
    rgb_stats_key: str = "rgb"
    x_stats_key: str = "x"


PRESETS: dict[str, DualModalLabelMeSpec] = {
    "escvehicle": DualModalLabelMeSpec(
        class_names=(
            "car",
            "suv",
            "truck",
            "bus",
            "van",
            "freight car",
            "construction vehicle",
        ),
        expected_counts={"train": 6983, "val": 507, "test": 3237},
        yaml_name="escvehicle_yolomm.yaml",
        rgb_stats_key="visible",
        x_stats_key="infrared",
    ),
}


@dataclass
class LabelStats:
    """累计一套模态标签的转换统计。"""

    source_objects: int = 0
    written_objects: int = 0
    clipped_objects: int = 0
    clipped_points: int = 0
    dropped_degenerate: int = 0
    dropped_duplicate: int = 0
    source_empty_files: int = 0
    output_empty_files: int = 0
    class_counts: Counter[str] = field(default_factory=Counter)
    clipped_examples: list[dict[str, Any]] = field(default_factory=list)
    dropped_examples: list[dict[str, Any]] = field(default_factory=list)
    duplicate_examples: list[dict[str, Any]] = field(default_factory=list)

    def merge(self, other: "LabelStats") -> None:
        """把单文件统计累加到当前统计。"""
        self.source_objects += other.source_objects
        self.written_objects += other.written_objects
        self.clipped_objects += other.clipped_objects
        self.clipped_points += other.clipped_points
        self.dropped_degenerate += other.dropped_degenerate
        self.dropped_duplicate += other.dropped_duplicate
        self.source_empty_files += other.source_empty_files
        self.output_empty_files += other.output_empty_files
        self.class_counts.update(other.class_counts)
        remaining = MAX_REPORT_EXAMPLES - len(self.clipped_examples)
        if remaining > 0:
            self.clipped_examples.extend(other.clipped_examples[:remaining])
        remaining = MAX_REPORT_EXAMPLES - len(self.dropped_examples)
        if remaining > 0:
            self.dropped_examples.extend(other.dropped_examples[:remaining])
        remaining = MAX_REPORT_EXAMPLES - len(self.duplicate_examples)
        if remaining > 0:
            self.duplicate_examples.extend(other.duplicate_examples[:remaining])

    def to_dict(self, class_names: Sequence[str]) -> dict[str, Any]:
        """返回可直接写入 JSON 的稳定结构。"""
        return {
            "source_objects": self.source_objects,
            "written_objects": self.written_objects,
            "clipped_objects": self.clipped_objects,
            "clipped_points": self.clipped_points,
            "dropped_degenerate": self.dropped_degenerate,
            "dropped_duplicate": self.dropped_duplicate,
            "source_empty_files": self.source_empty_files,
            "output_empty_files": self.output_empty_files,
            "class_counts": {name: self.class_counts.get(name, 0) for name in class_names},
            "clipped_examples": self.clipped_examples,
            "dropped_examples": self.dropped_examples,
            "duplicate_examples": self.duplicate_examples,
        }


def load_split_records(
    source: Path,
    spec: DualModalLabelMeSpec,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, list[str]]:
    """读取并校验 split 清单。"""
    split_file = source / spec.split_file
    if not split_file.is_file():
        raise ConversionError(f"缺少 split 文件: {split_file}")

    try:
        records = json.loads(split_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"无法解析 split 文件 {split_file}: {exc}") from exc
    if not isinstance(records, list):
        raise ConversionError(f"{split_file} 顶层必须是 list")

    by_split = {split: [] for split in spec.splits}
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ConversionError(f"{split_file} 第 {index} 条记录不是对象")
        image_name = record.get("img")
        split = record.get("split")
        if not isinstance(image_name, str) or not image_name:
            raise ConversionError(f"{split_file} 第 {index} 条记录缺少有效 img")
        image_path = Path(image_name)
        if image_path.name != image_name or image_path.suffix.lower() != ".jpg":
            raise ConversionError(f"{split_file} 第 {index} 条 img 必须是不带目录的 JPG: {image_name!r}")
        if split not in spec.splits:
            raise ConversionError(f"{split_file} 第 {index} 条含非法 split: {split!r}")
        if image_name in seen:
            raise ConversionError(f"{split_file} 含重复图片记录: {image_name}")
        seen.add(image_name)
        by_split[split].append(image_name)

    missing = [split for split, names in by_split.items() if not names]
    if missing:
        raise ConversionError(f"{split_file} 缺少非空 split: {missing}")
    if expected_counts is not None:
        actual_counts = {split: len(by_split[split]) for split in spec.splits}
        expected = {split: int(expected_counts[split]) for split in spec.splits}
        if actual_counts != expected:
            raise ConversionError(f"split 数量不符: expected={expected} actual={actual_counts}")
    return by_split


def _polygon_area2(points: list[tuple[float, float]]) -> float:
    """返回多边形鞋带公式的两倍有符号面积。"""
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))


def _float32_bytes(value: float) -> bytes:
    """用 IEEE-754 float32 位模式复现 Ultralytics 标签缓存的比较精度。"""
    return struct.pack("<f", value)


def _loader_duplicate_key(class_id: int, points: list[tuple[float, float]]) -> tuple[Any, ...]:
    """返回与 ``verify_image_label`` 水平外接框去重语义一致的键。"""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        class_id,
        _float32_bytes(min(xs)),
        _float32_bytes(min(ys)),
        _float32_bytes(max(xs)),
        _float32_bytes(max(ys)),
    )


def _parse_point(raw_point: Any, label_path: Path, shape_index: int, point_index: int) -> tuple[float, float]:
    """解析一个严格的二维有限数值点。"""
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        raise ConversionError(f"{label_path} shape[{shape_index}] point[{point_index}] 不是二维点: {raw_point!r}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_point):
        raise ConversionError(f"{label_path} shape[{shape_index}] point[{point_index}] 含非数值坐标: {raw_point!r}")
    x, y = float(raw_point[0]), float(raw_point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ConversionError(f"{label_path} shape[{shape_index}] point[{point_index}] 含非有限坐标: {raw_point!r}")
    return x, y


def convert_label(
    label_path: Path,
    class_to_id: dict[str, int],
    expected_image_size: tuple[int, int] | None = None,
) -> tuple[list[str], LabelStats]:
    """把一个 LabelMe JSON 转换为 YOLO OBB 文本行。"""
    try:
        document = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"无法解析标签 {label_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConversionError(f"标签顶层必须是对象: {label_path}")

    width = document.get("imageWidth")
    height = document.get("imageHeight")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or not math.isfinite(float(width))
        or not math.isfinite(float(height))
        or float(width) <= 0
        or float(height) <= 0
    ):
        raise ConversionError(f"标签宽高无效: {label_path} width={width!r} height={height!r}")
    width, height = float(width), float(height)
    if expected_image_size is not None and (width, height) != tuple(map(float, expected_image_size)):
        raise ConversionError(
            f"标签宽高与图像不一致: {label_path} label={(width, height)} image={expected_image_size}"
        )

    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        raise ConversionError(f"标签 shapes 必须是 list: {label_path}")

    lines: list[str] = []
    stats = LabelStats(source_empty_files=int(not shapes))
    seen_loader_boxes: set[tuple[Any, ...]] = set()
    for shape_index, shape in enumerate(shapes):
        stats.source_objects += 1
        if not isinstance(shape, dict):
            raise ConversionError(f"{label_path} shape[{shape_index}] 不是对象")
        raw_label = shape.get("label")
        if not isinstance(raw_label, str):
            raise ConversionError(f"{label_path} shape[{shape_index}] 缺少字符串 label")
        label = raw_label.strip().lower()
        if label not in class_to_id:
            raise ConversionError(f"{label_path} shape[{shape_index}] 含未知类别: {raw_label!r}")
        raw_points = shape.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            raise ConversionError(
                f"{label_path} shape[{shape_index}] OBB 必须恰好含四个角点，当前为: {raw_points!r}"
            )

        clipped_points: list[tuple[float, float]] = []
        clipped_point_count = 0
        for point_index, raw_point in enumerate(raw_points):
            x, y = _parse_point(raw_point, label_path, shape_index, point_index)
            clipped_x = min(max(x, 0.0), width)
            clipped_y = min(max(y, 0.0), height)
            if clipped_x != x or clipped_y != y:
                clipped_point_count += 1
            clipped_points.append((clipped_x, clipped_y))

        if clipped_point_count:
            stats.clipped_objects += 1
            stats.clipped_points += clipped_point_count
            stats.clipped_examples.append({"file": label_path.name, "shape_index": shape_index, "label": label})

        unique_points = set(clipped_points)
        if len(unique_points) < 3 or abs(_polygon_area2(clipped_points)) < 1e-9:
            stats.dropped_degenerate += 1
            stats.dropped_examples.append({"file": label_path.name, "shape_index": shape_index, "label": label})
            continue

        normalized = [(x / width, y / height) for x, y in clipped_points]
        formatted_points = [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in normalized]
        duplicate_key = _loader_duplicate_key(class_to_id[label], formatted_points)
        if duplicate_key in seen_loader_boxes:
            stats.dropped_duplicate += 1
            stats.duplicate_examples.append({"file": label_path.name, "shape_index": shape_index, "label": label})
            continue
        seen_loader_boxes.add(duplicate_key)

        values = [f"{class_to_id[label]}"]
        values.extend(f"{coordinate:.8f}" for point in formatted_points for coordinate in point)
        lines.append(" ".join(values))
        stats.written_objects += 1
        stats.class_counts[label] += 1

    stats.output_empty_files = int(not lines)
    return lines, stats


def _write_text_lines(path: Path, lines: list[str]) -> None:
    """写入标签行；空标签保留零字节文件。"""
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def inspect_jpeg(path: Path) -> tuple[int, int]:
    """验证 JPEG 可读性和结束标记，返回 ``(width, height)``。"""
    try:
        with path.open("rb") as handle:
            handle.seek(-2, os.SEEK_END)
            if handle.read() != b"\xff\xd9":
                raise ConversionError(f"JPEG 缺少结束标记，拒绝硬链接: {path}")
        with Image.open(path) as image:
            size = image.size
            image.verify()
            if image.format != "JPEG":
                raise ConversionError(f"扩展名为 JPG 但实际格式不是 JPEG: {path}")
    except ConversionError:
        raise
    except (OSError, SyntaxError) as exc:
        raise ConversionError(f"JPEG 无法读取: {path}: {exc}") from exc
    if size[0] <= 0 or size[1] <= 0:
        raise ConversionError(f"JPEG 尺寸无效: {path} size={size}")
    return int(size[0]), int(size[1])


def publish_no_replace(staging: Path, output: Path) -> None:
    """使用 Linux ``renameat2(RENAME_NOREPLACE)`` 原子发布且绝不覆盖目标。"""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ConversionError("当前系统不支持 renameat2，无法保证无覆盖原子发布")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ConversionError(f"发布时目标已存在，拒绝覆盖: {output}")
    raise ConversionError(
        f"原子发布失败: {staging} -> {output}: [Errno {error_number}] {os.strerror(error_number)}"
    )


def validate_label_file(path: Path, num_classes: int) -> int:
    """验证一个 YOLO OBB 标签文件并返回目标行数。"""
    count = 0
    seen_loader_boxes: set[tuple[Any, ...]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ConversionError(f"{path}:{line_number} 必须为九列，当前 {len(fields)} 列")
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ConversionError(f"{path}:{line_number} 类别 ID 不是整数") from exc
        if str(class_id) != fields[0] or not 0 <= class_id < num_classes:
            raise ConversionError(f"{path}:{line_number} 类别 ID 越界: {fields[0]}")
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ConversionError(f"{path}:{line_number} 坐标不是浮点数") from exc
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
            raise ConversionError(f"{path}:{line_number} 坐标非有限或超出 [0,1]")
        points = list(zip(coordinates[0::2], coordinates[1::2]))
        if len(set(points)) < 3 or abs(_polygon_area2(points)) < 1e-12:
            raise ConversionError(f"{path}:{line_number} 为退化 OBB")
        duplicate_key = _loader_duplicate_key(class_id, points)
        if duplicate_key in seen_loader_boxes:
            raise ConversionError(f"{path}:{line_number} 为 YOLOMM 加载语义下的重复 OBB")
        seen_loader_boxes.add(duplicate_key)
        count += 1
    return count


def validate_built_dataset(root: Path, report: dict[str, Any], spec: DualModalLabelMeSpec) -> None:
    """全量验证转换后的目录、配对关系和标签数值。"""
    expected_root_entries = {
        spec.rgb_image_out,
        spec.x_image_out,
        spec.rgb_label_out,
        spec.x_label_out,
        spec.yaml_name,
        "conversion_report.json",
    }
    actual_root_entries = {path.name for path in root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ConversionError(
            f"输出根目录内容不符合设计，expected={sorted(expected_root_entries)} actual={sorted(actual_root_entries)}"
        )

    rgb_lines = 0
    x_lines = 0
    num_classes = len(spec.class_names)
    for split in spec.splits:
        directory_sets: dict[str, set[str]] = {}
        for directory, suffix in (
            (spec.rgb_image_out, ".jpg"),
            (spec.x_image_out, ".jpg"),
            (spec.rgb_label_out, ".txt"),
            (spec.x_label_out, ".txt"),
        ):
            split_dir = root / directory / split
            if not split_dir.is_dir():
                raise ConversionError(f"输出缺少目录: {split_dir}")
            files = list(split_dir.glob(f"*{suffix}"))
            directory_sets[directory] = {path.stem for path in files}
            if len(files) != len(directory_sets[directory]):
                raise ConversionError(f"{split_dir} 存在重复 stem")

        reference = directory_sets[spec.rgb_image_out]
        mismatched = [name for name, stems in directory_sets.items() if stems != reference]
        if mismatched:
            raise ConversionError(f"split={split} 目录 stem 不一致: {mismatched}")
        expected_count = int(report["split_counts"][split])
        if len(reference) != expected_count:
            raise ConversionError(f"split={split} 样本数不符: expected={expected_count} actual={len(reference)}")
        rgb_label_dir = root / spec.rgb_label_out / split
        x_label_dir = root / spec.x_label_out / split
        rgb_lines += sum(validate_label_file(path, num_classes) for path in rgb_label_dir.glob("*.txt"))
        x_lines += sum(validate_label_file(path, num_classes) for path in x_label_dir.glob("*.txt"))

    expected_rgb = int(report["labels"][spec.rgb_stats_key]["written_objects"])
    expected_x = int(report["labels"][spec.x_stats_key]["written_objects"])
    if rgb_lines != expected_rgb or x_lines != expected_x:
        raise ConversionError(
            "标签行数与报告不一致: "
            f"{spec.rgb_stats_key}={rgb_lines}/{expected_rgb}, {spec.x_stats_key}={x_lines}/{expected_x}"
        )


def _write_dataset_yaml(staging: Path, output: Path, spec: DualModalLabelMeSpec) -> None:
    """写入 YOLOMM 双模态 OBB 数据配置。"""
    dataset_yaml = {
        "path": str(output),
        "train": f"{spec.rgb_image_out}/train",
        "val": f"{spec.rgb_image_out}/val",
        "test": f"{spec.rgb_image_out}/test",
        "Pch": 3,
        "Xch": 3,
        "modality": {spec.rgb_modality: spec.rgb_image_out, spec.x_modality: spec.x_image_out},
        "modality_used": [spec.rgb_modality, spec.x_modality],
        "names": {index: name for index, name in enumerate(spec.class_names)},
    }
    (staging / spec.yaml_name).write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def build_dataset(
    source: Path,
    output: Path,
    spec: DualModalLabelMeSpec,
    *,
    enforce_counts: bool | None = None,
) -> dict[str, Any]:
    """在 staging 中完整转换并原子发布双模态 YOLOMM 数据集。"""
    source = source.resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists():
        raise ConversionError(f"目标目录已存在，拒绝覆盖: {output}")
    if output == source or source in output.parents:
        raise ConversionError(f"目标目录不得等于或位于源目录内部: {output}")
    if not output.parent.is_dir():
        raise ConversionError(f"目标父目录不存在: {output.parent}")

    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise ConversionError(f"staging 已存在，拒绝覆盖: {staging}")

    should_enforce = spec.expected_counts is not None if enforce_counts is None else enforce_counts
    expected_counts = dict(spec.expected_counts) if should_enforce and spec.expected_counts is not None else None
    if should_enforce and expected_counts is None:
        raise ConversionError("已要求校验 split 数量，但规格未提供 expected_counts")

    by_split = load_split_records(source, spec, expected_counts)
    class_to_id = {name: index for index, name in enumerate(spec.class_names)}
    rgb_stats = LabelStats()
    x_stats = LabelStats()
    hardlinks = {spec.rgb_stats_key: 0, spec.x_stats_key: 0}
    created_staging = False

    try:
        staging.mkdir()
        created_staging = True
        for directory in (spec.rgb_image_out, spec.x_image_out, spec.rgb_label_out, spec.x_label_out):
            for split in spec.splits:
                (staging / directory / split).mkdir(parents=True)

        for split in spec.splits:
            for image_name in by_split[split]:
                stem = Path(image_name).stem
                rgb_image = source / spec.rgb_image_dir / image_name
                x_image = source / spec.x_image_dir / image_name
                rgb_label = source / spec.rgb_label_dir / f"{stem}.json"
                x_label = source / spec.x_label_dir / f"{stem}.json"
                required = (rgb_image, x_image, rgb_label, x_label)
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise ConversionError(f"样本 {image_name} 缺少源文件: {missing}")

                rgb_size = inspect_jpeg(rgb_image)
                x_size = inspect_jpeg(x_image)
                os.link(rgb_image, staging / spec.rgb_image_out / split / image_name)
                hardlinks[spec.rgb_stats_key] += 1
                os.link(x_image, staging / spec.x_image_out / split / image_name)
                hardlinks[spec.x_stats_key] += 1

                rgb_lines, file_stats = convert_label(rgb_label, class_to_id, expected_image_size=rgb_size)
                rgb_stats.merge(file_stats)
                _write_text_lines(staging / spec.rgb_label_out / split / f"{stem}.txt", rgb_lines)

                x_lines, file_stats = convert_label(x_label, class_to_id, expected_image_size=x_size)
                x_stats.merge(file_stats)
                _write_text_lines(staging / spec.x_label_out / split / f"{stem}.txt", x_lines)

        _write_dataset_yaml(staging, output, spec)
        report = {
            "format_version": 1,
            "source_root": str(source),
            "output_root": str(output),
            "link_mode": "hardlink",
            "class_names": list(spec.class_names),
            "split_counts": {split: len(by_split[split]) for split in spec.splits},
            "image_links": hardlinks,
            "labels": {
                spec.rgb_stats_key: rgb_stats.to_dict(spec.class_names),
                spec.x_stats_key: x_stats.to_dict(spec.class_names),
            },
            "pairing_key": f"{spec.split_file} img filename and label JSON stem",
            "ignored_json_fields": ["imagePath", "imageData"],
            "official_split_counts_enforced": bool(expected_counts),
        }
        (staging / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_built_dataset(staging, report, spec)
        publish_no_replace(staging, output)
        return report
    except Exception:
        if created_staging and staging.exists():
            shutil.rmtree(staging)
        raise


def _parse_expected_counts(text: str | None) -> dict[str, int] | None:
    """解析 ``train=1,val=2,test=3`` 形式的数量约束。"""
    if not text:
        return None
    counts: dict[str, int] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ConversionError(f"expected-counts 项缺少 '=': {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConversionError(f"expected-counts 项缺少 split 名: {item!r}")
        try:
            counts[key] = int(value)
        except ValueError as exc:
            raise ConversionError(f"expected-counts 数量不是整数: {item!r}") from exc
    return counts or None


def resolve_spec(args: argparse.Namespace) -> DualModalLabelMeSpec:
    """把 CLI 参数解析为转换规格。"""
    if args.preset:
        spec = PRESETS[args.preset]
    elif args.classes:
        spec = DualModalLabelMeSpec(class_names=tuple(args.classes))
    else:
        raise ConversionError("未使用 --preset 时必须提供 --classes")

    overrides: dict[str, Any] = {}
    if args.classes:
        overrides["class_names"] = tuple(args.classes)
    if args.split_file:
        overrides["split_file"] = args.split_file
    if args.rgb_images:
        overrides["rgb_image_dir"] = args.rgb_images
    if args.x_images:
        overrides["x_image_dir"] = args.x_images
    if args.rgb_labels:
        overrides["rgb_label_dir"] = args.rgb_labels
    if args.x_labels:
        overrides["x_label_dir"] = args.x_labels
    if args.rgb_images_out:
        overrides["rgb_image_out"] = args.rgb_images_out
    if args.x_name:
        overrides["x_modality"] = args.x_name
        if args.x_images_out is None:
            overrides["x_image_out"] = f"images_{args.x_name}"
        if args.x_labels_out is None:
            overrides["x_label_out"] = f"label-{args.x_name}"
    if args.x_images_out:
        overrides["x_image_out"] = args.x_images_out
    if args.x_labels_out:
        overrides["x_label_out"] = args.x_labels_out
    if args.yaml_name:
        overrides["yaml_name"] = args.yaml_name
    if args.expected_counts:
        overrides["expected_counts"] = _parse_expected_counts(args.expected_counts)
    if overrides:
        spec = replace(spec, **overrides)
    return spec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。不提供交互式菜单。"""
    parser = argparse.ArgumentParser(
        prog="mm_obb_convert",
        description="将双模态 LabelMe OBB 源数据转换为 YOLOMM 数据集。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m ultralytics.tools.mm_obb_convert --preset escvehicle "
            "--source /data/ESCVehicle --output /data/ESCVehicle_yolomm\n"
            "  python -m ultralytics.tools.mm_obb_convert --source /data/raw --output /data/yolomm "
            "--classes car suv truck --x-name ir --no-enforce-counts\n"
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="使用内置数据集规格")
    parser.add_argument("--source", type=Path, required=True, help="源数据集根目录")
    parser.add_argument("--output", type=Path, required=True, help="目标 YOLOMM 根目录（必须不存在）")
    parser.add_argument("--classes", nargs="+", default=None, help="类别名列表；未使用 --preset 时必填")
    parser.add_argument("--split-file", default=None, help="相对源目录的 split JSON，默认 data.json")
    parser.add_argument("--rgb-images", default=None, help="RGB 图像相对目录")
    parser.add_argument("--x-images", default=None, help="X 模态图像相对目录")
    parser.add_argument("--rgb-labels", default=None, help="RGB LabelMe JSON 相对目录")
    parser.add_argument("--x-labels", default=None, help="X 模态 LabelMe JSON 相对目录")
    parser.add_argument("--rgb-images-out", default=None, help="输出 RGB 图像目录名，默认 images")
    parser.add_argument("--x-images-out", default=None, help="输出 X 图像目录名，默认 images_<x-name>")
    parser.add_argument("--x-labels-out", default=None, help="输出 X 标签目录名，默认 label-<x-name>")
    parser.add_argument("--x-name", default=None, help="X 模态键，写入 yaml 的 modality / 默认输出目录名")
    parser.add_argument("--yaml-name", default=None, help="输出 yaml 文件名，默认 data.yaml")
    parser.add_argument("--expected-counts", default=None, help="split 数量约束，如 train=6983,val=507,test=3237")
    parser.add_argument(
        "--no-enforce-counts",
        action="store_true",
        help="不校验 split 数量（即使 preset 或 --expected-counts 提供了约束）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """执行转换并打印核心统计。"""
    args = parse_args(argv)
    spec = resolve_spec(args)
    enforce_counts = spec.expected_counts is not None and not args.no_enforce_counts
    report = build_dataset(args.source, args.output, spec, enforce_counts=enforce_counts)
    summary = {
        "output_root": report["output_root"],
        "yaml": spec.yaml_name,
        "split_counts": report["split_counts"],
        "labels": {
            spec.rgb_stats_key: report["labels"][spec.rgb_stats_key]["written_objects"],
            spec.x_stats_key: report["labels"][spec.x_stats_key]["written_objects"],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
