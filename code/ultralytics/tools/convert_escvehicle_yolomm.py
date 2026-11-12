"""ESCVehicle → YOLOMM OBB 转换 CLI。

通用实现见 ``ultralytics.tools.mm_obb_convert``。本文件只提供 ESCVehicle 预设与兼容导出。

用法：
    /root/miniconda3/envs/yolon/bin/python ultralytics/tools/convert_escvehicle_yolomm.py \\
        --source /data/ESCVehicle --output /data/ESCVehicle_yolomm
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mm_obb_convert as _impl
else:
    from ultralytics.tools import mm_obb_convert as _impl

PRESETS = _impl.PRESETS
ConversionError = _impl.ConversionError
build_dual_modal_dataset = _impl.build_dataset
mm_obb_convert_main = _impl.main
validate_label_file = _impl.validate_label_file

ESCVehicleConversionError = ConversionError
ESCVEHICLE_SPEC = PRESETS["escvehicle"]
CLASS_NAMES = ESCVEHICLE_SPEC.class_names
OFFICIAL_SPLIT_COUNTS = dict(ESCVEHICLE_SPEC.expected_counts or {})
ALLOWED_SPLITS = ESCVEHICLE_SPEC.splits


def _validate_label_file(path: Path) -> int:
    """兼容分析脚本：按 ESCVehicle 类别数校验 YOLO OBB 标签。"""
    return validate_label_file(path, num_classes=len(CLASS_NAMES))


def build_dataset(source: Path, output: Path, *, enforce_official_counts: bool = True) -> dict:
    """按 ESCVehicle 预设转换并原子发布。"""
    return build_dual_modal_dataset(
        source,
        output,
        ESCVEHICLE_SPEC,
        enforce_counts=enforce_official_counts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """注入 ``--preset escvehicle`` 后转交给通用 CLI。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--preset" not in args:
        args = ["--preset", "escvehicle", *args]
    return mm_obb_convert_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
