#!/usr/bin/env python3
"""
Preflight CLI for YOLOMM/RTDETRMM.

Standalone script that validates a YAML config can complete a full training
iteration using synthetic data.  No pip installation required.

Usage:
    python preflightMM.py yolo11n-mm-mid.yaml
    python preflightMM.py yolo11n-mm-mid.yaml --device 0 --iters 3
    python preflightMM.py rtdetr-r18-mm-mid.yaml --batch 4
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Preflight: validate YOLOMM/RTDETRMM YAML config with synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python preflightMM.py ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-mid.yaml\n"
            "  python preflightMM.py ultralytics/cfg/models/rtmm/detect/r18/base/rtdetr-r18-mm-mid.yaml --device 0\n"
            "  python preflightMM.py ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-mid.yaml --iters 3 --batch 4\n"
        ),
    )
    p.add_argument("model", type=str, help="Path to YAML config file (.yaml)")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu / 0 / 0,1 (default: cpu)")
    p.add_argument("--iters", type=int, default=1, help="Training iterations (default: 1)")
    p.add_argument("--batch", type=int, default=2, help="Synthetic batch size (default: 2)")
    p.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    p.add_argument("--task", type=str, default="detect", help="Task type (default: detect)")
    p.add_argument("--scale", type=str, default="", help="Model scale key, e.g. n/s/m/l/x (default: YAML first key)")
    p.add_argument("--Xch", type=int, default=3, help="X modality channels (default: 3)")
    p.add_argument("--x-modality", type=str, default="depth", help="X modality type (default: depth)")
    p.add_argument("--half", action="store_true", help="Use FP16")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output report as JSON")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        print(f"ERROR: file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Import here so argparse --help works even with broken env
    from ultralytics.engine.preflight import PreflightRunner

    runner = PreflightRunner(
        model_path,
        device=args.device,
        iters=args.iters,
        batch=args.batch,
        imgsz=args.imgsz,
        task=args.task,
        scale=args.scale,
        Xch=args.Xch,
        x_modality=args.x_modality,
        half=args.half,
        verbose=not args.quiet,
    )

    report = runner.run()

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, default=str))

    # Exit code: 0 = all passed, 1 = failure
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
