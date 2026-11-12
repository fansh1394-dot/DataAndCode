#!/usr/bin/env python3
"""Manual minimum-environment checker for train/predict workflows.

Run from the repository root:
    python tools/check_min_env.py

Behavior:
1. Fail fast when Python < 3.10.
2. Fail fast when torch is missing or < 2.0.
3. Otherwise, check the minimal Python-package baseline for the train/predict
   workflows using a hybrid of:
   - curated static dependency probing for the targeted import paths
   - controlled subprocess dry-run probes for train/predict anchors
4. Print only missing package names plus a single pip install command.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Sequence

TOTAL_PROGRESS_STEPS = 8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_MIN = (3, 10)
TORCH_MIN = (2, 0)

PACKAGE_NAME_MAP = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "pytorch_grad_cam": "grad-cam",
}

# 版本约束：pip install 时需要明确的版本范围
# 格式: pip_package_name -> (min_version, max_version, reason)
# max_version 为 None 表示不设上限
VERSION_CONSTRAINTS = {
    "numpy": {
        "min": (1, 24, 0),
        "max": None,
        "reason": "numpy>=2.0 移除了 np.float_/np.int_ 等别名，部分依赖尚未适配",
    },
    "grad-cam": {
        "min": (1, 4, 0),
        "max": (2, 0, 0),
        "reason": "grad-cam>=2.0 可能重构 CAM 算法类 API（GradCAM/ScoreCAM 等）",
    },
}
PACKAGE_CHECKS_ONLY = {"torchvision"}
IGNORED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "copy",
    "dataclasses",
    "functools",
    "importlib",
    "json",
    "logging",
    "math",
    "numbers",
    "os",
    "pathlib",
    "platform",
    "random",
    "re",
    "subprocess",
    "sys",
    "threading",
    "time",
    "typing",
    "urllib",
    "warnings",
    "ultralytics",
}

TARGET_FILES = (
    Path("trainMM.py"),
    Path("train_full.py"),
    Path("predictMM.py"),
    Path("ultralytics/__init__.py"),
    Path("ultralytics/models/yolo/model.py"),
    Path("ultralytics/engine/model.py"),
    Path("ultralytics/tools/__init__.py"),
    Path("ultralytics/tools/mm_sampler.py"),
    Path("ultralytics/utils/__init__.py"),
    Path("ultralytics/utils/checks.py"),
    Path("ultralytics/models/yolo/multimodal/train.py"),
    Path("ultralytics/models/yolo/multimodal/visualize/heatmap.py"),
    Path("ultralytics/data/dataset.py"),
)

# 探测用的模型 YAML 候选路径（按顺序取第一个存在的）。
# 说明：cfg 目录经过重构，早期版本位于
#   cfg/models/mm/detect/yolo11/base/yolo11n-mm-mid.yaml
# 现位于 cfg/models/mm/ 下。此处列出多个候选以兼容两种布局，
# 避免因目录调整导致探针误报"文件不存在"。
PROBE_MODEL_CANDIDATES = (
    Path("ultralytics/cfg/models/mm/yolo11n-mm-mid.yaml"),
    Path("ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-mid.yaml"),
    Path("ultralytics/cfg/models/mm/detect/yolo11/extraction/yolo11n-mm-c3k2_pkimodule.yaml"),
)


def resolve_probe_model(project_root: Path) -> Path:
    """返回第一个存在的探针模型 YAML；都不存在时返回首个候选作为报错提示。"""
    for cand in PROBE_MODEL_CANDIDATES:
        if (project_root / cand).is_file():
            return cand
    return PROBE_MODEL_CANDIDATES[0]


# OpenMP 线程数护栏：部分镜像里 OMP_NUM_THREADS 被设为空串或非数字，
# 会让 libgomp 打印 "Invalid value for environment variable OMP_NUM_THREADS"。
# 这里在探测子进程环境里兜底修正，避免噪音干扰判断。
def _sanitize_omp_env(env: dict) -> None:
    """把非法/空的 OMP_NUM_THREADS 修正为合法值。"""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        val = env.get(key)
        if val is None:
            continue
        try:
            if int(str(val).strip()) > 0:
                continue
        except (ValueError, TypeError):
            pass
        env[key] = "8"


@dataclass(frozen=True)
class ProbeResult:
    """Result of a controlled dry-run probe."""

    name: str
    returncode: int
    output: str


PROBE_SNIPPETS = {
    "train": textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        root = Path({str(PROJECT_ROOT)!r})
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from ultralytics import YOLOMM

        model = YOLOMM(str(root / {str(resolve_probe_model(PROJECT_ROOT))!r}))
        _ = model.task_map["detect"]["trainer"]
        print("train-probe-ok")
        """
    ).strip(),
    "predict": textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        root = Path({str(PROJECT_ROOT)!r})
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from ultralytics import YOLOMM
        from ultralytics.tools import MultiModalSampler

        model = YOLOMM(str(root / {str(resolve_probe_model(PROJECT_ROOT))!r}))
        _ = model.task_map["detect"]["predictor"]
        _ = MultiModalSampler
        print('predict-probe-ok')
        """
    ).strip(),
    "vis": textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        root = Path({str(PROJECT_ROOT)!r})
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from ultralytics.models.yolo.multimodal.visualize.heatmap import HeatmapVisualizer
        _ = HeatmapVisualizer
        print('vis-probe-ok')
        """
    ).strip(),
}


def print_progress(step: int, message: str) -> None:
    """Print a concise progress line so users know the current execution step."""
    print(f"[步骤 {step}/{TOTAL_PROGRESS_STEPS}] {message}")


def parse_version_tuple(version: str) -> tuple[int, ...]:
    """Extract a numeric version tuple from an arbitrary version string."""
    numbers = re.findall(r"\d+", version)
    if not numbers:
        return (0, 0, 0)
    parsed = tuple(int(part) for part in numbers[:3])
    return parsed + (0,) * (3 - len(parsed))


def format_python_version(version_info: Sequence[int]) -> str:
    """Return a dotted Python version string."""
    return ".".join(str(part) for part in version_info[:3])


def check_python_gate(version_info: Sequence[int] | None = None) -> str | None:
    """Return an early-exit message when Python is below the minimum."""
    current = tuple(version_info or sys.version_info[:3])
    if current >= PYTHON_MIN:
        return None
    return (
        f"当前 Python 版本为 {format_python_version(current)}，本项目最小要求为 Python "
        f"{format_python_version(PYTHON_MIN)}。\n"
        f"请先升级 Python 到 {format_python_version(PYTHON_MIN)} 或更高版本，然后重新执行检查。"
    )


def check_torch_gate(torch_version: str | None = None, missing: bool = False) -> str | None:
    """Return an early-exit message when torch is missing or below the minimum."""
    if missing:
        return (
            f"当前环境缺少 torch，本项目最小要求为 torch>={format_python_version(TORCH_MIN)}。\n"
            f"请先安装或升级 torch>={format_python_version(TORCH_MIN)}，然后重新执行检查。"
        )
    if torch_version is None:
        return (
            f"无法识别 torch 版本，本项目最小要求为 torch>={format_python_version(TORCH_MIN)}。\n"
            f"请先安装或升级 torch>={format_python_version(TORCH_MIN)}，然后重新执行检查。"
        )
    if parse_version_tuple(torch_version) >= TORCH_MIN:
        return None
    return (
        f"当前 torch 版本为 {torch_version}，本项目最小要求为 "
        f"torch>={format_python_version(TORCH_MIN)}。\n"
        f"请先升级 torch>={format_python_version(TORCH_MIN)}，然后重新执行检查。"
    )


def import_torch_version() -> tuple[bool, str | None]:
    """Return torch availability and version string."""
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return False, None
    return True, getattr(torch, "__version__", None)


# RTX 50 系列 (Blackwell, sm_120) 最低要求
RTX50_BLACKWELL_TORCH_MIN = (2, 7, 0)
RTX50_BLACKWELL_CUDA_MIN = "12.8"
RTX50_KEYWORDS = ("RTX 50", "5090", "5080", "5070", "5060", "5050")


def check_rtx50_compat(torch_version: str | None) -> str | None:
    """检测 RTX 50 系显卡并校验 CUDA/PyTorch 兼容性。

    Returns:
        警告消息（str），或 None 表示无需警告。
    """
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None

    # 检测是否有 RTX 50 系列 GPU
    has_rtx50 = False
    gpu_names: list[str] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            gpu_names.append(name)
            if any(kw in name.upper() for kw in RTX50_KEYWORDS):
                has_rtx50 = True

    if not has_rtx50:
        return None

    # 检查 PyTorch 版本
    torch_ver = parse_version_tuple(torch_version or "0.0.0")
    cuda_ver = torch.version.cuda  # e.g. "12.8" or "11.8"
    torch_ok = torch_ver >= RTX50_BLACKWELL_TORCH_MIN
    cuda_ok = cuda_ver is not None and parse_version_tuple(cuda_ver) >= parse_version_tuple(RTX50_BLACKWELL_CUDA_MIN)

    if torch_ok and cuda_ok:
        return None

    lines = [
        f"检测到 NVIDIA RTX 50 系显卡 ({gpu_names[0]})，"
        f"Blackwell 架构 (sm_120) 需要以下最低版本：",
        f"",
        f"  当前 torch:   {torch_version or '未知'}   (要求 >= {'.'.join(str(x) for x in RTX50_BLACKWELL_TORCH_MIN)})",
        f"  当前 CUDA:    {cuda_ver or '未知'}   (要求 >= {RTX50_BLACKWELL_CUDA_MIN})",
        f"",
    ]

    if not torch_ok:
        lines.append(f"torch 版本不满足，RTX 50 系需要 PyTorch >= {'.'.join(str(x) for x in RTX50_BLACKWELL_TORCH_MIN)}。")

    if not cuda_ok:
        lines.append(f"CUDA 版本不满足，RTX 50 系需要 CUDA >= {RTX50_BLACKWELL_CUDA_MIN}。")

    lines.append("")
    lines.append("建议执行以下命令重新安装 PyTorch：")
    lines.append(f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    lines.append("")
    lines.append("注意：安装前请先卸载旧版本（pip uninstall torch torchvision torchaudio）。")

    return "\n".join(lines)


def extract_import_roots(source: str) -> set[str]:
    """Collect third-party import roots from a Python source file."""
    tree = ast.parse(source)
    roots: set[str] = set()
    # Only inspect module-level imports so optional/nested imports do not inflate the minimum baseline.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                roots.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if not node.module:
                continue
            root = node.module.split(".", 1)[0]
            roots.add(root)
    return roots


def collect_seed_modules(project_root: Path = PROJECT_ROOT) -> set[str]:
    """Collect a curated set of candidate third-party modules for train/predict coverage."""
    modules: set[str] = set(PACKAGE_CHECKS_ONLY)
    for relative_path in TARGET_FILES:
        file_path = project_root / relative_path
        if not file_path.is_file():
            continue
        roots = extract_import_roots(file_path.read_text(encoding="utf-8"))
        for root in roots:
            if root in IGNORED_IMPORT_ROOTS:
                continue
            if root in getattr(sys, "stdlib_module_names", set()):
                continue
            modules.add(root)
    modules.add("tqdm")
    modules.add("torchvision")
    modules.discard("torch")
    return modules


def normalize_package_name(module_name: str) -> str:
    """Convert import/module names to user-facing pip package names."""
    root = module_name.split(".", 1)[0]
    return PACKAGE_NAME_MAP.get(root, root)


def extract_missing_modules(text: str) -> set[str]:
    """Extract missing module/package names from probe output."""
    missing = set(re.findall(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", text))
    missing.update(re.findall(r"No package metadata was found for ([A-Za-z0-9_.-]+)", text))
    return {name.split(".", 1)[0] for name in missing}


def check_module_available(module_name: str) -> bool:
    """Return True when the target module/package is importable enough for this checker."""
    if module_name in PACKAGE_CHECKS_ONLY:
        try:
            importlib_metadata.version(module_name)
            return True
        except importlib_metadata.PackageNotFoundError:
            return False
        except Exception:
            return True

    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False
    except Exception:
        return True


def find_missing_seed_packages(seed_modules: Iterable[str]) -> set[str]:
    """Check curated candidate modules individually to accumulate all missing packages."""
    missing = set()
    for module_name in seed_modules:
        if not check_module_available(module_name):
            missing.add(normalize_package_name(module_name))
    return missing


def run_probe(name: str, snippet: str) -> ProbeResult:
    """Execute a controlled dry-run probe in a subprocess using the current interpreter."""
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(PROJECT_ROOT)
    # Force probe stdio to UTF-8 and decode defensively. On Chinese Windows,
    # subprocess text mode otherwise defaults to GBK/cp936, while torch and
    # other libraries may emit UTF-8 symbols in warnings/progress output. A
    # decode failure in the parent process would hide the real probe result.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    _sanitize_omp_env(env)
    completed = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return ProbeResult(
        name=name,
        returncode=completed.returncode,
        output=(completed.stdout or "") + (completed.stderr or ""),
    )


def find_missing_probe_packages(results: Iterable[ProbeResult]) -> set[str]:
    """Aggregate normalized missing-package names from probe results."""
    missing = set()
    for result in results:
        for module_name in extract_missing_modules(result.output):
            if module_name == "torch":
                continue
            missing.add(normalize_package_name(module_name))
    return missing


def summarize_probe_output(output: str) -> str:
    """Return the last non-empty output line for user-facing probe failure reporting."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "probe failed without output"


def find_unexpected_probe_failures(results: Iterable[ProbeResult]) -> dict[str, str]:
    """Report probe failures that are not explainable by missing Python packages."""
    failures: dict[str, str] = {}
    for result in results:
        if result.returncode == 0:
            continue
        if extract_missing_modules(result.output):
            continue
        failures[result.name] = summarize_probe_output(result.output)
    return failures


def _format_version_tuple(v: tuple[int, ...]) -> str:
    """Convert a version tuple to dotted string."""
    return ".".join(str(x) for x in v)


def build_pip_install_command(packages: Iterable[str]) -> str:
    """Render a pip install command with version constraints for sensitive packages."""
    normalized = sorted({package for package in packages if package})
    if not normalized:
        return ""
    parts: list[str] = []
    for pkg in normalized:
        constraint = VERSION_CONSTRAINTS.get(pkg)
        if constraint:
            spec = f"{pkg}>={_format_version_tuple(constraint['min'])}"
            if constraint["max"]:
                spec += f",<{_format_version_tuple(constraint['max'])}"
            parts.append(spec)
        else:
            parts.append(pkg)
    return "pip install " + " ".join(parts)


def print_missing_report(missing_packages: Sequence[str]) -> None:
    """Print the final missing-package report."""
    print("检测结果：当前环境还缺少以下 Python 库：")
    for package in missing_packages:
        print(f"- {package}")
    command = build_pip_install_command(missing_packages)
    if command:
        print("\n可执行以下命令安装（已附带版本约束）：")
        print(f"  {command}")
    # 输出版本约束说明
    constrained = [p for p in missing_packages if p in VERSION_CONSTRAINTS]
    if constrained:
        print("\n版本约束说明：")
        for pkg in constrained:
            c = VERSION_CONSTRAINTS[pkg]
            min_s = _format_version_tuple(c["min"])
            max_s = _format_version_tuple(c["max"]) if c["max"] else "无上限"
            print(f"  {pkg}: >={min_s}, <{max_s} — {c['reason']}")


def check_installed_version_constraints() -> list[str]:
    """检查已安装的敏感包版本是否超出约束范围，返回警告列表。"""
    warnings: list[str] = []
    for pkg, constraint in VERSION_CONSTRAINTS.items():
        try:
            installed = parse_version_tuple(importlib_metadata.version(pkg))
        except importlib_metadata.PackageNotFoundError:
            continue
        min_v = constraint["min"]
        max_v = constraint["max"]
        if installed < min_v:
            spec = f">={_format_version_tuple(min_v)}"
            if max_v:
                spec += f",<{_format_version_tuple(max_v)}"
            warnings.append(
                f"{pkg} 当前版本 {importlib_metadata.version(pkg)} 低于最低要求 "
                f"{_format_version_tuple(min_v)}，建议执行：pip install {pkg}{spec}"
            )
        elif max_v and installed >= max_v:
            warnings.append(
                f"{pkg} 当前版本 {importlib_metadata.version(pkg)} 超出兼容上限 "
                f"{_format_version_tuple(max_v)} — {constraint['reason']}。"
                f"建议降级：pip install \"{pkg}>={_format_version_tuple(min_v)},<{_format_version_tuple(max_v)}\""
            )
    return warnings


def main() -> int:
    """CLI entrypoint."""
    print_progress(1, "检查 Python 版本")
    python_message = check_python_gate()
    if python_message:
        print(python_message)
        return 1

    print_progress(2, "检查 torch 版本")
    torch_installed, torch_version = import_torch_version()
    torch_message = check_torch_gate(torch_version=torch_version, missing=not torch_installed)
    if torch_message:
        print(torch_message)
        return 1

    print_progress(3, "检查 RTX 50 系显卡兼容性")
    rtx50_warning = check_rtx50_compat(torch_version)
    if rtx50_warning:
        print(rtx50_warning)
        return 1

    print_progress(4, "收集最小依赖候选")
    seed_modules = collect_seed_modules()
    missing_packages = find_missing_seed_packages(seed_modules)

    print_progress(5, "执行 train 探针")
    train_result = run_probe("train", PROBE_SNIPPETS["train"])
    print_progress(6, "执行 predict 探针")
    predict_result = run_probe("predict", PROBE_SNIPPETS["predict"])
    print_progress(7, "执行 vis 探针")
    vis_result = run_probe("vis", PROBE_SNIPPETS["vis"])
    probe_results = [train_result, predict_result, vis_result]
    missing_packages.update(find_missing_probe_packages(probe_results))
    missing_packages.discard("torch")

    print_progress(8, "汇总检查结果")
    unexpected_failures = find_unexpected_probe_failures(probe_results)
    if unexpected_failures:
        print("环境检查未能完成：最小 dry-run 探针遇到非缺库错误。")
        for probe_name, summary in unexpected_failures.items():
            print(f"- {probe_name}: {summary}")
        print("请先处理这些非缺库问题，然后重新执行检查。")
        return 1

    if missing_packages:
        print_missing_report(sorted(missing_packages))
        return 1

    # 检查已安装的敏感包版本是否在约束范围内
    version_warnings = check_installed_version_constraints()

    print("检测结果：当前环境已满足本项目最小 train + predict + vis 基线所需的 Python 库要求。")

    if version_warnings:
        print()
        print("版本兼容性警告：")
        for w in version_warnings:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
