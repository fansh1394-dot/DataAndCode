#!/usr/bin/env python3
"""多模态蒸馏配置 TUI 工具.

基于 rich 的轻量终端交互工具，引导用户配置 YOLOMM/RTDETRMM 的蒸馏 YAML。
独立脚本，不修改 ultralytics 包本身。

用法:
    python tools/distill_config.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# 确保 ultralytics 包可 import（脚本可能从项目根目录外运行）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

console = Console()

# 交互循环最大重试次数
_MAX_RETRIES = 10

# ultralytics 模型配置搜索根目录（rglob 递归覆盖所有子目录：mm/rtmm/11/v8/rt-detr 等）
_CFG_MODELS_ROOT = Path(_PROJECT_ROOT) / "ultralytics" / "cfg" / "models"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerInfo:
    """单个模型层的结构信息."""

    index: int
    name: str  # 模块类名
    out_channels: int  # 输出通道数, -1 表示未知
    input_source: Optional[str] = None  # 'RGB' / 'X' / 'Dual' / None
    f: object = None  # 连接来源 (-1, list, int)


@dataclass
class TeacherEntry:
    """用户交互中收集的教师信息."""

    name: str
    role: str  # rgb / x / dual
    weights: str  # .pt 路径
    yaml_path: Optional[str] = None
    layers: List[LayerInfo] = field(default_factory=list)


@dataclass
class FeatureMapping:
    """一条特征层映射."""

    teacher_name: str
    teacher_layer: int | tuple  # int 或 (start, end)
    student_layer: int | tuple  # int 或 (start, end)


# ---------------------------------------------------------------------------
# Task 1: 模型层结构解析
# ---------------------------------------------------------------------------


def _get_out_channels_from_state_dict(state_dict: dict, layer_idx: int) -> int:
    """从 state_dict 中推断某层的输出通道数."""
    prefix = f"model.{layer_idx}."
    # 优先找 conv.weight / weight
    for suffix in ("conv.weight", "weight"):
        key = prefix + suffix
        if key in state_dict:
            return state_dict[key].shape[0]
    # 兜底: 遍历该层所有参数, 取第一个 4D tensor 的 shape[0]
    for k, v in state_dict.items():
        if k.startswith(prefix) and v.dim() == 4:
            return v.shape[0]
    return -1


def _get_out_channels_from_module(module, state_dict: dict, layer_idx: int) -> int:
    """尝试多种方式获取层输出通道数."""
    # 方式1: 直接属性
    if hasattr(module, "out_channels"):
        return module.out_channels
    # 方式2: conv 子模块
    if hasattr(module, "conv") and hasattr(module.conv, "out_channels"):
        return module.conv.out_channels
    # 方式3: cv2 (C2f 等模块)
    if hasattr(module, "cv2") and hasattr(module.cv2, "conv"):
        return module.cv2.conv.out_channels
    # 方式4: 从 state_dict 推断
    return _get_out_channels_from_state_dict(state_dict, layer_idx)


def _search_yaml_in_cfg(name: str) -> Optional[Path]:
    """在 ultralytics/cfg/models 目录下递归搜索 YAML 文件.

    Args:
        name: YAML 文件名（如 'yolo11n-mm-mid.yaml'）或不带后缀的名称.

    Returns:
        找到的文件路径，或 None.
    """
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"
    if not _CFG_MODELS_ROOT.is_dir():
        return None
    unique = list(_CFG_MODELS_ROOT.rglob(name))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    # 多个匹配，让用户选择
    console.print(f"[yellow]找到 {len(unique)} 个匹配:[/yellow]")
    labels = [str(m.relative_to(Path(_PROJECT_ROOT))) for m in unique]
    idx = _select_index_from_list(labels, "选择配置文件")
    return unique[idx]


def _resolve_model_path(raw_path: str) -> Optional[str]:
    """解析模型路径：支持完整路径、相对路径、或仅 YAML 名称自动查找.

    Returns:
        解析后的实际文件路径，或 None（未找到）.
    """
    p = Path(raw_path)
    # 完整路径或相对路径直接存在
    if p.is_file():
        return str(p)
    # 仅文件名，尝试在 cfg/models 中搜索
    if not p.parent.parts or str(p.parent) == ".":
        found = _search_yaml_in_cfg(raw_path)
        if found:
            console.print(f"[green]自动定位: {found}[/green]")
            return str(found)
    return None


def _select_index_from_list(options: list[str], prompt_text: str) -> int:
    """展示带序号的选项列表，返回用户选择的索引.

    支持输入序号或完整名称.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("No.", justify="right", style="cyan", width=4)
    table.add_column("Option", style="white")
    for i, opt in enumerate(options):
        table.add_row(str(i + 1), opt)
    console.print(table)

    while True:
        raw = Prompt.ask(f"{prompt_text} (序号或名称)")
        raw = raw.strip()
        # 尝试序号
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
            console.print(f"[red]序号超出范围 [1, {len(options)}][/red]")
            continue
        except ValueError:
            pass
        # 尝试名称匹配
        for i, opt in enumerate(options):
            if raw == opt or raw in opt:
                return i
        console.print(f"[red]未匹配到选项: '{raw}'[/red]")


def _detect_and_select_scale(yaml_path: str) -> Optional[str]:
    """检测 YAML 是否有 scales 字段，如有则让用户选择 scale.

    Returns:
        选择的 scale 字符串（如 'n'），或 None（无 scales）.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None

    scales = cfg.get("scales")
    if not scales or not isinstance(scales, dict):
        return None

    scale_keys = list(scales.keys())
    console.print(f"\n[cyan]检测到模型 scale 选项:[/cyan]")
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("No.", justify="right", style="cyan", width=4)
    table.add_column("Scale", style="white", width=8)
    table.add_column("Depth", justify="right", style="magenta", width=8)
    table.add_column("Width", justify="right", style="magenta", width=8)
    table.add_column("Max Ch", justify="right", style="magenta", width=8)
    for i, k in enumerate(scale_keys):
        v = scales[k]
        table.add_row(str(i + 1), k, str(v[0]), str(v[1]), str(v[2]))
    console.print(table)

    while True:
        raw = Prompt.ask("选择 scale (序号或名称)", default=scale_keys[0])
        raw = raw.strip()
        # 序号
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(scale_keys):
                return scale_keys[idx]
        except ValueError:
            pass
        # 名称
        if raw in scale_keys:
            return raw
        console.print(f"[red]无效选择: '{raw}'，可选: {', '.join(scale_keys)}[/red]")


def parse_model_layers(path: str, scale: str | None = None) -> tuple[list[LayerInfo], str]:
    """解析模型层结构.

    Args:
        path: .pt 权重或 .yaml 配置文件路径.
        scale: 模型缩放级别（如 'n'/'s'/'m'/'l'/'x'），仅对 YAML 有效.

    Returns:
        (layers, family) -- 层信息列表和模型家族标识('yolomm'/'rtdetrmm'/'unknown').
    """
    p = Path(path)
    if not p.is_file():
        console.print(f"[red]文件不存在: {path}[/red]")
        return [], "unknown"

    if p.suffix == ".pt":
        return _parse_from_pt(p)
    elif p.suffix in (".yaml", ".yml"):
        return _parse_from_yaml(p, scale=scale)
    else:
        console.print(f"[red]不支持的文件格式: {p.suffix}，需要 .pt 或 .yaml/.yml[/red]")
        return [], "unknown"


def _parse_from_pt(pt_path: Path) -> tuple[list[LayerInfo], str]:
    """从 .pt 权重文件解析层结构."""
    console.print(
        f"[dim]加载权重: {pt_path.name} "
        f"(pickle 反序列化，请确保文件来源可信)[/dim]"
    )
    try:
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        console.print(f"[red]加载权重失败: {e}[/red]")
        return [], "unknown"

    model = ckpt.get("model", ckpt.get("ema", None))
    if model is None:
        console.print("[red]权重文件中未找到模型对象[/red]")
        return [], "unknown"

    # 兼容 FP16 模型
    if hasattr(model, "float"):
        model = model.float()

    family = _detect_family(model)
    layers = _extract_layers(model, model.state_dict() if hasattr(model, "state_dict") else {})
    return layers, family


def _detect_family_from_yaml(cfg: dict, yaml_path: Path) -> str:
    """从 YAML 内容和文件名检测模型家族."""
    # 优先检查 YAML 内容中的 RTDETR 特征
    head_layers = cfg.get("head", [])
    for layer_def in head_layers:
        if isinstance(layer_def, list) and len(layer_def) >= 3:
            module_name = str(layer_def[2])
            if "RTDETRDecoder" in module_name or "AIFI" in module_name:
                return "rtdetrmm"
    # 兜底: 文件名检测
    yaml_str = str(yaml_path).lower()
    if "rtdetr" in yaml_str:
        return "rtdetrmm"
    return "yolomm"


def _parse_from_yaml(yaml_path: Path, scale: str | None = None) -> tuple[list[LayerInfo], str]:
    """从 .yaml 配置文件构建模型并解析层结构."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    family = _detect_family_from_yaml(cfg, yaml_path)
    model = None

    # 构建 overrides 传入 scale
    overrides = {}
    if scale and cfg.get("scales"):
        overrides["scale"] = scale
        console.print(f"[dim]使用 scale: {scale}[/dim]")

    # 尝试通过模型类构建
    if family == "rtdetrmm":
        try:
            from ultralytics import RTDETRMM
            m = RTDETRMM(str(yaml_path))
            if overrides:
                m.overrides.update(overrides)
            model = m.model
        except Exception as e:
            console.print(f"[yellow]通过 RTDETRMM 构建失败: {e}，尝试通用方式...[/yellow]")
    else:
        try:
            from ultralytics import YOLOMM
            m = YOLOMM(str(yaml_path))
            if overrides:
                m.overrides.update(overrides)
            model = m.model
        except Exception as e:
            console.print(f"[yellow]通过 YOLOMM 构建失败: {e}，尝试通用方式...[/yellow]")

    if model is None:
        # 兜底: 直接用 parse_model
        try:
            from ultralytics.nn.tasks import parse_model
            ch = cfg.get("ch", 3)
            if cfg.get("multimodal"):
                ch = 6
            dataset_config = {}
            if "Xch" in cfg:
                dataset_config["Xch"] = cfg["Xch"]
            # 注入 scale 到配置字典
            if scale and cfg.get("scales"):
                cfg["scale"] = scale
            nn_model, _ = parse_model(
                cfg, ch, verbose=False,
                dataset_config=dataset_config if dataset_config else None,
            )
            layers = _extract_layers_from_sequential(nn_model)
            return layers, family
        except Exception as e:
            console.print(f"[red]无法从 YAML 构建模型: {e}[/red]")
            return [], family

    sd = model.state_dict() if hasattr(model, "state_dict") else {}
    layers = _extract_layers(model, sd)
    return layers, family


def _detect_family(model) -> str:
    """检测模型家族."""
    cls_name = type(model).__name__
    if "RTDETR" in cls_name:
        return "rtdetrmm"
    # 检查是否有多模态路由属性
    model_seq = getattr(model, "model", None)
    if model_seq is not None:
        has_mm = any(hasattr(m, "_mm_input_source") for m in model_seq)
        if not has_mm:
            console.print("[yellow]该模型未检测到多模态路由属性，可能不是 MM 变体[/yellow]")
    return "yolomm"


def _extract_layers(model, state_dict: dict) -> list[LayerInfo]:
    """从模型对象提取层信息列表."""
    model_seq = getattr(model, "model", None)
    if model_seq is None:
        console.print("[red]模型缺少 model 属性，无法遍历层[/red]")
        return []
    return _extract_layers_from_sequential(model_seq, state_dict)


def _extract_layers_from_sequential(model_seq, state_dict: dict | None = None) -> list[LayerInfo]:
    """从 nn.Sequential 提取层信息列表（兜底路径直接使用）."""
    if state_dict is None:
        state_dict = model_seq.state_dict() if hasattr(model_seq, "state_dict") else {}
    layers = []
    for idx, m in enumerate(model_seq):
        name = type(m).__name__
        out_ch = _get_out_channels_from_module(m, state_dict, idx)
        input_source = getattr(m, "_mm_input_source", None)
        f = getattr(m, "f", None)
        layers.append(LayerInfo(
            index=idx,
            name=name,
            out_channels=out_ch,
            input_source=input_source,
            f=f,
        ))
    return layers


# ---------------------------------------------------------------------------
# Task 2: TUI 层结构展示
# ---------------------------------------------------------------------------

# 模态路由颜色映射
_SOURCE_STYLE = {
    "RGB": "green",
    "X": "blue",
    "Dual": "yellow",
}


def display_layer_table(layers: list[LayerInfo], title: str):
    """以 rich Table 展示模型层结构."""
    table = Table(title=title, show_lines=False)
    table.add_column("Index", justify="right", style="cyan", width=6)
    table.add_column("Module", style="white", min_width=20)
    table.add_column("Out Ch", justify="right", style="magenta", width=8)
    table.add_column("Input Source", justify="center", width=14)
    table.add_column("From", justify="center", width=8)

    for layer in layers:
        # Input Source 样式
        if layer.input_source:
            style = _SOURCE_STYLE.get(layer.input_source, "white")
            source_str = f"[{style}]{layer.input_source}[/{style}]"
        else:
            source_str = "-"

        # 输出通道
        ch_str = str(layer.out_channels) if layer.out_channels > 0 else "?"

        # From 连接
        if isinstance(layer.f, list):
            f_str = f"[dim]{layer.f}[/dim]"
        elif layer.f is not None:
            f_str = str(layer.f)
        else:
            f_str = "-"

        table.add_row(str(layer.index), layer.name, ch_str, source_str, f_str)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Task 4: 通道对齐校验
# ---------------------------------------------------------------------------


def _resolve_layer_indices(spec) -> list[int]:
    """将层号(int 或 tuple)展开为索引列表."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, tuple) and len(spec) == 2:
        return list(range(spec[0], spec[1] + 1))
    raise TypeError(f"不支持的层号类型: {type(spec).__name__}({spec!r})，预期 int 或 (start, end)")


def validate_channel_alignment(
    teacher_spec,
    student_spec,
    teacher_layers: list[LayerInfo],
    student_layers: list[LayerInfo],
) -> list[str]:
    """校验教师-学生层映射的通道对齐.

    Returns:
        错误信息列表, 空列表表示通过.
    """
    t_indices = _resolve_layer_indices(teacher_spec)
    s_indices = _resolve_layer_indices(student_spec)

    errors = []

    if len(t_indices) != len(s_indices):
        errors.append(
            f"范围长度不匹配: 教师 {len(t_indices)} 层 vs 学生 {len(s_indices)} 层"
        )
        return errors

    t_map = {l.index: l for l in teacher_layers}
    s_map = {l.index: l for l in student_layers}

    for t_idx, s_idx in zip(t_indices, s_indices):
        t_layer = t_map.get(t_idx)
        s_layer = s_map.get(s_idx)

        if t_layer is None:
            errors.append(f"教师层 {t_idx} 不存在")
            continue
        if s_layer is None:
            errors.append(f"学生层 {s_idx} 不存在")
            continue

        t_ch = t_layer.out_channels
        s_ch = s_layer.out_channels

        if t_ch <= 0 or s_ch <= 0:
            errors.append(
                f"层 T{t_idx}({t_ch}ch) -> S{s_idx}({s_ch}ch): 无法确定通道数，请手动确认"
            )
        elif t_ch != s_ch:
            errors.append(
                f"通道不匹配: 教师层 {t_idx} ({t_ch}ch) -> 学生层 {s_idx} ({s_ch}ch)"
            )

    return errors


# ---------------------------------------------------------------------------
# Task 5: YAML 生成与合并
# ---------------------------------------------------------------------------


def _layer_spec_to_yaml(spec) -> int | list:
    """将层号转为 YAML 可序列化格式."""
    if isinstance(spec, int):
        return spec
    if isinstance(spec, (tuple, list)):
        return [int(x) for x in spec]
    return int(spec)


def build_yaml_dict(
    teachers: list[TeacherEntry],
    mode: str,
    feature_mappings: list[FeatureMapping],
    output_teachers: list[str],
) -> dict:
    """构建蒸馏 YAML 字典."""
    data = {"version": 1}

    # teachers
    data["teachers"] = []
    for t in teachers:
        entry = {"name": t.name, "role": t.role, "weights": t.weights}
        if t.yaml_path:
            entry["yaml"] = t.yaml_path
        data["teachers"].append(entry)

    # mappings
    mappings = {}

    if mode in ("feature", "both") and feature_mappings:
        feat_list = []
        for fm in feature_mappings:
            entry = {
                "teacher": fm.teacher_name,
                "teacher_layer": _layer_spec_to_yaml(fm.teacher_layer),
                "student_layer": _layer_spec_to_yaml(fm.student_layer),
            }
            feat_list.append(entry)
        mappings["feature"] = feat_list

    if mode in ("output", "both") and output_teachers:
        mappings["output"] = [{"teacher": name} for name in output_teachers]

    if mappings:
        data["mappings"] = mappings

    return data


def merge_yaml(existing: dict, new: dict) -> dict:
    """合并现有配置和新配置."""
    merged = dict(existing)
    merged["version"] = new.get("version", existing.get("version", 1))

    # 合并教师
    existing_teachers = {t["name"]: t for t in existing.get("teachers", [])}
    for t in new.get("teachers", []):
        if t["name"] in existing_teachers:
            console.print(f"[yellow]教师 '{t['name']}' 已存在，跳过[/yellow]")
        else:
            existing_teachers[t["name"]] = t
    merged["teachers"] = list(existing_teachers.values())

    # 合并 mappings
    existing_mappings = existing.get("mappings", {})
    new_mappings = new.get("mappings", {})

    merged_mappings = {}

    # feature (按 teacher+teacher_layer+student_layer 去重)
    existing_feat = list(existing_mappings.get("feature", []))
    existing_feat_keys = {
        (f.get("teacher"), str(f.get("teacher_layer")), str(f.get("student_layer")))
        for f in existing_feat
    }
    for f in new_mappings.get("feature", []):
        key = (f.get("teacher"), str(f.get("teacher_layer")), str(f.get("student_layer")))
        if key not in existing_feat_keys:
            existing_feat.append(f)
            existing_feat_keys.add(key)
        else:
            console.print(f"[yellow]特征映射 {key} 已存在，跳过[/yellow]")
    if existing_feat:
        merged_mappings["feature"] = existing_feat

    # output
    existing_out = list(existing_mappings.get("output", []))
    new_out = new_mappings.get("output", [])
    existing_out_names = {o["teacher"] for o in existing_out}
    for o in new_out:
        if o["teacher"] not in existing_out_names:
            existing_out.append(o)
    if existing_out:
        merged_mappings["output"] = existing_out

    if merged_mappings:
        merged["mappings"] = merged_mappings

    return merged


def write_yaml(data: dict, output_path: str):
    """写入 YAML 文件并校验."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    console.print(f"[green]已写入: {p}[/green]")

    # 回读校验
    try:
        from ultralytics.nn.mm.distill.schema import load_distill_config
        load_distill_config(str(p))
        console.print("[green]校验通过: YAML 符合 DistillConfig schema[/green]")
    except Exception as e:
        console.print(f"[red]校验失败: {e}[/red]")
        console.print("[yellow]请检查生成的 YAML 是否符合 schema 要求[/yellow]")


# ---------------------------------------------------------------------------
# Task 3: 交互流程主逻辑
# ---------------------------------------------------------------------------


def _parse_layer_input(raw: str) -> int | tuple:
    """解析用户输入的层号: '6' -> 6, '3-6' -> (3, 6)."""
    raw = raw.strip()
    # 先尝试直接解析为整数（处理负数等情况）
    try:
        val = int(raw)
    except ValueError:
        pass  # 不是纯整数，继续尝试范围解析
    else:
        if val < 0:
            raise ValueError(f"层号不支持负数索引: {val}，请使用 0 或正整数")
        return val
    # 范围格式: '3-6'
    if "-" in raw:
        parts = raw.split("-", 1)
        try:
            start, end = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            raise ValueError(f"无效的层范围格式: '{raw}'，应为 '起始-结束' 如 '3-6'")
        if start < 0 or end < 0:
            raise ValueError(f"层号不支持负数索引: [{start}, {end}]")
        if start > end:
            raise ValueError(f"范围起始 {start} 大于结束 {end}")
        return (start, end)
    raise ValueError(f"无效的层号: '{raw}'，应为整数或范围如 '3-6'")


def _validate_layer_exists(spec, layers: list[LayerInfo], label: str):
    """验证层号/范围在模型中存在."""
    indices = _resolve_layer_indices(spec)
    max_idx = max(l.index for l in layers) if layers else -1
    for idx in indices:
        if idx < 0 or idx > max_idx:
            raise ValueError(f"{label}层 {idx} 超出范围 [0, {max_idx}]")


def _step_load_existing() -> Optional[dict]:
    """Step 0: 询问是否加载现有蒸馏 YAML."""
    console.print(Panel("多模态蒸馏配置工具", style="bold cyan"))
    console.print()

    if Confirm.ask("是否加载现有蒸馏 YAML 进行编辑？", default=False):
        path = Prompt.ask("现有蒸馏 YAML 路径")
        p = Path(path)
        if not p.is_file():
            console.print(f"[red]文件不存在: {path}[/red]")
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            console.print(f"[green]已加载现有配置: {p}[/green]")
            # schema 校验
            try:
                from ultralytics.nn.mm.distill.schema import load_distill_config
                load_distill_config(str(p))
                console.print("[green]现有配置 schema 校验通过[/green]")
            except Exception as e:
                console.print(f"[yellow]现有配置 schema 校验警告: {e}[/yellow]")
                console.print("[dim]将继续加载，但输出时可能需要修正[/dim]")
            # 展示现有配置摘要
            teachers = data.get("teachers", [])
            mappings = data.get("mappings", {})
            console.print(f"  教师数量: {len(teachers)}")
            console.print(f"  特征映射: {len(mappings.get('feature', []))} 条")
            console.print(f"  输出蒸馏: {len(mappings.get('output', []))} 条")
            console.print()
            return data
        except Exception as e:
            console.print(f"[red]加载失败: {e}[/red]")
            return None
    return None


def _step_student_model() -> tuple[list[LayerInfo], str]:
    """Step 1: 学生模型输入与解析."""
    console.print(Panel("[bold]Step 1: 学生模型[/bold]", style="cyan"))
    console.print("[dim]支持: 完整路径、相对路径、或 YAML 名称自动查找 (如 yolo11n-mm-mid.yaml)[/dim]")
    for attempt in range(_MAX_RETRIES):
        raw_path = Prompt.ask("学生模型路径 (.pt/.yaml，或 YAML 名称，输入 q 退出)")
        if raw_path.strip().lower() == "q":
            raise KeyboardInterrupt
        # 解析路径（支持自动查找）
        resolved = _resolve_model_path(raw_path)
        if resolved is None:
            console.print(f"[red]未找到: {raw_path}[/red]")
            console.print(f"[yellow]请重新输入 (剩余 {_MAX_RETRIES - attempt - 1} 次)[/yellow]")
            continue
        # 检测 scale
        scale = None
        if resolved.endswith((".yaml", ".yml")):
            scale = _detect_and_select_scale(resolved)
        layers, family = parse_model_layers(resolved, scale=scale)
        if layers:
            scale_label = f", scale={scale}" if scale else ""
            display_layer_table(layers, f"学生模型层结构 ({family}{scale_label}) - {Path(resolved).name}")
            return layers, family
        console.print(f"[yellow]请重新输入有效路径 (剩余 {_MAX_RETRIES - attempt - 1} 次)[/yellow]")
    console.print("[red]超过最大重试次数，退出[/red]")
    raise SystemExit(1)


def _step_teachers(student_family: str) -> list[TeacherEntry]:
    """Step 2: 教师模型输入（支持多个）."""
    console.print(Panel("[bold]Step 2: 教师模型[/bold]", style="cyan"))
    teachers: list[TeacherEntry] = []

    while True:
        console.print(f"[dim]当前已添加 {len(teachers)} 个教师[/dim]")

        # 权重路径
        weights = Prompt.ask("教师权重路径 (.pt)")
        if not weights.endswith(".pt"):
            console.print("[red]教师权重必须是 .pt 文件[/red]")
            continue
        if not Path(weights).is_file():
            console.print(f"[red]文件不存在: {weights}[/red]")
            continue

        # 角色（支持序号选择）
        role_options = ["rgb", "x", "dual"]
        console.print("[dim]教师角色:[/dim]")
        for i, r in enumerate(role_options):
            console.print(f"  [cyan]{i+1}[/cyan] {r}")
        while True:
            role_raw = Prompt.ask("教师角色 (序号或名称)", default="1")
            try:
                idx = int(role_raw) - 1
                if 0 <= idx < len(role_options):
                    role = role_options[idx]
                    break
            except ValueError:
                pass
            if role_raw in role_options:
                role = role_raw
                break
            console.print(f"[red]无效选择: '{role_raw}'[/red]")

        # 名称
        default_name = f"teacher_{role}" if not teachers else f"teacher_{role}_{len(teachers)+1}"
        name = Prompt.ask("教师名称", default=default_name)

        # 检查名称唯一
        if any(t.name == name for t in teachers):
            console.print(f"[red]教师名称 '{name}' 已存在，请使用其他名称[/red]")
            continue

        # 可选 YAML（支持自动查找）
        yaml_path = None
        if Confirm.ask("是否提供辅助 YAML？", default=False):
            yaml_raw = Prompt.ask("辅助 YAML 路径 (或 YAML 名称)")
            resolved_yaml = _resolve_model_path(yaml_raw)
            if resolved_yaml:
                yaml_path = resolved_yaml
            else:
                console.print(f"[yellow]未找到: {yaml_raw}，已忽略[/yellow]")

        # 解析层结构
        layers, family = parse_model_layers(weights)
        if layers:
            display_layer_table(layers, f"教师 '{name}' ({role}) 层结构 - {Path(weights).name}")

        # 家族兼容性校验
        if family != "unknown" and family != student_family:
            console.print(
                f"[yellow]教师家族 ({family}) 与学生家族 ({student_family}) 不一致，"
                f"蒸馏训练时可能报错[/yellow]"
            )

        teachers.append(TeacherEntry(
            name=name, role=role, weights=weights,
            yaml_path=yaml_path, layers=layers,
        ))

        if not Confirm.ask("添加更多教师？", default=False):
            break

    return teachers


def _step_distill_mode() -> str:
    """Step 3: 蒸馏模式选择."""
    console.print(Panel("[bold]Step 3: 蒸馏模式[/bold]", style="cyan"))
    mode_options = [
        ("output", "输出蒸馏 (检测头输出级别)"),
        ("feature", "特征蒸馏 (中间层特征对齐)"),
        ("both", "同时使用两种蒸馏"),
    ]
    for i, (key, desc) in enumerate(mode_options):
        console.print(f"  [cyan]{i+1}[/cyan] [bold]{key}[/bold] - {desc}")
    console.print()

    while True:
        raw = Prompt.ask("选择蒸馏模式 (序号或名称)", default="3")
        raw = raw.strip()
        # 序号
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(mode_options):
                return mode_options[idx][0]
        except ValueError:
            pass
        # 名称
        for key, _ in mode_options:
            if raw == key:
                return key
        console.print(f"[red]无效选择: '{raw}'[/red]")


def _step_feature_mappings(
    teachers: list[TeacherEntry],
    student_layers: list[LayerInfo],
) -> list[FeatureMapping]:
    """Step 4: 特征蒸馏层映射配置."""
    console.print(Panel("[bold]Step 4: 特征层映射[/bold]", style="cyan"))
    mappings: list[FeatureMapping] = []

    for teacher in teachers:
        console.print(f"\n[bold]配置教师 '{teacher.name}' ({teacher.role}) 的层映射:[/bold]")

        if not teacher.layers:
            console.print(f"[red]教师 '{teacher.name}' 无层信息，跳过[/red]")
            continue

        while True:
            # 展示教师层表
            display_layer_table(teacher.layers, f"教师 '{teacher.name}' 层结构")

            # 输入教师层号
            while True:
                raw = Prompt.ask("教师层号 (单层如 '6' 或范围如 '3-6')")
                try:
                    t_spec = _parse_layer_input(raw)
                    _validate_layer_exists(t_spec, teacher.layers, "教师")
                    break
                except ValueError as e:
                    console.print(f"[red]{e}[/red]")

            # 展示学生层表
            display_layer_table(student_layers, "学生模型层结构")

            # 输入学生层号
            while True:
                raw = Prompt.ask("学生层号 (单层如 '6' 或范围如 '3-6')")
                try:
                    s_spec = _parse_layer_input(raw)
                    _validate_layer_exists(s_spec, student_layers, "学生")
                    break
                except ValueError as e:
                    console.print(f"[red]{e}[/red]")

            # 通道对齐校验
            errors = validate_channel_alignment(t_spec, s_spec, teacher.layers, student_layers)
            if errors:
                console.print("[red]通道对齐校验失败:[/red]")
                for err in errors:
                    console.print(f"  [red]- {err}[/red]")
                if not Confirm.ask("仍然添加此映射？", default=False):
                    continue
            else:
                # 显示匹配结果
                t_indices = _resolve_layer_indices(t_spec)
                s_indices = _resolve_layer_indices(s_spec)
                t_map = {l.index: l for l in teacher.layers}
                s_map = {l.index: l for l in student_layers}
                for ti, si in zip(t_indices, s_indices):
                    tl = t_map.get(ti)
                    sl = s_map.get(si)
                    t_ch = tl.out_channels if tl else "?"
                    s_ch = sl.out_channels if sl else "?"
                    console.print(
                        f"  [green]T{ti}({tl.name if tl else '?'}, {t_ch}ch) "
                        f"-> S{si}({sl.name if sl else '?'}, {s_ch}ch)[/green]"
                    )

            mappings.append(FeatureMapping(
                teacher_name=teacher.name,
                teacher_layer=t_spec,
                student_layer=s_spec,
            ))
            console.print(f"[green]已添加映射 (当前共 {len(mappings)} 条)[/green]")

            if not Confirm.ask(f"为教师 '{teacher.name}' 添加更多映射？", default=False):
                break

    return mappings


def _step_output_teachers(teachers: list[TeacherEntry]) -> list[str]:
    """Step 5: 输出蒸馏教师选择."""
    console.print(Panel("[bold]Step 5: 输出蒸馏教师[/bold]", style="cyan"))
    console.print("选择参与输出蒸馏的教师:")

    selected = []
    for t in teachers:
        if Confirm.ask(f"  教师 '{t.name}' ({t.role}) 参与输出蒸馏？", default=True):
            selected.append(t.name)

    return selected


def _step_preview_and_write(
    data: dict,
    existing_data: Optional[dict],
):
    """Step 6: 预览与输出."""
    console.print(Panel("[bold]Step 6: 预览与输出[/bold]", style="cyan"))

    # 如果有现有配置则合并
    if existing_data:
        data = merge_yaml(existing_data, data)
        console.print("[dim]已与现有配置合并[/dim]")

    # 预览
    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    console.print(Panel(yaml_str, title="蒸馏配置预览", border_style="green"))

    # 输出路径
    default_output = "distill.yaml"
    output_path = Prompt.ask("输出文件路径", default=default_output)

    if Path(output_path).is_file():
        if not Confirm.ask(f"文件 '{output_path}' 已存在，覆盖？", default=False):
            output_path = Prompt.ask("请输入新的文件路径")

    write_yaml(data, output_path)


def main():
    """主入口."""
    console.print()
    console.print("[bold cyan]YOLOMM / RTDETRMM 多模态蒸馏配置工具[/bold cyan]")
    console.print("[dim]基于引导式交互生成蒸馏 YAML 配置[/dim]")
    console.print()

    try:
        # Step 0: 加载现有配置
        existing_data = _step_load_existing()

        # Step 1: 学生模型
        student_layers, student_family = _step_student_model()

        # Step 2: 教师模型
        teachers = _step_teachers(student_family)
        if not teachers:
            console.print("[red]未添加任何教师，退出[/red]")
            return

        # Step 3: 蒸馏模式
        mode = _step_distill_mode()

        # Step 4: 特征层映射 (feature / both)
        feature_mappings: list[FeatureMapping] = []
        if mode in ("feature", "both"):
            feature_mappings = _step_feature_mappings(teachers, student_layers)
            if not feature_mappings:
                console.print("[yellow]未配置任何特征映射[/yellow]")
                if mode == "feature":
                    console.print("[red]feature 模式至少需要一条特征映射，退出[/red]")
                    return

        # Step 5: 输出蒸馏 (output / both)
        output_teachers: list[str] = []
        if mode in ("output", "both"):
            output_teachers = _step_output_teachers(teachers)
            if not output_teachers:
                console.print("[yellow]未选择任何输出蒸馏教师[/yellow]")
                if mode == "output":
                    console.print("[red]output 模式至少需要一个教师，退出[/red]")
                    return

        # 构建 YAML
        data = build_yaml_dict(teachers, mode, feature_mappings, output_teachers)

        # Step 6: 预览与输出
        _step_preview_and_write(data, existing_data)

        console.print()
        console.print("[bold green]配置完成[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[dim]已取消[/dim]")
    except Exception as e:
        console.print(f"\n[red]发生错误: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
