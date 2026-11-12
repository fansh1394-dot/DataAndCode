# -*- coding: utf-8 -*-
"""服务器训练前自检：权重盘点 + 数据配对校验。

在**服务器上**、训练之前运行本脚本，用于回答两个问题：
    1. 服务器上现有哪些权重？它们是单模态（COCO）还是本项目多模态权重？
    2. 数据集里 RGB / IR 是否真的严格同名配对？有没有错配、缺漏、损坏？

用法（在项目根目录执行）：
    python tools/preflight_train.py                       # 全量检查（配对抽样 2000）
    python tools/preflight_train.py --pair-samples 5000   # 加大配对抽检量
    python tools/preflight_train.py --data D:/uav_data/data.yaml
    python tools/preflight_train.py --skip-pairing        # 只查权重，跳过数据扫描

退出码：0 = 可训练；1 = 存在阻断问题（缺模态目录 / 配对率过低等）。

设计说明
--------
- 只读，不修改任何数据或权重。
- 采样检查先看"有没有"，再抽查"对不对"，避免 5.5 万对全量扫描耗时过长。
- 不导入 ultralytics 重型模块，保证在环境尚未装全时也能跑。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# 权重搜索根目录（服务器上的常见位置）
DEFAULT_WEIGHT_ROOTS = (
    "/root/autodl-tmp",
    "/root/code",
    "/root",
    "/home",
    "~/.cache/ultralytics",
    ".",  # 当前项目目录
)

# 判断是否为本项目多模态权重的元数据 key
MM_META_KEYS = (
    "multimodal_config",
    "modality",
    "x_modality",
    "x_modality_name",
    "single_modal",
    "prune_info",
)


def hr(title: str = "") -> None:
    """打印分隔线与小标题。"""
    if title:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    else:
        print("-" * 72)


# --------------------------------------------------------------------------- #
# 第一部分：权重盘点
# --------------------------------------------------------------------------- #

def find_weight_files(roots, max_per_root: int = 40, verbose: bool = True):
    """在候选根目录下搜索 .pt / .pth 权重文件。

    Args:
        roots: 待搜索的根目录序列。
        max_per_root: 每个根目录最多返回多少个，避免在超大目录里卡死。
        verbose: 是否打印搜索过程。

    Returns:
        去重排序后的 Path 列表。
    """
    found: set[Path] = set()
    skip_parts = {"site-packages", "dist-packages", "node_modules", ".git"}

    for root in roots:
        p = Path(root).expanduser()
        if not p.exists():
            if verbose:
                print(f"  [跳过] 不存在: {p}")
            continue
        if verbose:
            print(f"  [搜索] {p}")
        count = 0
        try:
            for dirpath, dirnames, filenames in os.walk(p):
                # 原地裁剪，避免进入无关的大目录
                dirnames[:] = [d for d in dirnames if d not in skip_parts]
                for fn in filenames:
                    if fn.lower().endswith((".pt", ".pth")):
                        found.add((Path(dirpath) / fn).resolve())
                        count += 1
                        if count >= max_per_root:
                            dirnames.clear()
                            break
                if count >= max_per_root:
                    break
        except (PermissionError, OSError) as exc:
            if verbose:
                print(f"  [警告] 访问受限，跳过: {exc}")
    return sorted(found)


def inspect_weight(path: Path) -> dict:
    """读取权重文件的元数据，判断单模态 / 多模态。

    Args:
        path: 权重文件路径。

    Returns:
        描述该权重的字典，含 kind（mm/single/unknown）、params、元数据等。
    """
    info: dict = {"path": str(path), "kind": "unknown", "params": None,
                  "meta": {}, "note": "", "size_mb": None}

    try:
        info["size_mb"] = round(path.stat().st_size / 1024 / 1024, 1)
    except OSError:
        pass

    try:
        import torch  # noqa: F401
    except ImportError:
        info["note"] = "未安装 torch，无法读取内容"
        return info

    # 关键：本项目 checkpoint 里存的是 ultralytics 的模型对象，
    # torch.load 反序列化时需要能 import ultralytics 才能重建类。
    # 因此在**子进程**里加载：子进程可干净地把项目根加入 sys.path，
    # 既不污染当前进程，也避免 "No module named 'ultralytics'"。
    result = _inspect_weight_subprocess(path)
    if result is None:
        info["note"] = "读取失败：子进程无法解析该权重"
        return info
    info.update(result)
    return info


def _inspect_weight_subprocess(path: Path) -> dict | None:
    """在子进程中加载权重并返回元数据。

    Returns:
        dict（含 params / meta / x_like_count 等）或 None（失败）。
    """
    import subprocess
    import textwrap

    proj_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        f"""
        import json, re, sys
        sys.path.insert(0, {str(proj_root)!r})
        import torch

        out = {{"params": None, "meta": {{}}, "x_like_count": 0,
                "param_count_named": 0, "ok": False, "err": ""}}
        try:
            ckpt = torch.load({str(path)!r}, map_location="cpu", weights_only=False)
        except Exception as e:
            out["err"] = f"{{type(e).__name__}}: {{e}}"
            print("__RESULT__" + json.dumps(out))
            raise SystemExit(0)

        if not isinstance(ckpt, dict):
            out["err"] = "非标准 checkpoint（顶层不是 dict）"
            print("__RESULT__" + json.dumps(out))
            raise SystemExit(0)

        for k in {list(MM_META_KEYS)!r}:
            if k in ckpt:
                v = ckpt[k]
                try:
                    json.dumps(v)
                    out["meta"][k] = v
                except Exception:
                    out["meta"][k] = str(v)

        m = ckpt.get("model")
        if m is None:
            out["err"] = "无 'model' 字段"
        else:
            try:
                out["params"] = int(sum(p.numel() for p in m.parameters()))
            except Exception:
                pass
            try:
                names = [n for n, _ in m.named_parameters()]
                out["param_count_named"] = len(names)
                pat = r"(^|\\.)(x_model|model_x|modal|x_backbone)|_x\\.|\\.x\\."
                out["x_like_count"] = len([n for n in names if re.search(pat, n, re.I)])
            except Exception:
                pass
            out["ok"] = True

        print("__RESULT__" + json.dumps(out))
        """
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        val = env.get(key)
        if val is not None:
            try:
                if int(str(val).strip()) > 0:
                    continue
            except (ValueError, TypeError):
                pass
            env[key] = "8"

    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=str(proj_root),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=300)
    except Exception:
        return None

    import json as _json
    for line in ((r.stdout or "") + (r.stderr or "")).splitlines():
        if line.startswith("__RESULT__"):
            try:
                return _json.loads(line[len("__RESULT__"):])
            except Exception:
                return None
    return None


def classify_weight(info: dict) -> dict:
    """根据已取得的元数据，判定权重类型并补全 note。

    Args:
        info: 含 params / meta / x_like_count 的字典（原地修改）。

    Returns:
        同一个字典。
    """
    if info.get("err"):
        info["note"] = info["err"]
        info["kind"] = "unknown"
        return info

    params = info.get("params")
    x_like_count = info.get("x_like_count") or 0
    has_mm_meta = bool(info.get("meta"))

    if has_mm_meta or x_like_count > 0:
        info["kind"] = "mm"
    elif params:
        info["kind"] = "single"
        info["note"] = ("未发现多模态元数据/双分支，疑似单模态权重"
                        "（仅可用于骨干初始化）")
    else:
        info["kind"] = "unknown"
    return info


def report_weights(verbose: bool = True) -> list[dict]:
    """盘点服务器上的权重并打印报告。

    Returns:
        各权重的信息字典列表。
    """
    hr("第一部分：服务器权重盘点")
    print("搜索路径：" + ", ".join(str(Path(r).expanduser()) for r in DEFAULT_WEIGHT_ROOTS))
    files = find_weight_files(DEFAULT_WEIGHT_ROOTS, verbose=verbose)

    if not files:
        hr()
        print("未发现任何 .pt / .pth 权重文件。")
        print("\n结论：将走【路线 A：随机初始化】。")
        print("     也可以先手动下载 yolo11n.pt 到服务器，再考虑路线 B。")
        return []

    hr(f"共发现 {len(files)} 个权重文件，逐个读取元数据")
    results = []
    for f in files:
        info = classify_weight(inspect_weight(f))
        results.append(info)
        kind_label = {"mm": "多模态(本项目)", "single": "单模态(COCO)", "unknown": "未知"}[info["kind"]]
        print(f"\n  {info['path']}")
        if info.get("params"):
            print(f"    大小 {info['size_mb']} MB ｜ 参数量 {info['params'] / 1e6:.2f}M")
        else:
            print(f"    大小 {info['size_mb']} MB")
        print(f"    判定: {kind_label}")
        if info["meta"]:
            for k, v in info["meta"].items():
                sv = str(v)
                print(f"      {k} = {sv[:70]}{'...' if len(sv) > 70 else ''}")
        if info.get("x_like_count"):
            print(f"    双分支特征参数数: {info['x_like_count']} / {info.get('param_count_named')}")
        if info["note"]:
            print(f"    备注: {info['note']}")

    mm = [r for r in results if r["kind"] == "mm"]
    single = [r for r in results if r["kind"] == "single"]

    hr("权重盘点结论")
    if mm:
        print(f"发现 {len(mm)} 个多模态权重，可用于断点续训或微调：")
        for r in mm:
            print(f"  - {r['path']}")
        print("\n建议：走【路线 C：多模态权重续训】（优于 A/B）")
    else:
        print("未发现多模态权重。")
        if single:
            print(f"发现 {len(single)} 个单模态权重，可作为骨干初始化候选（路线 B）：")
            for r in single:
                print(f"  - {r['path']}  ({r['params'] / 1e6:.2f}M)" if r["params"] else f"  - {r['path']}")
            print("\n注意：路线 B 能否生效取决于参数名能否对上，务必先做一次 load 测试（见报告末尾提示）。")
        else:
            print("也未识别到可用的单模态权重。")
        print("\n建议：默认走【路线 A：随机初始化】。")

    return results


def test_weight_load(weight_path: str | None) -> None:
    """实测某个权重能否加载进多模态模型，并打印转移比例。

    Args:
        weight_path: 候选权重路径；None 时跳过。
    """
    if not weight_path:
        return
    hr("路线 B 可行性实测：把单模态权重加载进多模态模型")
    print(f"候选权重: {weight_path}")
    print("说明：多模态模型有 RGB / X 两条骨干，单模态权重只有一条，")
    print("      名字能对上的通常只有部分层。看下面日志里 Transferred 的项数占比。\n")

    proj = Path(__file__).resolve().parents[1]
    cfg = proj / "ultralytics/cfg/models/mm/detect/yolo11/extraction/yolo11n-mm-c3k2_pkimodule.yaml"
    if not cfg.exists():
        print(f"[跳过] 找不到模型配置: {cfg}")
        return

    code = f'''
import sys
sys.path.insert(0, {str(proj)!r})
from ultralytics import YOLOMM
m = YOLOMM({str(cfg)!r})
m.load({weight_path!r})
'''
    import subprocess
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=str(proj),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            if any(t in line for t in ("Transferred", "Params", "Error", "error", "Traceback")):
                print("   " + line.strip())
        if r.returncode != 0:
            print("   加载过程报错，路线 B 不可直接用。")
        else:
            print("\n   判读：Transferred 的占比高（>50%）则路线 B 划算；")
            print("         占比很低则收益有限，改用路线 A 更简单。")
    except Exception as exc:
        print(f"   [失败] {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 第二部分：数据配对校验
# --------------------------------------------------------------------------- #

def load_data_yaml(path: str | None) -> dict:
    """读取 data.yaml。

    Args:
        path: data.yaml 路径；None 时在常见位置自动寻找。

    Returns:
        解析后的字典；失败时返回空 dict。
    """
    if path:
        candidates = [Path(path)]
    else:
        proj = Path(__file__).resolve().parents[1]
        candidates = [proj / "data.yaml", Path("D:/uav_data/data.yaml"),
                      Path("/root/autodl-tmp/uav_data/data.yaml")]

    for c in candidates:
        if c.exists():
            try:
                import yaml
                with open(c, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                print(f"  已读取配置: {c}")
                return data
            except Exception as exc:
                print(f"  [警告] 解析失败 {c}: {exc}")
    print("  [错误] 未找到可用的 data.yaml，请用 --data 指定")
    return {}


def resolve_paths(data: dict) -> tuple[Path, dict, dict]:
    """解析 data.yaml 中的根目录与各模态目录。

    Args:
        data: data.yaml 内容。

    Returns:
        (根目录, split 名 -> 相对目录, 模态名 -> 相对目录)
    """
    root = Path(str(data.get("path", "."))).expanduser()
    splits = {k: data[k] for k in ("train", "val", "test") if k in data}
    modalities = data.get("modalities", {}) or {}
    return root, splits, modalities


def collect_stems(d: Path, exts=IMAGE_EXTS, cap: int = 100000) -> set[str]:
    """收集目录下指定扩展名文件的文件名（不含扩展名）。

    Args:
        d: 目录。
        exts: 允许的扩展名元组。
        cap: 上限，防止异常大目录拖死。

    Returns:
        stem 集合。
    """
    stems: set[str] = set()
    if not d.exists():
        return stems
    exts_l = tuple(e.lower() for e in exts)
    for i, p in enumerate(d.iterdir()):
        if i >= cap:
            break
        if p.is_file() and p.suffix.lower() in exts_l:
            stems.add(p.stem)
    return stems


def check_label_pairing(root: Path, split_dir: str, label_dir: str,
                        sample: int, verbose: bool = True) -> dict:
    """校验某个 split 下 RGB / IR / 标签 三者是否严格同名配对。

    同时抽查若干样本，确认 RGB 与 IR 的**图像尺寸比**是否稳定
    （尺寸比若不一致，说明存在错配或传感器设置变更）。

    Args:
        root: 数据根目录。
        split_dir: 如 'images/train'。
        label_dir: 如 'labels/train'。
        sample: 抽查数量。
        verbose: 是否打印细节。

    Returns:
        统计结果字典。
    """
    rgb_dir = root / split_dir
    label_path = root / label_dir
    # IR 目录名由 RGB 目录名推导：images/train -> images_ir/train
    parts = split_dir.split("/", 1)
    ir_dir = root / (parts[0] + "_ir" + ("/" + parts[1] if len(parts) > 1 else ""))

    label_ir_path = root / ("labels_ir/" + split_dir.split("/", 1)[-1])
    res = {"split": split_dir, "rgb": rgb_dir, "ir": ir_dir, "label": label_path,
           "label_ir": label_ir_path,
           "n_rgb": 0, "n_ir": 0, "n_label": 0,
           "n_label_ir": 0,
           "only_rgb": [], "only_ir": [], "no_label": [], "ratio_bad": [], "ok": False}

    if not rgb_dir.exists():
        res["note"] = f"RGB 目录不存在: {rgb_dir}"
        return res
    if not ir_dir.exists():
        res["note"] = f"IR 目录不存在: {ir_dir}"
        return res

    s_rgb = collect_stems(rgb_dir)
    s_ir = collect_stems(ir_dir)
    s_lab = collect_stems(label_path, exts=(".txt",)) if label_path.exists() else set()
    s_lab_ir = collect_stems(label_ir_path, exts=(".txt",)) if label_ir_path.exists() else set()

    res.update(n_rgb=len(s_rgb), n_ir=len(s_ir), n_label=len(s_lab), n_label_ir=len(s_lab_ir))
    res["only_rgb"] = sorted(s_rgb - s_ir)[:5]
    res["only_ir"] = sorted(s_ir - s_rgb)[:5]
    res["no_label"] = sorted(s_rgb - s_lab)[:5] if s_lab else []
    res["no_label_ir"] = sorted(s_rgb - s_lab_ir)[:5] if s_lab_ir else []

    # 抽查图像尺寸比
    try:
        import cv2
        import numpy as np

        def imread_u(p):
            return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

        common = sorted(s_rgb & s_ir)
        if common:
            rng = random.Random(0)
            picks = rng.sample(common, min(sample, len(common)))
            ratios = []
            miss = 0
            for st in picks:
                if verbose and len(ratios) % 20 == 0:
                    print(f"    抽查中 {len(ratios)}/{len(picks)} ...", end="\r")
                fr = next((rgb_dir / (st + e) for e in IMAGE_EXTS if (rgb_dir / (st + e)).exists()), None)
                fi = next((ir_dir / (st + e) for e in IMAGE_EXTS if (ir_dir / (st + e)).exists()), None)
                if fr is None or fi is None:
                    miss += 1
                    continue
                a, b = imread_u(fr), imread_u(fi)
                if a is None or b is None:
                    miss += 1
                    continue
                # 高度比 / 宽度比（RGB 相对 IR）
                ratios.append((a.shape[0] / b.shape[0], a.shape[1] / b.shape[1]))
            if verbose and picks:
                print(" " * 60, end="\r")

            res["sampled"] = len(picks)
            res["unreadable"] = miss
            if ratios:
                hs = [r[0] for r in ratios]
                ws = [r[1] for r in ratios]
                res["h_ratio_median"] = round(float(np.median(hs)), 4)
                res["w_ratio_median"] = round(float(np.median(ws)), 4)
                res["h_ratio_std"] = round(float(np.std(hs)), 4)
                res["w_ratio_std"] = round(float(np.std(ws)), 4)
                # 尺寸比若离散度大 -> 可疑错配
                if np.std(hs) > 0.02 or np.std(ws) > 0.02:
                    res["ratio_bad"] = [f"h_ratio_std={np.std(hs):.4f}, w_ratio_std={np.std(ws):.4f}"]
    except ImportError:
        res["note"] = "未安装 opencv/numpy，跳过图像抽查"

    res["ok"] = (len(res["only_rgb"]) == 0 and len(res["only_ir"]) == 0
                 and len(res["no_label"]) == 0 and len(res["no_label_ir"]) == 0
                 and not res["ratio_bad"])
    return res


def check_frame_metadata(root: Path, metadata_cfg, split: str, stems: set[str]) -> dict:
    """Validate the optional v2 per-frame modality-validity contract."""
    if not metadata_cfg:
        return {"configured": False, "ok": True, "count": 0, "kinds": {}}
    meta_root = Path(str(metadata_cfg))
    if not meta_root.is_absolute():
        meta_root = root / meta_root
    path = meta_root if meta_root.suffix.lower() == ".json" else meta_root / f"{split}.json"
    if not path.exists():
        return {"configured": True, "ok": False, "note": f"metadata 不存在: {path}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
        if not isinstance(frames, dict):
            return {"configured": True, "ok": False, "note": "frames 不是 object"}
        missing = sorted(stems - set(frames))[:5]
        extra = sorted(set(frames) - stems)[:5]
        kinds = {}
        bad = []
        for stem, rec in frames.items():
            if not isinstance(rec, dict):
                bad.append(stem)
                continue
            vr, vi = bool(rec.get("valid_rgb", True)), bool(rec.get("valid_ir", True))
            expected = "both" if vr and vi else "only_rgb" if vr else "only_ir" if vi else "background"
            kinds[expected] = kinds.get(expected, 0) + 1
            if rec.get("kind", expected) != expected:
                bad.append(stem)
        return {"configured": True, "ok": not (missing or extra or bad), "count": len(frames),
                "kinds": kinds, "missing": missing, "extra": extra, "bad": bad[:5]}
    except Exception as exc:
        return {"configured": True, "ok": False, "note": f"metadata 解析失败: {type(exc).__name__}: {exc}"}


def report_pairing(data_path: str | None, sample: int, verbose: bool = True) -> bool:
    """执行数据配对校验并打印报告。

    Args:
        data_path: data.yaml 路径。
        sample: 每个 split 的抽查数量。
        verbose: 是否打印细节。

    Returns:
        True 表示全部通过。
    """
    hr("第二部分：数据配对 / 标签 一致性校验")
    data = load_data_yaml(data_path)
    if not data:
        return False

    root, splits, modalities = resolve_paths(data)
    metadata_cfg = data.get("frame_metadata")
    print(f"  数据根目录 path = {root}")
    print(f"  模态映射 modalities = {modalities}")
    print(f"  本次训练的模态组合 modality_used = {data.get('modality_used')}")
    print(f"  类别 names = {data.get('names')}")

    if not root.exists():
        print(f"\n  [错误] 根目录不存在: {root}")
        print("         服务器上请把 data.yaml 的 path 改成实际数据路径。")
        return False

    all_ok = True
    for key, split_dir in splits.items():
        if key == "test":
            continue  # test 不参与训练，仅提示
        print(f"\n  ---- split: {key}  ({split_dir}) ----")
        r = check_label_pairing(root, split_dir, "labels/" + split_dir.split("/", 1)[-1],
                                sample=sample, verbose=verbose)
        if "note" in r and "不存在" in r.get("note", ""):
            print(f"    [错误] {r['note']}")
            all_ok = False
            continue

        print(f"    RGB {r['n_rgb']} ｜ IR {r['n_ir']} ｜ 标签 {r['n_label']} ｜ IR标签 {r['n_label_ir']}")
        if r.get("h_ratio_median"):
            print(f"    尺寸比 RGB/IR  中位 h={r['h_ratio_median']} w={r['w_ratio_median']}  "
                  f"std ({r['h_ratio_std']}, {r['w_ratio_std']})   [抽查 {r.get('sampled')}]")
        if r["only_rgb"]:
            print(f"    [错误] 有 RGB 但缺 IR: {len(r['only_rgb'])} 例，如 {r['only_rgb'][:3]}")
            all_ok = False
        if r["only_ir"]:
            print(f"    [错误] 有 IR 但缺 RGB: {len(r['only_ir'])} 例，如 {r['only_ir'][:3]}")
            all_ok = False
        if r["no_label"]:
            print(f"    [错误] 有图但缺标签: {len(r['no_label'])} 例，如 {r['no_label'][:3]}")
            all_ok = False
        if r.get("no_label_ir"):
            print(f"    [错误] 有图但缺 IR 标签：{len(r['no_label_ir'])} 例，如 {r['no_label_ir'][:3]}")
            all_ok = False
        stems = collect_stems(r["rgb"])
        mr = check_frame_metadata(root, metadata_cfg, key, stems)
        if mr.get("configured"):
            if mr.get("ok"):
                print(f"    [通过] metadata {mr.get('count', 0)} 条，类型统计 {mr.get('kinds', {})}")
            else:
                print(f"    [错误] metadata 契约失败: {mr.get('note', '')} missing={mr.get('missing', [])} extra={mr.get('extra', [])} bad={mr.get('bad', [])}")
                all_ok = False
        if r["ratio_bad"]:
            print(f"    [警告] 尺寸比离散度偏大，可能错配: {r['ratio_bad']}")
        if not (r["only_rgb"] or r["only_ir"] or r["no_label"] or r.get("no_label_ir") or r["ratio_bad"] or (mr.get("configured") and not mr.get("ok"))):
            print("    [通过] 三者严格同名配对，尺寸比稳定")

    hr("数据校验结论")
    if all_ok:
        print("通过：RGB / IR / 标签 严格对齐，可以开始训练。")
    else:
        print("存在问题：请先修复上面标出的缺漏/错配，再开始训练。")
        print("错配的数据会直接污染训练，务必处理干净。")
    return all_ok


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    """CLI 入口。"""
    ap = argparse.ArgumentParser(
        description="训练前自检：权重盘点 + 数据配对校验",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data.yaml 路径")
    ap.add_argument("--pair-samples", type=int, default=2000,
                    help="每个 split 抽查多少对图像（默认 2000）")
    ap.add_argument("--test-weight", default=None,
                    help="指定一个权重，实测能否加载进多模态模型（路线 B 可行性）")
    ap.add_argument("--skip-weights", action="store_true", help="跳过权重盘点")
    ap.add_argument("--skip-pairing", action="store_true", help="跳过数据配对校验")
    args = ap.parse_args()

    print("=" * 72)
    print("服务器训练前自检 preflight_train.py")
    print("=" * 72)

    weights = []
    if not args.skip_weights:
        weights = report_weights()

        # 若用户没指定，自动挑一个单模态权重做加载实测
        target = args.test_weight
        if target is None:
            cands = [r["path"] for r in weights if r["kind"] == "single"]
            if cands:
                target = cands[0]
                print(f"\n[自动] 选取单模态权重做路线 B 实测: {target}")
        test_weight_load(target)

    if not args.skip_pairing:
        report_pairing(args.data, args.pair_samples)

    hr("下一步")
    print("1) 修好上面报出的任何数据问题")
    print("2) 保持 epochs=2 先跑一次冒烟测试，确认能出 best.pt")
    print("3) 再改用 train_full.py 启动完整训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
