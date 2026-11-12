"""
NormalGen：离线表面法向（surface normal）模态生成器。

从单目深度反投影得到相机系法向，作为图图融合的副模态 X。
相对 DepthGen：深度编码距离，法向编码局部朝向。
相对 DEMGen 的 aspect：那是亮度梯度方向，不是 3D 法向。

深度来源
--------
- file：读取已有 images_depth/<split>/（npy / uint16 png / 单通道 tif）
- generate：用 DepthGen 后端（DA-V2 / ZipDepth）从可见光现算深度
- auto：深度图齐全则 file，否则在提供 weights 时 generate

落盘
----
- 默认 images_normal/<split>/<stem>.png
- 3 通道，相机系 OpenCV（x 右、y 下、z 前），PNG 为 uint8 的 (n+1)/2
- 禁止伪彩调色板

训练侧
------
  modality: {rgb: images, normal: images_normal}
  x_modality: normal
  Xch: 3
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_yaml

from .base import ModalGeneratorBase, SaveOptions


DEPTH_FILE_EXTS = (".npy", ".npz", ".png", ".tif", ".tiff")


class FromDepthNormalGenerator(nn.Module):
    """
    深度 → 相机系法向。

    3F2N 风格：u/v 各取两侧差分中 |ΔZ| 更小的一边再做叉积，减轻遮挡边污染。
    大跳变像素回落到 (0, 0, 1)。
    """

    def __init__(self, jump_rel: float = 0.08) -> None:
        super().__init__()
        self.jump_rel = float(jump_rel)

    def forward(
        self,
        depth: torch.Tensor,
        fx: float | torch.Tensor,
        fy: float | torch.Tensor,
        cx: float | torch.Tensor,
        cy: float | torch.Tensor,
    ) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() != 4 or depth.shape[1] != 1:
            raise ValueError(f"depth 形状必须是 [B,1,H,W]，实际 {tuple(depth.shape)}")
        z = depth.clamp_min(1e-6)
        b, _, h, w = z.shape
        device, dtype = z.device, z.dtype

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        xx = xx.unsqueeze(0).unsqueeze(0)
        yy = yy.unsqueeze(0).unsqueeze(0)
        fx_t = self._as_hw_scalar(fx, b, device, dtype)
        fy_t = self._as_hw_scalar(fy, b, device, dtype)
        cx_t = self._as_hw_scalar(cx, b, device, dtype)
        cy_t = self._as_hw_scalar(cy, b, device, dtype)

        pts = self._unproject(z, xx, yy, fx_t, fy_t, cx_t, cy_t)
        pts_pad = F.pad(pts, (1, 1, 1, 1), mode="replicate")
        p_l = pts_pad[:, :, 1:-1, :-2]
        p_r = pts_pad[:, :, 1:-1, 2:]
        p_u = pts_pad[:, :, :-2, 1:-1]
        p_d = pts_pad[:, :, 2:, 1:-1]

        dz_l = (pts - p_l)[:, 2:3].abs()
        dz_r = (p_r - pts)[:, 2:3].abs()
        dz_u = (pts - p_u)[:, 2:3].abs()
        dz_d = (p_d - pts)[:, 2:3].abs()
        use_r = dz_r < dz_l
        use_d = dz_d < dz_u
        tu = torch.where(use_r, p_r - pts, pts - p_l)
        tv = torch.where(use_d, p_d - pts, pts - p_u)

        n = F.normalize(torch.cross(tu, tv, dim=1), dim=1, eps=1e-8)
        jump = (torch.minimum(dz_l, dz_r) > self.jump_rel * z) | (
            torch.minimum(dz_u, dz_d) > self.jump_rel * z
        )
        fallback = torch.zeros_like(n)
        fallback[:, 2] = 1.0
        n = torch.where(jump, fallback, n)
        return F.normalize(n, dim=1, eps=1e-8)

    @staticmethod
    def _as_hw_scalar(
        value: float | torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value.to(device=device, dtype=dtype)
            if t.numel() == 1:
                return t.view(1, 1, 1, 1)
            if t.numel() == batch:
                return t.view(batch, 1, 1, 1)
            raise ValueError(f"内参张量元素数必须是 1 或 B={batch}，实际 {t.numel()}")
        return torch.tensor(float(value), device=device, dtype=dtype).view(1, 1, 1, 1)

    @staticmethod
    def _unproject(
        z: torch.Tensor,
        xx: torch.Tensor,
        yy: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
    ) -> torch.Tensor:
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        return torch.cat([x, y, z], dim=1)


def _load_depth_array(path: str | Path) -> np.ndarray:
    """读取 DepthGen 落盘的 X 深度，拒绝伪彩 3 通道。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(p), allow_pickle=False)
    elif suffix == ".npz":
        packed = np.load(str(p), allow_pickle=False)
        key = None
        for k in ("image", "arr_0", "array", "data", "features", "depth"):
            if k in packed:
                key = k
                break
        if key is None:
            if len(packed.files) != 1:
                raise ValueError(f"无法从 npz 解析深度数组: {p} keys={packed.files}")
            key = packed.files[0]
        arr = packed[key]
    else:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"无法读取深度图: {p}")

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        raise ValueError(
            f"深度图必须是单通道几何量，拒绝伪彩/RGB: shape={arr.shape} file={p}。"
            f"请用 DepthGen save_raw16 / save_npy / grayscale_color，不要用 Spectral 伪彩。"
        )
    if arr.ndim != 2:
        raise ValueError(f"深度图必须是 HxW，实际 shape={arr.shape} file={p}")
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _find_depth_file(depth_dir: Path, stem: str) -> Optional[Path]:
    for ext in DEPTH_FILE_EXTS:
        cand = depth_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


class NormalGen(ModalGeneratorBase):
    """
    离线表面法向生成器。

    输入为 data.yaml 时，默认把输出写到 images_normal/<split>/xxx.(png|npy|npz|tif)。
    输出固定 3 通道相机系法向。
    """

    backend_name = "from_depth"

    def __init__(
        self,
        source_modality: str = "rgb",
        device: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        save_dir: str | Path | None = None,
        keep_structure: bool = True,
        overwrite: bool = False,
        split: Optional[str | List[str]] = None,
        save_format: str = "png",
        xch: int = 3,
        depth_source: str = "auto",
        depth_dir: str | Path | None = None,
        weights: Optional[str] = None,
        depth_model: str = "da",
        encoder: Optional[str] = None,
        invert: bool = True,
        metric: bool = False,
        fov_deg: float = 60.0,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        jump_rel: float = 0.08,
        z_near: float = 0.5,
        z_far: float = 10.0,
        **kwargs,
    ) -> None:
        self.source_modality = str(source_modality).lower()
        self.split = split
        self.save_format = str(save_format).lower()
        if self.save_format == "tiff":
            self.save_format = "tif"
        if self.save_format not in {"png", "npy", "npz", "tif"}:
            raise ValueError("save_format 必须为 'png'/'npy'/'tif'（可选: 'npz'）")

        self.xch = int(xch)
        if self.xch != 3:
            raise ValueError("表面法向固定 3 通道，xch 必须为 3")

        self.depth_source = str(depth_source).lower()
        if self.depth_source not in {"auto", "file", "generate"}:
            raise ValueError("depth_source 必须为 'auto' / 'file' / 'generate'")
        self.depth_dir = Path(depth_dir) if depth_dir is not None else None
        self.weights = Path(weights) if weights else None
        self.depth_model = depth_model
        self.encoder = encoder
        self.invert = bool(invert)
        self.metric = bool(metric)
        self.fov_deg = float(fov_deg)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.jump_rel = float(jump_rel)
        self.z_near = float(z_near)
        self.z_far = float(z_far)
        if self.z_far <= self.z_near:
            raise ValueError("z_far 必须大于 z_near")

        if self.depth_source == "generate" and self.weights is None:
            raise ValueError("depth_source='generate' 必须提供 DepthGen 权重路径 weights=")
        if self.weights is not None and not self.weights.exists():
            raise FileNotFoundError(f"权重文件不存在: {self.weights}")

        save_options = SaveOptions(
            save_dir=save_dir,
            keep_structure=keep_structure,
            overwrite=overwrite,
            enable_save=True,
        )
        super().__init__(
            method=self.backend_name,
            method_cfg={},
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            save_options=save_options,
        )

        self.normal_model: Optional[FromDepthNormalGenerator] = None
        self._depth_gen = None
        self._resolved_depth_mode: Optional[str] = None
        self._img_formats = {
            "bmp",
            "dng",
            "jpeg",
            "jpg",
            "mpo",
            "png",
            "tif",
            "tiff",
            "webp",
            "pfm",
            "heic",
        }
        self._path_meta: Dict[str, Dict[str, Any]] = {}
        self._yaml_root: Optional[Path] = None
        self._debug_logged: set[str] = set()

    def run(self, source: str | Path | Iterable[str | Path], save_dir: str | Path | None = None):
        if save_dir is not None:
            self.save_options.save_dir = save_dir

        split = getattr(self, "split", None)
        if split:
            if isinstance(split, str):
                self.split_names = [s.strip() for s in split.split(",") if s.strip()]
            else:
                self.split_names = list(split)
        else:
            self.split_names = []

        return super().run(source)

    def _guess_depth_file(self, src: Path) -> Optional[Path]:
        """非 yaml 输入时定位同 stem 深度图。"""
        if self.depth_dir is not None:
            found = _find_depth_file(self.depth_dir, src.stem)
            if found is not None:
                return found
            found = _find_depth_file(self.depth_dir / src.parent.name, src.stem)
            if found is not None:
                return found
        if src.parent.parent.name == "images":
            found = _find_depth_file(src.parent.parent.parent / "images_depth" / src.parent.name, src.stem)
            if found is not None:
                return found
        if src.parent.name == "images":
            found = _find_depth_file(src.parent.parent / "images_depth", src.stem)
            if found is not None:
                return found
        return _find_depth_file(src.parent, src.stem)

    def _depth_dir_for_meta(self, images_root: Path, split_subdir: str) -> Path:
        if self.depth_dir is not None:
            d = self.depth_dir
            split_dir = d / split_subdir
            return split_dir if split_dir.is_dir() else d
        return images_root.parent / "images_depth" / split_subdir

    def _resolve_depth_mode(self, paths: List[str]) -> str:
        if self.depth_source == "generate":
            return "generate"
        found = 0
        missing: List[str] = []
        for p in paths:
            meta = self._path_meta.get(p, {})
            depth_path = meta.get("depth_path")
            if depth_path and Path(depth_path).is_file():
                found += 1
            else:
                missing.append(p)
        if self.depth_source == "file":
            if missing:
                preview = "\n".join(missing[:8])
                raise FileNotFoundError(
                    f"depth_source='file' 但缺少深度图 {len(missing)}/{len(paths)}。"
                    f"默认查找 images_depth/<split>/<stem>.(npy|png|tif)。示例:\n{preview}"
                )
            return "file"
        if found == len(paths):
            return "file"
        if found == 0 and self.weights is not None:
            LOGGER.info(f"[{self.method}] 未找到 images_depth，改用 DepthGen 权重生成深度再做法向")
            return "generate"
        if found == 0:
            raise FileNotFoundError(
                "未找到深度图且未提供 weights。先运行 DepthGen 得到 images_depth/，"
                "或传入 weights=... 并设 depth_source='generate'"
            )
        raise FileNotFoundError(
            f"深度图不完整: {found}/{len(paths)} 有文件。不要混用已有深度与现算深度，"
            "请补齐 images_depth 或改用 depth_source='generate'"
        )

    def _gather_sources(self, source: str | Path | Iterable[str | Path]) -> List[str]:
        if isinstance(source, (str, Path)) and str(source).lower().endswith((".yaml", ".yml")):
            yaml_path = Path(check_yaml(source, hard=True))
            self._yaml_root = yaml_path.parent
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))

            base = Path(data.get("path", yaml_path.parent))
            if not base.is_absolute():
                base = yaml_path.parent / base

            modality_dirs = data.get("modality", {"rgb": "images"})
            if self.source_modality not in modality_dirs:
                raise ValueError(f"data.yaml 未包含 source_modality={self.source_modality} 的目录映射")
            source_dir = modality_dirs[self.source_modality]

            splits_cfg = []
            for k in ("train", "val", "test"):
                v = data.get(k)
                if v:
                    splits_cfg.append((k, v))
            if not splits_cfg:
                raise ValueError(f"{yaml_path} 中未找到 train/val/test 配置")

            paths: List[str] = []
            self._path_meta.clear()
            for split_name, split_val in splits_cfg:
                if getattr(self, "split_names", None) and self.split_names:
                    if split_name not in self.split_names:
                        continue
                split_list = split_val if isinstance(split_val, list) else [split_val]
                for s in split_list:
                    p = Path(s)
                    if not p.is_absolute():
                        p = base / p
                    if not p.exists():
                        raise FileNotFoundError(f"数据集路径不存在: {p}")

                    split_subdir = p.name
                    images_root = base / source_dir
                    input_dir = images_root / split_subdir
                    if not input_dir.exists():
                        raise FileNotFoundError(
                            f"source_modality={self.source_modality} split 目录不存在: {input_dir}"
                        )

                    normal_root_default = images_root.parent / "images_normal"
                    depth_split_dir = self._depth_dir_for_meta(images_root, split_subdir)
                    for f in input_dir.rglob("*.*"):
                        if f.is_file() and f.suffix[1:].lower() in self._img_formats:
                            fstr = str(f)
                            paths.append(fstr)
                            depth_file = _find_depth_file(depth_split_dir, f.stem)
                            self._path_meta[fstr] = {
                                "split": split_subdir,
                                "images_root": images_root,
                                "normal_root_default": normal_root_default,
                                "depth_path": str(depth_file) if depth_file is not None else None,
                            }

            if not paths:
                raise RuntimeError(f"{yaml_path} 未找到任何图像文件")
            paths = sorted(paths)
            self._resolved_depth_mode = self._resolve_depth_mode(paths)
            if self._resolved_depth_mode == "generate":
                self._load_depth_backend()
            return paths

        paths = super()._gather_sources(source)
        filtered = [p for p in paths if Path(p).suffix[1:].lower() in self._img_formats]
        if not filtered:
            raise RuntimeError("未找到任何可处理的图像文件")
        filtered = sorted(filtered)
        self._path_meta.clear()
        for fstr in filtered:
            src = Path(fstr)
            depth_file = self._guess_depth_file(src)
            if src.parent.name == "images":
                normal_root = src.parent.parent / "images_normal"
            elif src.parent.parent.name == "images":
                normal_root = src.parent.parent.parent / "images_normal"
            else:
                normal_root = src.parent.parent / "images_normal"
            self._path_meta[fstr] = {
                "split": src.parent.name,
                "normal_root_default": normal_root,
                "depth_path": str(depth_file) if depth_file is not None else None,
            }
        self._resolved_depth_mode = self._resolve_depth_mode(filtered)
        if self._resolved_depth_mode == "generate":
            self._load_depth_backend()
        return filtered

    def load_model(self):
        self.normal_model = FromDepthNormalGenerator(jump_rel=self.jump_rel).to(self.device).eval()
        self.model = self.normal_model
        if self.depth_source == "generate":
            self._load_depth_backend()
        LOGGER.info(
            f"[{self.method}] Normal 生成器初始化完成: format={self.save_format}, "
            f"depth_source={self.depth_source}, fov_deg={self.fov_deg}, invert={self.invert}, metric={self.metric}"
        )

    def _load_depth_backend(self):
        if self._depth_gen is not None:
            return
        if self.weights is None:
            raise ValueError("generate 模式需要 weights")
        from .depth_anything_v2 import DepthGen

        self._depth_gen = DepthGen(
            weights=str(self.weights),
            model=self.depth_model,
            encoder=self.encoder,
            device=self.device,
            batch_size=self.batch_size,
            save_color=False,
            save_raw16=False,
            save_npy=False,
            invert=False,
        )
        self._depth_gen.save_options.enable_save = False
        self._depth_gen._ensure_loaded()
        LOGGER.info(f"[{self.method}] 已加载 DepthGen 后端: model={self.depth_model} @ {self.weights}")

    def preprocess(self, item: str):
        mode = self._resolved_depth_mode or self.depth_source
        if mode == "generate":
            if self._depth_gen is None:
                self._load_depth_backend()
            tensor, meta = self._depth_gen.preprocess(item)
            if item in self._path_meta:
                meta.update(self._path_meta[item])
            return tensor, meta

        img = cv2.imread(item, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {item}")
        h, w = img.shape[:2]
        depth_path = (self._path_meta.get(item) or {}).get("depth_path")
        if not depth_path:
            raise FileNotFoundError(f"缺少对应深度图: {item}")
        depth_np = _load_depth_array(depth_path)
        if depth_np.shape[0] != h or depth_np.shape[1] != w:
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(self.device, non_blocking=True)
        meta = {"path": item, "orig_hw": (h, w)}
        if item in self._path_meta:
            meta.update(self._path_meta[item])
        return depth_t, meta

    def infer(self, batch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        mode = self._resolved_depth_mode or self.depth_source
        if mode == "generate":
            assert self._depth_gen is not None
            return self._depth_gen.infer(batch_inputs)
        return [t.squeeze(0) if t.dim() == 4 else t for t in batch_inputs]

    def _intrinsics(self, width: int, height: int) -> Tuple[float, float, float, float]:
        cx = float(self.cx) if self.cx is not None else 0.5 * (width - 1)
        cy = float(self.cy) if self.cy is not None else 0.5 * (height - 1)
        if self.fx is not None:
            fx = float(self.fx)
            fy = float(self.fy) if self.fy is not None else fx
            return fx, fy, cx, cy
        fov = math.radians(self.fov_deg)
        fx = 0.5 * width / math.tan(0.5 * fov)
        fy = float(self.fy) if self.fy is not None else fx
        return fx, fy, cx, cy

    def _depth_to_z(self, depth: torch.Tensor) -> torch.Tensor:
        if self.metric:
            return depth.clamp_min(1e-4)
        d = depth
        if self.invert:
            d_min = d.amin(dim=(-2, -1), keepdim=True)
            d_max = d.amax(dim=(-2, -1), keepdim=True)
            d = d_max - d
        flat = d.flatten(1)
        lo = torch.quantile(flat, 0.02, dim=1).view(-1, 1, 1, 1)
        hi = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
        d = d.clamp(lo, hi)
        d = (d - lo) / (hi - lo + 1e-6)
        return self.z_near + d * (self.z_far - self.z_near)

    def postprocess(self, outputs: List[Any], metas: List[Dict[str, Any]]) -> List[np.ndarray]:
        assert self.normal_model is not None
        mode = self._resolved_depth_mode or self.depth_source
        results: List[np.ndarray] = []

        depth_arrays: List[np.ndarray] = []
        if mode == "generate":
            assert self._depth_gen is not None
            depth_arrays = self._depth_gen.postprocess(outputs, metas)
        else:
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    t = out.detach().float()
                    while t.dim() > 2:
                        t = t.squeeze(0)
                    depth_arrays.append(t.cpu().numpy())
                else:
                    depth_arrays.append(np.asarray(out, dtype=np.float32))

        for depth_np, meta in zip(depth_arrays, metas):
            tgt_h, tgt_w = meta["orig_hw"]
            if depth_np.shape[0] != tgt_h or depth_np.shape[1] != tgt_w:
                depth_np = cv2.resize(depth_np, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            depth_t = (
                torch.from_numpy(np.ascontiguousarray(depth_np))
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            z = self._depth_to_z(depth_t)
            fx, fy, cx, cy = self._intrinsics(tgt_w, tgt_h)
            with torch.no_grad():
                n = self.normal_model(z, fx, fy, cx, cy)
            n_np = n[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
            results.append(n_np)
        return results

    def save(self, outputs: List[np.ndarray], metas: List[Dict[str, Any]]) -> List[str]:
        paths: List[str] = []
        save_opt: SaveOptions = self.save_options

        for n_np, meta in zip(outputs, metas):
            src_path = Path(meta["path"])
            split = meta.get("split", "unknown")

            if save_opt.save_dir is None and "normal_root_default" in meta:
                out_dir = Path(meta["normal_root_default"]) / split
            else:
                base_dir = Path(save_opt.save_dir or "runs/normal_gen") / self.method
                rel = (
                    src_path.relative_to(getattr(self, "_source_root", src_path.parent))
                    if save_opt.keep_structure
                    else Path(src_path.name)
                )
                out_dir = (base_dir / rel.parent) if save_opt.keep_structure else base_dir

            out_dir.mkdir(parents=True, exist_ok=True)
            stem = src_path.stem

            debug_key = str(out_dir)
            if debug_key not in self._debug_logged:
                LOGGER.info(f"[{self.method}] 保存目录: {out_dir}")
                self._debug_logged.add(debug_key)

            if self.save_format == "npy":
                out_path = out_dir / f"{stem}.npy"
                if save_opt.overwrite or not out_path.exists():
                    np.save(out_path, n_np.astype(np.float32))
                paths.append(str(out_path))
            elif self.save_format == "npz":
                out_path = out_dir / f"{stem}.npz"
                if save_opt.overwrite or not out_path.exists():
                    np.savez_compressed(out_path, features=n_np.astype(np.float32))
                paths.append(str(out_path))
            else:
                n_u8 = np.clip(0.5 * (n_np + 1.0), 0.0, 1.0)
                n_u8 = (n_u8 * 255.0).astype(np.uint8)
                ext = ".tif" if self.save_format == "tif" else ".png"
                out_path = out_dir / f"{stem}{ext}"
                img = n_u8[:, :, :3][:, :, ::-1]
                if save_opt.overwrite or not out_path.exists():
                    cv2.imwrite(str(out_path), img)
                paths.append(str(out_path))

        return paths


__all__ = ["NormalGen", "FromDepthNormalGenerator"]
