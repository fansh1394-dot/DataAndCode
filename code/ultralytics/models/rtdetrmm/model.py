# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RTDETRMM：独立的多模态 RT-DETR 家族模型入口。

设计目标：
- 不继承 RTDETR；不依赖 ultralytics.models.rtdetr.*。
- 不依赖文件名是否包含 "-mm"；而是基于 YAML/CKPT 的内容判据严格 Fail-Fast。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ultralytics.engine.model import Model
from ultralytics.nn.mm.complexity import (
    compute_default_multimodal_complexity_report,
    log_default_complexity,
)
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.val_results import COCOValResults, MMValResults

from .utils import detect_mm, require_multimodal
from ultralytics.nn.mm.utils import INPUT_SOURCE_ALIASES, resolve_input_source_role


class RTDETRMM(Model):
    """RT-DETR MultiModal（独立家族）。"""

    @staticmethod
    def _set_complexity_handshake(target: Any, *, pending=None, emitted=None, direct_ingress=None) -> None:
        """Persist lightweight owner/suppression hints via torch_utils constants only."""
        from ultralytics.utils.torch_utils import set_multimodal_complexity_owner_state
        # 只通过 torch_utils 的统一通道设置（使用 _mm_ 前缀的私有属性）
        # 不再向 kwargs dict 写入无前缀的 public key，避免 check_dict_alignment 报错
        if hasattr(target, '__dict__') and not isinstance(target, dict):
            set_multimodal_complexity_owner_state(
                target, pending=pending, emitted=emitted, direct_ingress=direct_ingress
            )

    def _log_canonical_complexity_summary(self, imgsz: int) -> bool:
        """Emit the canonical multimodal complexity summary from the model-level owner."""
        try:
            report = compute_default_multimodal_complexity_report(self.model, imgsz=imgsz)
            from ultralytics.utils import LOGGER as _LOGGER

            log_default_complexity(self.model, report, _LOGGER)
        except Exception:
            self._set_complexity_handshake(self.model, pending=False, emitted=False, direct_ingress=True)
            return False

        self._set_complexity_handshake(self.model, pending=False, emitted=True, direct_ingress=False)
        return True

    def __init__(
        self,
        model: Union[str, Path] = "rtdetr-r18-mm.pt",
        task: Optional[str] = None,
        ch: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize RTDETRMM; task is inferred from YAML/checkpoint when None.

        task=None follows YOLOMM's routing contract:
        - YAML: Model._new() calls guess_model_task(cfg_dict)
        - PT: Model._load() uses checkpoint model.task
        - explicit task overrides inference for scripts/debugging
        """
        p = Path(str(model))
        is_yaml_model = p.suffix.lower() in {".yaml", ".yml"}
        if is_yaml_model:
            ev = detect_mm(model)
            self.mm_evidence = ev.to_dict()
            self._requires_multimodal_build = not ev.is_multimodal
        else:
            self.mm_evidence = require_multimodal(model, who="RTDETRMM").to_dict()
            self._requires_multimodal_build = False

        self.input_channels = ch
        self.modality_config: Dict[str, Any] = {}
        self.is_multimodal = not self._requires_multimodal_build
        super().__init__(model=str(model), task=task, verbose=verbose)
        if self._requires_multimodal_build:
            self.modality_config = {
                "requires_multimodal_build": True,
                "supported_modalities": list(INPUT_SOURCE_ALIASES.keys()),
                "default_modality": "unbuilt",
            }
        else:
            self._configure_multimodal_settings(verbose=verbose)
            self._ensure_mm_router(verbose=verbose)

    @property
    def task_map(self) -> dict:
        from .mm_predictor import RTDETRMMPredictor
        from .train import RTDETRMMTrainer
        from .val import RTDETRMMValidator
        from .mm_predictor_obb import RTDETRMMOBBPredictor
        from .train_obb import RTDETRMMOBBTrainer
        from .val_obb import RTDETRMMOBBValidator
        from ultralytics.nn.tasks import RTDETRMMOBBModel

        return {
            "detect": {
                "predictor": RTDETRMMPredictor,
                "validator": RTDETRMMValidator,
                "trainer": RTDETRMMTrainer,
                "model": RTDETRDetectionModel,
            },
            "obb": {
                "predictor": RTDETRMMOBBPredictor,
                "validator": RTDETRMMOBBValidator,
                "trainer": RTDETRMMOBBTrainer,
                "model": RTDETRMMOBBModel,
            },
        }

    def _configure_multimodal_settings(self, verbose: bool = False) -> None:
        """Detect routing markers from self.model.yaml, infer channels and modality set."""
        try:
            model_yaml = getattr(self.model, "yaml", None)
            if not isinstance(model_yaml, dict) or not model_yaml:
                self.input_channels = self.input_channels or 3
                self.modality_config = {
                    "primary_channels": list(range(self.input_channels or 3)),
                    "secondary_channels": [],
                    "supported_modalities": list(INPUT_SOURCE_ALIASES.keys()),
                    "default_modality": "RGB",
                }
                return

            has_mm = self._detect_multimodal_layers(model_yaml)
            model_channels = int(model_yaml.get("ch", model_yaml.get("channels", 3)) or 3)
            if has_mm and self._has_dual_modality_layers(model_yaml):
                model_channels = (model_channels if model_channels > 3 else 3) * 2
            elif has_mm:
                model_channels = model_channels if model_channels >= 3 else 3

            if self.input_channels is None:
                self.input_channels = model_channels
            self._validate_input_channels()

            pch = min(3, self.input_channels)
            sch = max(0, self.input_channels - pch)
            if has_mm and self._has_dual_modality_layers(model_yaml):
                self.modality_config = {
                    "primary_channels": list(range(pch)),
                    "secondary_channels": list(range(pch, pch + sch)),
                    "supported_modalities": list(INPUT_SOURCE_ALIASES.keys()),
                    "default_modality": "Dual",
                }
            elif has_mm:
                self.modality_config = {
                    "primary_channels": list(range(pch)),
                    "secondary_channels": list(range(pch, pch + sch)),
                    "supported_modalities": ["RGB", "X", "A", "B"],
                    "default_modality": "RGB",
                }
            else:
                self.modality_config = {
                    "primary_channels": list(range(pch)),
                    "secondary_channels": [],
                    "supported_modalities": ["RGB"],
                    "default_modality": "RGB",
                }

            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.info(
                    f"RTDETRMM configured: ch={self.input_channels}, "
                    f"modalities={self.modality_config.get('supported_modalities', [])}"
                )

        except Exception as e:
            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.warning(f"RTDETRMM: multimodal settings failed: {e}")
            self.input_channels = self.input_channels or 3
            self.modality_config = {
                "supported_modalities": list(INPUT_SOURCE_ALIASES.keys()),
                "default_modality": "RGB",
            }

    def _detect_multimodal_layers(self, model_yaml: dict) -> bool:
        all_layers = model_yaml.get("backbone", []) + model_yaml.get("head", [])
        for layer in all_layers:
            if isinstance(layer, list) and len(layer) >= 5 and resolve_input_source_role(layer[4]) is not None:
                return True
        return False

    def _has_dual_modality_layers(self, model_yaml: dict) -> bool:
        all_layers = model_yaml.get("backbone", []) + model_yaml.get("head", [])
        for layer in all_layers:
            if isinstance(layer, list) and len(layer) >= 5:
                role = resolve_input_source_role(layer[4])
                if role == "FUSED":
                    return True
        return False

    def _validate_input_channels(self) -> None:
        if self.input_channels is not None and self.input_channels < 1:
            raise ValueError(
                f"RTDETRMM: Invalid input channels: {self.input_channels}. Must be >= 1."
            )

    def _ensure_mm_router(self, verbose: bool = False) -> None:
        """确保底层 RTDETRDetectionModel 持有 mm_router。"""
        m = getattr(self, "model", None)
        if m is None:
            return

        # 直连 PyTorch 模型
        if hasattr(m, "mm_router") and m.mm_router is not None:
            return

        # AutoBackend(PyTorch)
        if hasattr(m, "pt") and getattr(m, "pt", False) and hasattr(m, "model"):
            inner = getattr(m, "model", None)
            if inner is not None and hasattr(inner, "mm_router") and inner.mm_router is not None:
                return

        # 尝试补建
        try:
            from ultralytics.nn.mm import MultiModalRouter

            config_dict = getattr(m, "yaml", None)
            if hasattr(m, "mm_router"):
                m.mm_router = MultiModalRouter(config_dict, verbose=verbose)
            elif hasattr(m, "model") and getattr(m, "pt", False):
                inner = getattr(m, "model", None)
                if inner is not None and hasattr(inner, "mm_router"):
                    inner.mm_router = MultiModalRouter(config_dict, verbose=verbose)
        except Exception as e:
            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.warning(f"RTDETRMM: 无法创建 MultiModalRouter：{e}")

    def ensure_multimodal_model_channel_contract(
        self,
        *,
        data_cfg: dict,
        requested_modality: Optional[str] = None,
    ) -> None:
        """在 val/vis 前确认当前 PyTorch 模型与 data.yaml 的 Pch/Xch 契约一致。

        若契约不匹配，自动重建模型骨架并按 shape 兼容方式加载现有权重。
        """
        if not hasattr(self, "model") or not hasattr(self.model, "yaml"):
            return

        from copy import deepcopy

        cfg_dict = deepcopy(self.model.yaml) if isinstance(self.model.yaml, dict) else None
        if cfg_dict is None:
            return

        from ultralytics.nn.mm.utils import resolve_multimodal_channel_contract
        from ultralytics.utils import LOGGER

        cfg_dict["dataset_config"] = dict(data_cfg)
        contract = resolve_multimodal_channel_contract(
            data=data_cfg,
            requested_modality=requested_modality,
            model_cfg=cfg_dict,
        )
        expected_ch = contract.fused_ch if contract.is_multimodal_model else contract.primary_ch

        current_router = getattr(self.model, "mm_router", None)
        if current_router is not None and hasattr(current_router, "_build_contract"):
            build = current_router._build_contract
            current_ch = build["fused_ch"]
        else:
            current_ch = int(cfg_dict.get("ch", expected_ch))

        if current_ch == expected_ch:
            if current_router is not None:
                current_router.update_dataset_config(data_cfg)
                current_router.set_runtime_params(requested_modality)
            return

        prune_info = getattr(self.model, "prune_info", None)
        if not prune_info and isinstance(self.ckpt, dict):
            prune_info = self.ckpt.get("prune_info")
        if isinstance(prune_info, dict) and prune_info.get("is_pruned"):
            raise RuntimeError(
                f"通道契约不匹配（current ch={current_ch}，expected ch={expected_ch}）且模型已剪枝："
                "按 YAML 重建会摧毁剪枝结构。请使用与剪枝时一致的数据配置（Xch/x_modality）。"
            )

        LOGGER.info(f"RTDETRMM rebuilding model: current ch={current_ch} → expected ch={expected_ch}")
        model_cls = self.task_map[self.task]["model"]
        rebuilt = model_cls(cfg_dict, ch=expected_ch, verbose=False)
        current_state = self.model.state_dict()
        rebuilt_state = rebuilt.state_dict()
        compatible_state = {
            k: v
            for k, v in current_state.items()
            if k in rebuilt_state and rebuilt_state[k].shape == v.shape
        }
        rebuilt.load_state_dict(compatible_state, strict=False)
        if hasattr(rebuilt, "mm_router") and rebuilt.mm_router:
            rebuilt.mm_router.update_dataset_config(data_cfg)
            rebuilt.mm_router.set_runtime_params(requested_modality)
        self.model = rebuilt

    # --- API overrides: train / predict / val / vis ---

    def _raise_if_multimodal_build_required(self, method_name: str) -> None:
        """Prevent runtime use of a plain YAML before method-level single_modal build injection."""
        if getattr(self, "_requires_multimodal_build", False):
            raise RuntimeError(
                f"RTDETRMM.{method_name} cannot run from a plain RT-DETR YAML before multimodal build. "
                "Run train(data=..., single_modal=...) so the YAML can be injected at trainer build time, "
                "or instantiate RTDETRMM from an already multimodal YAML/PT."
            )

    def train(self, **kwargs):
        """RTDETRMM train. single_modal is passed through as a raw arg; YAML injection happens in trainer."""
        info = getattr(self.model, "prune_info", None)
        if not info and isinstance(self.ckpt, dict):
            info = self.ckpt.get("prune_info")
        if isinstance(info, dict) and info.get("is_pruned"):
            raise RuntimeError(
                "train() 不支持剪枝模型：会按原 YAML 重建未剪结构并静默丢弃剪枝权重。请使用 finetrain()。"
            )
        single_modal = kwargs.pop("single_modal", None)
        if single_modal:
            kwargs["single_modal"] = single_modal
        result = super().train(**kwargs)
        if single_modal:
            ev = detect_mm(getattr(self.model, "yaml", {}))
            if not ev.is_multimodal:
                raise RuntimeError(
                    "RTDETRMM.train(single_modal=...) completed without injected multimodal YAML metadata."
                )
            self.mm_evidence = ev.to_dict()
            self._requires_multimodal_build = False
            self.is_multimodal = True
        return result

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        """RTDETRMM predict."""
        self._raise_if_multimodal_build_required("predict")
        return super().predict(source=source, stream=stream, predictor=predictor, **kwargs)

    def get_modality_info(self) -> Dict[str, Any]:
        self._raise_if_multimodal_build_required("get_modality_info")
        return {
            "input_channels": self.input_channels,
            "modality_config": dict(self.modality_config),
            "model_type": "RTDETRMM",
            "task": getattr(self, "task", "detect"),
            "is_multimodal": getattr(self, "is_multimodal", True),
        }

    def vis(
        self,
        rgb_source: Optional[Union[str, np.ndarray, list]] = None,
        x_source: Optional[Union[str, np.ndarray, list]] = None,
        method: str = "heat",
        layers: Optional[list[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        overlay: Optional[str] = None,
        project: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        out_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        overlay_opacity: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """可视化入口：委托给 RTDETRMMVisualizationRunner（独立家族）。

        Args:
            overlay_opacity: 热力图叠加透明度 0-100%，值越大热力图越不透明（默认 50）。
        """
        self._raise_if_multimodal_build_required("vis")
        from .visualize.runner import RTDETRMMVisualizationRunner

        explicit_modality = modality  # only confirm if user explicitly passed modality

        # overlay_opacity (0-100) -> blend_alpha (0-1)
        if overlay_opacity is not None:
            if not isinstance(overlay_opacity, (int, float)):
                raise ValueError(f"overlay_opacity 必须为数值，收到: {type(overlay_opacity)}")
            val = float(overlay_opacity)
            if not (0 <= val <= 100):
                raise ValueError(f"overlay_opacity 范围 0-100，收到: {val}")
            kwargs['blend_alpha'] = val / 100.0
        elif 'blend_alpha' not in kwargs and 'alpha' not in kwargs:
            kwargs['blend_alpha'] = 0.5

        return RTDETRMMVisualizationRunner.run(
            model=self.model,
            rgb_source=rgb_source,
            x_source=x_source,
            method=method,
            layers=layers,
            modality=modality,
            save=save,
            overlay=overlay,
            project=str(project) if project is not None else None,
            name=name,
            out_dir=str(out_dir) if out_dir is not None else None,
            device=device,
            **kwargs,
        )

    def preflight(self, **kwargs):
        """
        Quick validation of RTDETRMM YAML config: verify it can complete a full training
        iteration (forward + loss + backward + optimizer step) using synthetic data.

        No real dataset required. Supports CLI mode: yolo preflight model=xxx.yaml

        Args:
            iters (int): Number of training iterations. Default 1.
            device (str | int): Device. Default 'cpu'.
            batch (int): Synthetic batch size. Default 2.
            imgsz (int): Image size. Default 640.
            scale (str): Model scale key (n/s/m/l/x). Requires YAML to define 'scales'.
                         Default: uses the first key in 'scales'.
            Xch (int): X modality channels. Default 3.
                        Also accepts primary_modality via dataset config for non-RGB primary modalities.
            verbose (bool): Verbose output. Default True.
            half (bool): Use FP16. Default False.

        Returns:
            PreflightReport: Report with pass/fail status per stage.
        """
        self._raise_if_multimodal_build_required("preflight")
        from ultralytics.engine.preflight import PreflightRunner

        model_path = ""
        if hasattr(self, "model") and hasattr(self.model, "yaml"):
            import tempfile
            import yaml as _yaml

            cfg_copy = dict(self.model.yaml) if isinstance(self.model.yaml, dict) else {}
            tmp = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
            _yaml.dump(cfg_copy, tmp)
            tmp.close()
            model_path = tmp.name

        runner = PreflightRunner(model_path, **kwargs)
        return runner.run()

    def val(self, validator=None, coco: bool = False, **kwargs: Any):
        """
        多模态 RT-DETR 的验证入口：统一 `val(coco=...)`。

        关键点：
        - RT-DETR 的后处理默认按 `imgsz`（单一标量）缩放 bbox，天然假设输入为方形；
        - `Model.val()` 的通用默认值是 `rect=True`，会产生非方形 batch shape，从而导致 bbox 缩放错误、mAP 异常偏低；
        - 因此这里对 RTDETRMM 默认设置 `rect=False`（用户显式传入 rect 时尊重用户设置）。
        """
        self._raise_if_multimodal_build_required("val")
        from ultralytics.utils import LOGGER

        # --- ensure channel contract matches data.yaml before validation ---

        # --- ensure channel contract matches data.yaml before validation ---
        data_arg = kwargs.get("data")
        data_cfg = None
        if data_arg and isinstance(data_arg, str):
            try:
                from ultralytics.data.utils import check_det_dataset
                data_cfg = check_det_dataset(data_arg)
            except Exception:
                pass
        elif data_arg and isinstance(data_arg, dict):
            data_cfg = data_arg

        if data_cfg is not None:
            try:
                self.ensure_multimodal_model_channel_contract(
                    data_cfg=data_cfg,
                    requested_modality=kwargs.get("modality"),
                )
            except Exception as e:
                if kwargs.get("verbose"):
                    LOGGER.warning(f"Channel contract check skipped: {e}")

        kwargs.setdefault("rect", False)
        imgsz = int(kwargs.get("imgsz", self.overrides.get("imgsz", 640)))
        self._set_complexity_handshake(self.model, pending=True, emitted=False, direct_ingress=False)
        owner_emitted = self._log_canonical_complexity_summary(imgsz)
        self._set_complexity_handshake(
            kwargs,
            pending=not owner_emitted,
            emitted=owner_emitted,
            direct_ingress=not owner_emitted,
        )
        if not coco:
            standard = super().val(validator=validator, **kwargs)
            results_dict = getattr(standard, "results_dict", {}) or {}
            save_dir = getattr(standard, "save_dir", None)
            out = MMValResults(standard=standard, coco=None, results_dict=results_dict, save_dir=save_dir)
            self.metrics = out
            return out

        # COCO 验证器仅支持 PyTorch 模型
        self._check_is_pytorch_model()

        # COCO 验证默认参数（允许用户覆盖）；RT-DETR 仍保持默认 rect=False
        kwargs.setdefault("save_json", True)
        kwargs.setdefault("plots", True)
        kwargs.setdefault("conf", 0.05)
        kwargs.setdefault("mode", "cocoval")

        # 目录隔离：仅当用户未指定 save_dir/name 时写入默认 name
        if kwargs.get("save_dir", None) is None and kwargs.get("name", None) is None:
            kwargs["name"] = "val-coco"

        args = {**self.overrides, **kwargs, "mode": "cocoval"}
        self._set_complexity_handshake(
            args,
            pending=not owner_emitted,
            emitted=owner_emitted,
            direct_ingress=not owner_emitted,
        )

        from ultralytics.models.rtdetr.cocoval import RTDETRMMCOCOValidator

        validator = (validator or RTDETRMMCOCOValidator)(
            dataloader=None,
            save_dir=None,
            pbar=None,
            args=args,
            _callbacks=self.callbacks,
        )
        stats = validator(model=self.model)
        standard_metrics = getattr(validator, "metrics", None)
        if standard_metrics is None:
            raise RuntimeError("RTDETRMM.val(coco=True): validator 未生成 metrics，无法构建 MMValResults。")

        save_dir = getattr(standard_metrics, "save_dir", None)
        coco_obj = COCOValResults.from_stats_dict(stats, save_dir=save_dir)
        out = MMValResults(standard=standard_metrics, coco=coco_obj, results_dict=stats, save_dir=save_dir)
        self.metrics = out
        return out

    def cocoval(self, validator=None, **kwargs: Any):
        """
        使用 COCO 评估指标对 RTDETRMM 进行验证。

        说明：
        - 返回包含 `metrics/coco/*` 的指标字典（12 项 COCO 标准指标为主）；
        - 默认 `rect=False`，避免 RT-DETR 在非方形输入下因 bbox 缩放假设导致指标异常（用户显式传入 rect 时尊重用户设置）。
        """
        res = self.val(validator=validator, coco=True, **kwargs)
        results = res.results_dict
        # 兼容旧行为：cocoval 结束后 self.metrics 为 dict
        self.metrics = results
        return results

    # ------------------------------------------------------------------
    # 结构化剪枝与剪枝后微调（对齐 YOLOMM 的 prune / finetrain 闭环）
    # ------------------------------------------------------------------

    # Deprecated: the zero-registration engine discovers structural constraints
    # (AIFI embed fixation, decoder hd coupling) automatically — correctness no
    # longer depends on user type lists. Kept one release for compatibility.
    _PRUNE_DEFAULT_PRUNE_TYPES = ("ConvNormLayer", "BasicBlock", "Blocks")
    _PRUNE_DEFAULT_SKIP_TYPES = (
        "AIFI", "RepC3", "Conv", "Upsample", "Concat",
        "RTDETRDecoder", "RTDETROBB",
    )

    def prune(self, method: str = "l1", ratio: float = 0.3, save_dir: Optional[str] = None,
              imgsz: int = 256, round_to: int = 8, min_ch: int = 8,
              prune_types: Optional[List[str]] = None, skip_types: Optional[List[str]] = None,
              mode: str = "structured", scope: Optional[List[int]] = None,
              save_report: Union[bool, str, Path] = False):
        """对 RTDETRMM 执行零注册引擎剪枝。

        依赖发现完全自动：AIFI 的 embed 固化、decoder 宽度耦合、denoising
        embedding 等结构约束由引擎的算子级追踪与围栏自动保护（保守跳过并在
        报告中说明），不再依赖类型白名单保命。``prune_types``/``skip_types``
        降级为用户范围偏好（deprecated，一个版本后移除）。

        剪枝后可直接 ``finetrain()``（内存直通，单卡流程）；``save_dir`` 落盘
        ``pruned.pt`` / ``pruned.yaml`` / ``prune_spec.json`` 并执行出厂自校验。

        Returns:
            self（可链式调用）。
        """
        from ultralytics.nn.mm.pruning.api import run_structured_prune
        from ultralytics.utils import LOGGER

        if mode != "structured":
            raise ValueError(f"prune(mode='{mode}') — structured only; use prune_sparse() for masks.")
        if prune_types is not None or skip_types is not None:
            LOGGER.warning(
                "RTDETRMM.prune: prune_types/skip_types 已降级为范围偏好（正确性由引擎结构约束保证），"
                "将在后续版本移除；建议改用 scope 参数。"
            )
        run_structured_prune(
            self, method=method, ratio=ratio, save_dir=save_dir, imgsz=imgsz,
            round_to=round_to, min_ch=min_ch, prune_types=prune_types,
            skip_types=skip_types, scope=scope, save_report=save_report,
        )
        return self

    def prune_sparse(self, sparsity: float = 0.5, pattern: str = "unstructured",
                     save_dir: Optional[str] = None):
        """权重级稀疏（unstructured / 2:4 semi-structured），结构不变，
        finetrain 时 mask 在 optimizer_step 内（EMA 前）重贴。"""
        from ultralytics.nn.mm.pruning.api import run_sparse_prune

        run_sparse_prune(self, sparsity=sparsity, pattern=pattern, save_dir=save_dir)
        return self

    def finetrain(self, trainer=None, **kwargs):
        """对剪枝后的 RTDETRMM 模型执行微调训练。

        仅接受从带 ``prune_info = {"is_pruned": True}`` 的 .pt 权重加载出的
        RTDETRMM 实例（该标记由 ``prune(save_dir=...)`` 写入）。

        支持与 ``train()`` 相同的常用训练参数（data/epochs/batch/imgsz/device 等），
        但显式拒绝重建结构的参数（scale / model_scale）—— 剪枝后结构必须直接来自
        checkpoint。

        Args:
            trainer: 自定义训练器类。默认按 task_map 包装对应训练器（detect / obb）。
            **kwargs: 训练配置（同 train()）。

        Returns:
            训练指标（如有），否则 None。
        """
        self._check_is_pytorch_model()

        if not isinstance(self.ckpt, dict):
            raise ValueError(
                "finetrain 仅支持从带 prune_info 的剪枝后 .pt 权重加载出的 RTDETRMM 实例。"
            )

        prune_info = self.ckpt.get("prune_info")
        if not isinstance(prune_info, dict) or prune_info.get("is_pruned") is not True:
            raise ValueError(
                'finetrain 仅接受带 `prune_info={"is_pruned": True}` 的剪枝权重。'
            )

        from ultralytics.cfg import TASK2DATA
        from ultralytics.utils import DEFAULT_CFG_DICT, RANK, YAML, checks
        from ultralytics.nn.mm.pruning.trainer import make_pruned_trainer_class
        from ultralytics.nn.tasks import attempt_load_one_weight

        checks.check_pip_update_available()
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        model_ref = (
            self.overrides.get("model")
            or getattr(self.model, "pt_path", "")
            or ""
        )
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": model_ref,
            "task": self.task,
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}

        trainer_cls = trainer or make_pruned_trainer_class(self.task_map[self.task]["trainer"])
        finetrain_trainer = trainer_cls(overrides=args, _callbacks=self.callbacks)

        # 传播完整 prune_info，使保存的权重保留完整溯源信息
        finetrain_trainer.prune_info = prune_info

        # 直接使用已加载的剪枝后模型 —— 不从 YAML 重建
        finetrain_trainer.model = self.model

        finetrain_trainer.hub_session = self.session
        self.trainer = finetrain_trainer
        self.trainer.train()

        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics
