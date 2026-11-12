# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.nn.tasks import SemSegModel
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.patches import torch_load
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    set_single_modal_build_metadata_in_ckpt,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
)
from .val import MultiModalSemSegValidator


class MultiModalSemSegTrainer(MultiModalDetectionTrainer):
    """Trainer for YOLOMM semantic segmentation.

    Inherits MultiModalDetectionTrainer for modality ablation + AFSS + distill.
    Overrides only the task-specific methods below.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = overrides or {}
        overrides["task"] = "semseg"
        self._semseg_class_weights_applied = False
        super().__init__(cfg, overrides, _callbacks)

    def _ensure_semseg_criterion(self):
        """确保 SemSegLoss 实例创建并绑定到 self.model.criterion。

        Returns:
            SemSegLoss: 后续训练 loss 与 class weights 共用的实际 criterion 对象。
        """
        criterion = getattr(self.model, "criterion", None)
        if criterion is None:
            criterion = self.model.init_criterion()
            self.model.criterion = criterion
        return criterion

    def build_dataset(self, img_path, mode="train", batch=None):
        self.args.ignore_index = int(self.data.get("ignore_index", getattr(self.args, "ignore_index", 255)))
        x_modality = self._determine_x_modality_from_data()
        x_dir = self._get_x_modality_path(x_modality)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,  # SemSeg 不需要矩形批处理，mask 像素严格对齐
            stride=gs,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
            precision_report_dir=self.save_dir,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        x_channels = self.data.get("Xch", 3)
        channels = 3 + x_channels
        from ultralytics.nn.mm.utils import prepare_multimodal_build_cfg

        cfg_dict = prepare_multimodal_build_cfg(cfg, self.data, self.args, LOGGER)
        model = SemSegModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        if model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )
        if weights:
            model.load(weights)
        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(self.data))
        if verbose and RANK in {-1, 0}:
            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
                log_default_complexity(model, report, LOGGER)
            except Exception as e:
                LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")
        return model

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
        )
        batch["sem_mask"] = batch["sem_mask"].to(self.device, non_blocking=True).long()
        return batch

    def get_validator(self):
        criterion = self._ensure_semseg_criterion()
        if not self._semseg_class_weights_applied:
            self.set_class_weights(criterion=criterion)
            self._semseg_class_weights_applied = True
        self.loss_names = getattr(criterion, "names", ("semseg_ce_loss", "semseg_dice_loss", "semseg_aux_loss"))
        return MultiModalSemSegValidator(
            dataloader=self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def progress_string(self):
        """Return SemSeg training progress columns with compact loss display names."""
        return ("\n" + "%11s" * 7) % (
            "Epoch",
            "GPU_mem",
            "ce_loss",
            "dice_loss",
            "aux_loss",
            "Instances",
            "Size",
        )

    def save_model(self):
        """Save checkpoints with BaseTrainer lifecycle plus SemSeg metadata."""
        super().save_model()

        targets = [self.last]
        if self.best.exists():
            targets.append(self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            epoch_pt = self.wdir / f"epoch{self.epoch}.pt"
            if epoch_pt.exists():
                targets.append(epoch_pt)

        x_modality_name = getattr(self.model, "x_modality_name", None)
        for path in targets:
            ckpt = torch_load(path, map_location="cpu")
            ckpt["yaml"] = self.model.yaml
            ckpt["nc"] = self.data["nc"]
            ckpt["names"] = self.data["names"]
            ckpt["Xch"] = self.data.get("Xch", 3)
            ckpt["ignore_index"] = self.args.ignore_index
            set_x_modality_name_in_ckpt_metadata(ckpt, x_modality_name)
            set_single_modal_build_metadata_in_ckpt(ckpt, model=self.model, args=self.args)
            torch.save(ckpt, path)

    def plot_metrics(self):
        from ultralytics.utils.plotting import plot_distill_results
        plot_results(file=self.csv, semseg=True, on_plot=self.on_plot)
        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="yolomm", on_plot=self.on_plot)

    def _sync_afss_task_name(self):
        self.args.afss_task_name = "semseg"

    def plot_training_samples(self, batch, ni):
        """Plot multimodal semseg training samples — 薄包装，委托 MultiModalPlotter。

        SemSeg 无 bbox：传占位 batch_idx/cls/bboxes，GT 由 sem_masks=batch["sem_mask"] 承载。
        支持双模态（RGB / X / 并排）与模态消融（单模态）。
        """
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
        )

        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3
        B = batch["img"].shape[0]
        device = batch["img"].device

        plotter = MultiModalPlotter(self.save_dir, on_plot=self.on_plot)
        plotter.plot_batch(
            stage="train", source="labels", images=batch["img"],
            batch_idx=torch.zeros(B, dtype=torch.long, device=device),
            cls=torch.zeros(0, dtype=torch.long, device=device),
            bboxes=torch.zeros((0, 4), device=device),
            xch=xch,
            x_modality_name=MultiModalPlotter.resolve_x_modality_name(
                getattr(self, "modality", None),
                getattr(self, "data", None),
                getattr(self, "x_modality_name", None),
            ),
            precision_contract=extract_precision_contract_from_batch(batch),
            modality_mode=MultiModalPlotter.determine_modality_mode(
                getattr(self, "modality", None)
            ),
            sem_masks=batch["sem_mask"],
            paths=batch["im_file"], names=getattr(self, "names", None), batch_ni=ni,
            threaded=True,  # 异步绘图，恢复旧版 @threaded 行为
        )

    def set_class_weights(self, criterion=None):
        """Compute per-class pixel frequency weights from training masks and set to SemSegLoss.

        Scans up to 1000 mask files from the training dataset, accumulates per-class
        pixel counts, and computes inverse-frequency weights (median normalization).
        Passes the weights to SemSegLoss.set_class_weights for balanced CE loss.
        Skipped for binary segmentation (nc=1) which uses BCEWithLogitsLoss.
        Also supports loading pre-defined weights from data.yaml `class_weights` field.
        """
        nc = self.data["nc"]
        if nc == 1:
            return

        criterion = criterion or self._ensure_semseg_criterion()

        # Check for pre-defined class_weights in data.yaml
        class_weights_cfg = self.data.get("class_weights", None)
        if class_weights_cfg is not None:
            if isinstance(class_weights_cfg, (list, tuple)):
                weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
            elif isinstance(class_weights_cfg, dict):
                weights = torch.zeros(nc, dtype=torch.float32)
                for cls_id, w in class_weights_cfg.items():
                    weights[int(cls_id)] = float(w)
            elif isinstance(class_weights_cfg, str):
                from ultralytics.utils import yaml_load
                import json
                path = Path(class_weights_cfg)
                if path.suffix in {".yaml", ".yml"}:
                    data = yaml_load(path)
                    weights = torch.tensor(data.get("class_weights", data), dtype=torch.float32)
                elif path.suffix == ".json":
                    with open(path) as f:
                        data = json.load(f)
                    weights = torch.tensor(data.get("class_weights", data), dtype=torch.float32)
                else:
                    LOGGER.warning(f"Unsupported class_weights file format: {path.suffix}, skipping")
                    return
            else:
                LOGGER.warning(f"Unsupported class_weights type: {type(class_weights_cfg)}, skipping")
                return

            if weights.numel() != nc:
                LOGGER.warning(f"class_weights length {weights.numel()} does not match nc={nc}, skipping")
                return

            LOGGER.info(f"Loaded pre-defined class weights from data.yaml: {weights.tolist()}")
            criterion.set_class_weights(weights)
            return

        # Compute weights from training mask pixel frequencies
        dataset = self.train_loader.dataset
        mask_files = getattr(dataset, "sem_files", None) or getattr(dataset, "mask_files", [])
        if not mask_files:
            LOGGER.warning("No semantic mask files found, skipping class weight computation")
            return

        pixel_counts = np.zeros(nc, dtype=np.int32)
        sample_size = min(1000, len(mask_files))
        indices = np.linspace(0, len(mask_files) - 1, sample_size).astype(int)

        ignore_index = int(self.data.get("ignore_index", getattr(self.args, "ignore_index", 255)))
        for idx in indices:
            try:
                mask = np.array(PILImage.open(mask_files[idx]))
            except Exception:
                continue
            if hasattr(dataset, "label_mapping") and dataset.label_mapping:
                for old, new in dataset.label_mapping.items():
                    mask[mask == old] = new
            valid = (mask >= 0) & (mask < nc) & (mask != ignore_index)
            if valid.any():
                classes, counts = np.unique(mask[valid], return_counts=True)
                for c, count in zip(classes, counts):
                    pixel_counts[int(c)] += int(count)

        if pixel_counts.sum() == 0:
            LOGGER.warning("No valid pixels found for class weight computation, skipping")
            return

        # Inverse frequency weights with median normalization
        freq = pixel_counts.astype(np.float64)
        freq[freq == 0] = freq[freq > 0].mean()  # avoid inf for absent classes
        weights = torch.tensor(1.0 / freq, dtype=torch.float32)
        weights = weights / weights.median()  # normalize by median
        LOGGER.info(f"Computed class weights from pixel frequencies: {weights.tolist()}")

        criterion.set_class_weights(weights)
