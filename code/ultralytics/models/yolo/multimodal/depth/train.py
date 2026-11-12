# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    prepare_multimodal_build_cfg,
    set_single_modal_build_metadata_in_ckpt,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
)
from ultralytics.nn.tasks import DepthModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import de_parallel


class MultiModalDepthTrainer(MultiModalDetectionTrainer):
    """Trainer for YOLOMM dense metric depth."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = overrides or {}
        overrides["task"] = "depth"
        overrides["mosaic"] = 0.0
        overrides["mixup"] = 0.0
        overrides["cutmix"] = 0.0
        overrides["copy_paste"] = 0.0
        overrides["rect"] = False
        # AFSS_SUPPORTED_TASKS has no depth; detect adapter would crash on dense maps.
        overrides["afss_enabled"] = False
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "depth"
        self.args.rect = False
        self.args.afss_task_name = "detect"

    def _sync_afss_task_name(self):
        # Parent AFSSConfig.from_args validates args.task first and requires
        # afss_task_name == task. Depth is unsupported; facade as detect, then
        # MultiModalDepthTrainer.__init__ restores args.task = "depth".
        self.args.afss_task_name = "detect"
        if str(getattr(self.args, "task", "")) == "depth":
            self.args.task = "detect"

    def build_dataset(self, img_path, mode="train", batch=None):
        x_modality = self._determine_x_modality_from_data()
        x_dir = self._get_x_modality_path(x_modality)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,
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
        cfg_dict = prepare_multimodal_build_cfg(cfg, self.data, self.args, LOGGER)
        model = DepthModel(cfg_dict, nc=self.data.get("nc", 1), ch=channels, verbose=verbose and RANK == -1)
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
                report = compute_default_multimodal_complexity_report(model, imgsz=int(getattr(self.args, "imgsz", 640)))
                log_default_complexity(model, report, LOGGER)
            except Exception as e:
                LOGGER.warning(f"模型复杂度统计失败（可忽略）：{e}")
        return model

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
        )
        batch["depth"] = batch["depth"].to(self.device, non_blocking=True).float()
        return batch

    def get_validator(self):
        from .val import MultiModalDepthValidator

        criterion = getattr(self.model, "criterion", None) or self.model.init_criterion()
        self.model.criterion = criterion
        self.loss_names = getattr(criterion, "names", ("dlog_loss", "dgrad_loss"))
        return MultiModalDepthValidator(
            dataloader=self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_labels(self):
        """Skip detect plot_labels: empty depth boxes make cls.max() raise."""
        LOGGER.info("Depth plot_training_labels: skip detect box plot")

    def plot_training_samples(self, batch, ni):
        """RGB+X images with GT depth overlay. Parent detect plotter is not used."""
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import extract_precision_contract_from_batch

        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3
        B = batch["img"].shape[0]
        device = batch["img"].device
        plotter = MultiModalPlotter(self.save_dir, on_plot=self.on_plot)
        plotter.plot_batch(
            stage="train",
            source="labels",
            images=batch["img"],
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
            modality_mode=MultiModalPlotter.determine_modality_mode(getattr(self, "modality", None)),
            paths=batch["im_file"],
            names=self.data.get("names", {0: "depth"}) if isinstance(getattr(self, "data", None), dict) else {0: "depth"},
            batch_ni=ni,
            depth_maps=batch.get("depth"),
            threaded=True,
        )

    def auto_batch(self):
        return self.args.batch

    def _depth_export_model(self):
        if getattr(self, "ema", None) is not None and getattr(self.ema, "ema", None) is not None:
            return de_parallel(self.ema.ema)
        return de_parallel(self.model)

    def save_model(self):
        super().save_model()
        targets = [self.last]
        if self.best.exists():
            targets.append(self.best)
        x_modality_name = getattr(self.model, "x_modality_name", None)
        export_model = self._depth_export_model()
        head = export_model.model[-1] if hasattr(export_model, "model") else None
        for path in targets:
            ckpt = torch_load(path, map_location="cpu")
            ckpt["yaml"] = self.model.yaml
            ckpt["nc"] = self.data.get("nc", 1)
            ckpt["names"] = self.data.get("names", {0: "depth"})
            ckpt["Xch"] = self.data.get("Xch", 3)
            ckpt["min_depth"] = self.data.get("min_depth", getattr(self.args, "min_depth", 0.001))
            ckpt["max_depth"] = self.data.get("max_depth", getattr(self.args, "max_depth", 100.0))
            set_x_modality_name_in_ckpt_metadata(ckpt, x_modality_name)
            set_single_modal_build_metadata_in_ckpt(ckpt, model=self.model, args=self.args)
            if hasattr(head, "cal_a") and hasattr(head, "cal_b"):
                cal = {
                    "cal_a": float(head.cal_a.reshape(-1)[0].item()),
                    "cal_b": float(head.cal_b.reshape(-1)[0].item()),
                }
                ckpt["depth_calibration"] = cal
                ema = ckpt.get("ema")
                ema_head = ema.model[-1] if ema is not None and hasattr(ema, "model") else None
                if ema_head is not None and hasattr(ema_head, "cal_a") and hasattr(ema_head, "cal_b"):
                    ema_head.cal_a.fill_(cal["cal_a"])
                    ema_head.cal_b.fill_(cal["cal_b"])
            torch.save(ckpt, path)

    def plot_metrics(self):
        plot_results(file=self.csv, depth=True, on_plot=self.on_plot)

    def final_eval(self):
        super().final_eval()
        try:
            from .calibrate import calibrate_depth_scale

            model = self._depth_export_model()
            model.eval()
            data = self.data if isinstance(self.data, dict) else {}
            calibrate_depth_scale(
                model,
                self.test_loader,
                self.device,
                min_depth=float(data.get("min_depth", getattr(self.args, "min_depth", 0.001))),
                max_depth=float(data.get("max_depth", getattr(self.args, "max_depth", 100.0))),
            )
            self.save_model()
        except Exception as e:
            LOGGER.warning(f"Depth calibrate failed: {e}")
