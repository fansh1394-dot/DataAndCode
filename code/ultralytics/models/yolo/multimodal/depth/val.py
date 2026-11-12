# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn.functional as F

from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.metrics import DepthMetrics


class MultiModalDepthValidator(MultiModalDetectionValidator):
    """Validator for YOLOMM dense metric depth. No NMS; parent detect CSV/plots are disabled."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar=None, args=args, _callbacks=_callbacks)
        self.args.task = "depth"
        self.args.rect = False
        self.args.save_csv = False  # parent independent val writes detect CSV via metrics.box
        self.metrics = DepthMetrics()
        self._target_shape = None

    def __call__(self, trainer=None, model=None):
        # Parent __call__ plots via these methods and save_validation_csv(..., "detect").
        self.args.task = "depth"
        self.args.rect = False
        self.args.save_csv = False
        return super().__call__(trainer=trainer, model=model)

    def build_dataset(self, img_path, mode="val", batch=None):
        """Copy MultiModalDetectionValidator.build_dataset, only force rect=False."""
        self.args.rect = False
        if self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
        ds_kwargs = dict(
            cfg=self.args,
            img_path=img_path,
            batch=batch,
            data=self.data,
            mode=mode,
            rect=False,
            stride=self.stride if hasattr(self, "stride") and self.stride else 32,
            multi_modal_image=True,
            precision_report_dir=self.save_dir,
        )
        if self.multimodal_config:
            from ultralytics.nn.mm.utils import resolve_modality_pair

            modalities = self.multimodal_config["models"]
            modalities_dict = self.multimodal_config["modalities"]
            _rgb, x_modality = resolve_modality_pair(modalities, strict=True)
            ds_kwargs["x_modality"] = x_modality
            ds_kwargs["x_modality_dir"] = modalities_dict.get(x_modality) if x_modality else None
            ds_kwargs["modalities"] = modalities
        return build_yolo_dataset(**ds_kwargs)

    def init_metrics(self, model):
        from ultralytics.nn.mm.utils import get_runtime_x_modality_name

        data = self.data if isinstance(self.data, dict) else {}
        self.metrics = DepthMetrics(
            min_depth=float(data.get("min_depth", getattr(self.args, "min_depth", 0.001))),
            max_depth=float(data.get("max_depth", getattr(self.args, "max_depth", 100.0))),
            align=str(data.get("align", "median")),
        )
        self.seen = 0
        self.names = {0: "depth"}
        self.nc = 1
        self.x_modality_name = get_runtime_x_modality_name(model)

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
        )
        batch["depth"] = batch["depth"].to(self.device, non_blocking=True).float()
        self._target_shape = tuple(batch["depth"].shape[-2:])
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["depth"]
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if preds.shape[-2:] != self._target_shape:
            preds = F.interpolate(preds, size=self._target_shape, mode="bilinear", align_corners=True)
        return preds

    def update_metrics(self, preds, batch):
        self.metrics.update_stats(preds, batch["depth"])
        self.seen += preds.shape[0]

    def get_stats(self):
        self.metrics.process()
        return self.metrics.results_dict

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.process()

    def get_desc(self):
        return "Depth val"

    def print_results(self):
        self.metrics.process()
        m = self.metrics
        LOGGER.info(f"\n{colorstr('Depth metrics:')}")
        LOGGER.info(
            f"delta1={m.delta1:.4f} abs_rel={m.results_dict.get('metrics/abs_rel', 0):.4f} "
            f"rmse={m.results_dict.get('metrics/rmse', 0):.4f} silog={m.results_dict.get('metrics/silog', 0):.4f} "
            f"images={self.seen}"
        )

    def _plot_rgb_x(self, batch, ni, source, depth_maps=None):
        """RGB+X images with optional metric-depth overlay. Never pass dense depth as detection preds."""
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import extract_precision_contract_from_batch

        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3
        B = batch["img"].shape[0]
        device = batch["img"].device
        plotter = MultiModalPlotter(self.save_dir, on_plot=self.on_plot)
        plotter.plot_batch(
            stage="val",
            source=source,
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
            names=self.names,
            batch_ni=ni,
            depth_maps=depth_maps,
        )

    def plot_val_samples(self, batch, ni):
        self._plot_rgb_x(batch, ni, source="labels", depth_maps=batch.get("depth"))

    def plot_predictions(self, batch, preds, ni):
        # preds is dense depth [B,1,H,W]; parent output_to_target would crash
        self._plot_rgb_x(batch, ni, source="pred", depth_maps=preds)

    def afss_score_sample(self, pred, batch, si):
        raise RuntimeError("AFSS is not supported for task=depth")

    def afss_score_batch(self, preds, batch):
        raise RuntimeError("AFSS is not supported for task=depth")
