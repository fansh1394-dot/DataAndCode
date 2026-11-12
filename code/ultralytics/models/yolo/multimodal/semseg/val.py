# Ultralytics YOLO, AGPL-3.0 license

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.metrics import SemSegMetrics


class MultiModalSemSegValidator(MultiModalDetectionValidator):
    """Validator for YOLOMM semantic segmentation with bincount-based metrics."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar=None, args=args, _callbacks=_callbacks)
        self.args.task = "semseg"
        self.metrics = SemSegMetrics()
        self._target_shape = None
        self.results_dir = None

    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(self.names)
        self.metrics = SemSegMetrics(names=self.names)
        self.seen = 0
        self.dataset = getattr(self.dataloader, "dataset", None)
        if self.args.save_json:
            self.results_dir = self.save_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
        )
        batch["sem_mask"] = batch["sem_mask"].to(self.device, non_blocking=True).long()
        self._target_shape = tuple(batch["sem_mask"].shape[-2:])
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch

    def postprocess(self, preds):
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if preds.shape[-2:] != self._target_shape:
            preds = F.interpolate(preds, size=self._target_shape, mode="bilinear", align_corners=False)
        if self.nc > 1:
            return preds.argmax(dim=1).to(torch.int32)
        else:
            return preds.gt(0).squeeze(1).to(torch.int32)

    def update_metrics(self, preds, batch):
        if self.args.save_json:
            self._save_pred_masks(preds, batch)
        self.metrics.update_stats(preds, batch["sem_mask"])
        self.seen += preds.shape[0]

    def get_stats(self):
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        return self.metrics.results_dict

    def gather_stats(self):
        """DDP: reduce confusion matrix to rank 0."""
        import torch.distributed as dist
        if RANK == -1 or not dist.is_available() or not dist.is_initialized():
            return
        if self.metrics.matrix is None:
            cm_nc = self.metrics.cm_nc
            self.metrics.matrix = torch.zeros((cm_nc, cm_nc), device=self.device, dtype=torch.float32)
        dist.reduce(self.metrics.matrix, dst=0, op=dist.ReduceOp.SUM)
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.nt_per_image, gathered, dst=0)
            self.metrics.nt_per_image = np.sum(gathered, axis=0)
        elif RANK > 0:
            dist.gather_object(self.metrics.nt_per_image, None, dst=0)

    def get_desc(self):
        return "SemSeg val"

    def print_results(self):
        stats = self.metrics
        LOGGER.info(f"\n{colorstr('SemSeg metrics:')}")
        LOGGER.info(
            f"mIoU={stats.miou:.4f}, mPA={stats.mpa:.4f}, "
            f"mDice={stats.mdice:.4f}, PA={stats.pixel_accuracy:.4f}"
        )
        LOGGER.info(("%22s" + "%11s" * 5) % ("Class", "Images", "Pixels", "IoU", "PA", "Dice"))
        summary = stats.summary()
        if summary:
            for row in summary:
                LOGGER.info(
                    ("%22s" + "%11d" * 2 + "%11.4f" * 3) % (
                        row["Class"], row["Images"], row["Pixels"], row["IoU"], row["PA"], row["Dice"]
                    )
                )
        if self.args.save_json and self.results_dir:
            LOGGER.info(f"Semantic prediction masks saved to {self.results_dir}")

    def _save_pred_masks(self, preds, batch):
        if self.results_dir is None:
            return
        im_files = batch.get("im_file", [])
        ori_shapes = batch.get("ori_shape", [])
        ratio_pads = batch.get("ratio_pad", [])
        if not im_files:
            return
        preds_np = preds.cpu().numpy()
        if self.dataset and hasattr(self.dataset, "label_mapping") and self.dataset.label_mapping:
            preds_np = self.dataset.convert_label(preds_np, inverse=True)
        preds_np = preds_np.astype(np.uint8, copy=False)
        for pred, im_file, ori_shape, ratio_pad in zip(
            preds_np, im_files, ori_shapes, ratio_pads
        ):
            if ratio_pad is not None:
                ratio, (left, top) = ratio_pad
                h, w = ori_shape
                pred = pred[top:top + int(h / ratio[0]), left:left + int(w / ratio[1])]
            else:
                pred = cv2.resize(pred, tuple(reversed(ori_shape)), interpolation=cv2.INTER_NEAREST)
            save_path = self.results_dir / Path(im_file).with_suffix(".png").name
            Image.fromarray(pred).save(save_path)

    def finalize_metrics(self, *args, **kwargs):
        """Ensure metrics object exposes conf_matrix attribute for external access."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)

    def _semseg_visual_components(self, batch):
        """返回 SemSeg 可视化所需的 RGB/X display tensors 和路径元信息。

        Returns:
            tuple: (rgb_visual, x_visual, x_modality, paths, ignore_index)
        """
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            prepare_rgb_for_display,
            split_modalities,
            visualize_x_to_3ch,
            resolve_x_modality,
        )

        images = batch["img"]
        data = getattr(self, "data", None)
        data = data if isinstance(data, dict) else {}
        args = getattr(self, "args", None)

        # 训练期 BaseValidator.__call__ 会从 trainer 注入 self.data；独立 val 时
        # self.data 可能较晚才初始化，因此这里按 data -> args -> batch 通道数兜底。
        inferred_xch = max(1, int(images.shape[1]) - 3)
        xch = int(data.get("Xch", getattr(args, "Xch", inferred_xch)))
        rgb_images, x_images = split_modalities(images, xch)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        modality = getattr(self, "modality", getattr(args, "modality", None))
        x_modality = resolve_x_modality(modality, data)
        if x_modality == "unknown":
            x_modality = (
                getattr(self, "x_modality_name", None)
                or data.get("x_modality")
                or getattr(args, "x_modality", None)
                or "unknown"
            )
        x_visual = visualize_x_to_3ch(
            x_images,
            colorize=False,
            x_modality=x_modality,
            precision=precision_contract["x"],
        )
        ignore_index = int(data.get("ignore_index", getattr(args, "ignore_index", 255)))
        return rgb_visual, x_visual, x_modality, batch["im_file"], ignore_index

    def plot_val_samples(self, batch, ni):
        """绘制 SemSeg 验证 GT mask — 薄包装，委托 MultiModalPlotter。

        SemSeg 无 bbox：传占位 batch_idx/cls/bboxes，GT 由 sem_masks=batch["sem_mask"] 承载。
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
            stage="val", source="labels", images=batch["img"],
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
            paths=batch["im_file"], names=self.names, batch_ni=ni,
        )

    def plot_predictions(self, batch, preds, ni):
        """绘制 SemSeg 验证预测 mask — 薄包装，委托 MultiModalPlotter。

        preds 是 postprocess 后的 class-id tensor [B,H,W]，由 sem_masks=preds 传入。
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
            stage="val", source="pred", images=batch["img"],
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
            sem_masks=preds,
            paths=batch["im_file"], names=self.names, batch_ni=ni,
        )
