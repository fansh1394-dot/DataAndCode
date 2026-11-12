"""RTDETRMM OBB Validator module for rotated bounding box evaluation."""

import numpy as np
import torch

from ultralytics.utils import ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou

from .val import RTDETRMMValidator


class RTDETRMMOBBValidator(RTDETRMMValidator):
    """RTDETRMM validator for OBB predictions in xywhr format."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize RTDETRMM OBB validation state while preserving RTDETRMM runtime behavior."""
        if args is not None:
            if isinstance(args, dict):
                args["task"] = "obb"
            else:
                setattr(args, "task", "obb")
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics()
        self.is_dota = False

    def init_metrics(self, model):
        super().init_metrics(model)
        val = self.data.get(self.args.split, "") if isinstance(self.data, dict) else ""
        self.is_dota = isinstance(val, str) and "DOTA" in val

    def postprocess(self, preds):
        """Convert RTDETROBB [B, Q, 5 + nc] output into DetectionValidator prediction dicts."""
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        y = preds[0]
        bs, _, nd = y.shape
        if nd <= 5:
            raise ValueError(f"RTDETRMM OBB validator expects [B,Q,5+nc], got shape={tuple(y.shape)}")

        bboxes, scores = y.split((5, nd - 5), dim=-1)
        imgsz_hw = getattr(self, "_imgsz_hw", None)
        if imgsz_hw is not None and len(imgsz_hw) == 2:
            h, w = int(imgsz_hw[0]), int(imgsz_hw[1])
        else:
            h = w = int(self.args.imgsz)
        scale = torch.tensor([w, h, w, h, 1.0], device=bboxes.device, dtype=bboxes.dtype)
        bboxes = ops.regularize_rboxes((bboxes * scale).reshape(-1, 5)).reshape(bs, -1, 5)

        outputs = []
        for i in range(bs):
            conf, cls = scores[i].max(-1)
            keep = conf > self.args.conf
            if keep.any():
                order = conf[keep].argsort(descending=True)
                box = bboxes[i][keep][order]
                outputs.append({"bboxes": box, "conf": conf[keep][order], "cls": cls[keep][order].to(box.dtype)})
            else:
                outputs.append(
                    {
                        "bboxes": bboxes.new_zeros((0, 5)),
                        "conf": bboxes.new_zeros((0,)),
                        "cls": bboxes.new_zeros((0,)),
                    }
                )
        return outputs

    def _prepare_batch(self, si, batch):
        """Prepare native-space OBB ground truth without detect xyxy conversion."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bboxes = batch["bboxes"][idx].to(self.device)
        if bboxes.ndim != 2 or bboxes.shape[-1] != 5:
            raise ValueError(f"RTDETRMMOBBValidator expects xywhr [N,5], got {tuple(bboxes.shape)}")

        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bboxes = bboxes.clone()
            bboxes[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])
            bboxes[..., :4] = ops.scale_boxes(imgsz, bboxes[..., :4], ori_shape, ratio_pad=ratio_pad, xywh=True)
        return {"cls": cls, "bboxes": bboxes, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Scale postprocessed input-space OBB predictions to original image space."""
        cls = pred["cls"]
        if self.args.single_cls:
            cls = cls * 0

        bboxes = pred["bboxes"].clone()
        if len(bboxes):
            bboxes[:, :4] = ops.scale_boxes(
                pbatch["imgsz"],
                bboxes[:, :4],
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
                xywh=True,
            )
        return {"bboxes": bboxes, "conf": pred["conf"], "cls": cls}

    def _process_batch(self, preds, batch):
        """Evaluate OBB predictions with probabilistic IoU."""
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def afss_score_sample(self, pred, batch, si):
        """RTDETRMM OBB AFSS is intentionally disabled."""
        raise NotImplementedError("RTDETRMM OBB AFSS scoring is intentionally disabled.")

    def afss_score_batch(self, preds, batch):
        """RTDETRMM OBB AFSS is intentionally disabled."""
        raise NotImplementedError("RTDETRMM OBB AFSS scoring is intentionally disabled.")
