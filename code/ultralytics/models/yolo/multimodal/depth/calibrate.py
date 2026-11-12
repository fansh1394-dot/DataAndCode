# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn.functional as F

from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DepthMetrics


def _prep_calibrate_batch(batch, device):
    img = batch["img"].to(device)
    raw_dtype = img.dtype
    img = apply_dataset_precision_contract(
        img.float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
    )
    gt = batch["depth"].to(device).float()
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    return img, gt


def _predict_depth(model, img, gt):
    pred = model(img)
    if isinstance(pred, dict):
        pred = pred["depth"]
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)
    return pred


def _write_cal_buffers(head, a: float, b: float) -> None:
    """Write cal buffers outside InferenceMode (final_eval wraps validator in it)."""
    a_t = torch.tensor([float(a)], device=head.cal_a.device, dtype=head.cal_a.dtype)
    b_t = torch.tensor([float(b)], device=head.cal_b.device, dtype=head.cal_b.dtype)
    head.cal_a.copy_(a_t)
    head.cal_b.copy_(b_t)


@torch.no_grad()
def calibrate_depth_scale(model, dataloader, device, min_depth=0.001, max_depth=100.0, margin=0.002):
    """Scale-only log calibration. Writes cal_a=1, cal_b if raw δ1 beats identity by margin."""
    with torch.inference_mode(False), torch.no_grad():
        return _calibrate_depth_scale_impl(model, dataloader, device, min_depth, max_depth, margin)


def _calibrate_depth_scale_impl(model, dataloader, device, min_depth, max_depth, margin):
    model.eval()
    head = model.model[-1]
    identity = (1.0, 0.0)
    # Eval forward applies cal_a/cal_b; measure raw δ1 from identity buffers.
    if hasattr(head, "cal_a") and hasattr(head, "cal_b"):
        _write_cal_buffers(head, 1.0, 0.0)
    logs = []
    for batch in dataloader:
        img, gt = _prep_calibrate_batch(batch, device)
        pred = _predict_depth(model, img, gt)
        for pi, gi in zip(pred, gt):
            mask = (gi > min_depth) & (gi < max_depth)
            if int(mask.sum()) < 10:
                continue
            logs.append((torch.log(gi[mask].clamp(min=min_depth)) - torch.log(pi[mask].clamp(min=min_depth))).mean())
    if len(logs) < 2:
        LOGGER.warning("Depth calibrate skipped: fewer than 2 valid images")
        return identity

    n = len(logs)
    mid = n // 2
    folds = [logs[:mid], logs[mid:]]
    b_vals = [float(torch.stack(fold).mean()) for fold in folds if fold]
    cal_b = float(sum(b_vals) / len(b_vals))

    def score(a, b):
        m = DepthMetrics(min_depth=min_depth, max_depth=max_depth, align="none")
        for batch in dataloader:
            img, gt = _prep_calibrate_batch(batch, device)
            pred = _predict_depth(model, img, gt)
            pred = pred.pow(a) * torch.exp(torch.tensor(b, device=pred.device, dtype=pred.dtype))
            m.update_stats(pred, gt)
        m.process()
        return m.delta1

    d_id = score(1.0, 0.0)
    d_sc = score(1.0, cal_b)
    if d_sc > d_id + margin:
        _write_cal_buffers(head, 1.0, cal_b)
        LOGGER.info(f"Depth calibrate wrote scale-only cal_b={cal_b:.6f} (delta1 {d_id:.4f}->{d_sc:.4f})")
        return 1.0, cal_b
    LOGGER.info(f"Depth calibrate kept identity (delta1 scale={d_sc:.4f} identity={d_id:.4f})")
    return identity
