"""RT-DETR OBB target adapters."""

from typing import Any, Dict, Optional, Tuple

import torch


def split_rtdetr_obb_targets(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Split RT-DETR OBB targets into a 4D bbox view and angle targets.

    The external training contract is normalized xywhr in ``batch["bboxes"]`` with shape ``[N, 5]``.
    RT-DETR DN, encoder, and Hungarian matching remain 4D xywh logic, while angle supervision is isolated to the
    decoder OBB branches.
    """
    if "bboxes" not in batch:
        raise KeyError("RTDETR OBB targets require batch['bboxes']")

    bboxes = batch["bboxes"]
    if not isinstance(bboxes, torch.Tensor):
        raise TypeError(f"RTDETR OBB batch['bboxes'] must be torch.Tensor, got {type(bboxes).__name__}")
    if bboxes.ndim != 2 or bboxes.shape[-1] != 5:
        raise ValueError(f"RTDETR OBB expects batch['bboxes'] as xywhr [N,5], got shape={tuple(bboxes.shape)}")

    batch4 = dict(batch)
    batch4["bboxes"] = bboxes[:, :4].contiguous()
    gt_angles = bboxes[:, 4].contiguous()
    return batch4, gt_angles


def make_rtdetr_obb_dn_batch(batch: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the 4D batch consumed by ``get_cdn_group``; preserve ``None`` for inference."""
    if batch is None:
        return None
    batch4, _ = split_rtdetr_obb_targets(batch)
    return batch4
