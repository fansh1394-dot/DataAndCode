"""RTDETRMM OBB Trainer module for rotated bounding box training."""

import torch
from copy import copy
from typing import Optional

from ultralytics.nn.tasks import RTDETRMMOBBModel
from ultralytics.utils import LOGGER, DEFAULT_CFG
from ultralytics.utils.patches import torch_load
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    sync_x_modality_name_to_model,
)

from .train import RTDETRMMTrainer


def _set_obb_task_on_model(model):
    """Persist OBB task identity on a checkpoint model-like object."""
    if model is None:
        return
    model.task = "obb"
    args = getattr(model, "args", None)
    if isinstance(args, dict):
        args["task"] = "obb"
    elif args is not None and hasattr(args, "__dict__"):
        setattr(args, "task", "obb")


def _mark_obb_checkpoint_task(ckpt):
    """Ensure checkpoint metadata restores through the RTDETRMM OBB task path."""
    for key in ("model", "ema"):
        _set_obb_task_on_model(ckpt.get(key))
    train_args = ckpt.get("train_args")
    if isinstance(train_args, dict):
        train_args["task"] = "obb"
    ckpt["task"] = "obb"
    return ckpt


class RTDETRMMOBBTrainer(RTDETRMMTrainer):
    """RTDETRMM trainer for OBB task.

    Reuses multimodal dataset/preprocess/save/AFSS/distill behavior from RTDETRMMTrainer.
    Overrides task identity, model construction, validator, and displayed loss names.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        if overrides.get("distill"):
            raise NotImplementedError(
                "RTDETRMM OBB distill is disabled for now. Set distill=None before training OBB models."
            )
        if overrides.get("afss_enabled", False):
            raise NotImplementedError(
                "RTDETRMM OBB AFSS is disabled for now. Set afss_enabled=False before training OBB models."
            )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.args.task = "obb"
        self.loss_names = ("giou_loss", "cls_loss", "l1_loss", "angle_loss")

    def get_model(self, cfg: Optional[dict] = None, weights: Optional[str] = None, verbose: bool = True):
        from ultralytics.utils import RANK
        from ultralytics.nn.mm.utils import prepare_multimodal_build_cfg

        x_channels = self.data.get("Xch", 3)
        channels = 3 + x_channels
        self.data["channels"] = channels

        cfg_in = cfg or self.args.model
        cfg_dict = prepare_multimodal_build_cfg(cfg_in, self.data, self.args, LOGGER)
        model = RTDETRMMOBBModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        model.task = "obb"
        model.args = getattr(model, "args", {})
        model.args["task"] = "obb"

        if weights:
            model.load(weights)
        if hasattr(model, "multimodal_router") and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(getattr(self, "data", None)))
        return model

    def get_validator(self):
        self.loss_names = ("giou_loss", "cls_loss", "l1_loss", "angle_loss")
        from .val_obb import RTDETRMMOBBValidator

        validator = RTDETRMMOBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        validator.data = self.data
        return validator

    def save_model(self):
        super().save_model()
        for path in [self.last, self.best]:
            if not path.exists():
                continue
            ckpt = torch_load(path, map_location="cpu")
            _mark_obb_checkpoint_task(ckpt)
            torch.save(ckpt, path)
