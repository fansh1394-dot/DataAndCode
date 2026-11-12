"""RTDETRMM OBB Predictor module for rotated bounding box inference."""

from ultralytics.engine.multimodal.predictor import MultiModalOBBPredictor
from ultralytics.utils import LOGGER

from .mm_predictor import RTDETRMMPredictor


class RTDETRMMOBBPredictor(RTDETRMMPredictor):
    """RTDETRMM OBB predictor adapter.

    The model emits [cx, cy, w, h, theta, class scores]. Downstream result conversion
    must preserve theta and use OBB result semantics.
    """

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose=verbose)
        if getattr(self.model, "task", None) != "obb":
            self.model.task = "obb"

        self.predictor = MultiModalOBBPredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            device=str(self.device),
            verbose=verbose,
            debug=debug or getattr(self.args, "debug", False),
            font_size=getattr(self.args, "font_size", None),
            show_filename=getattr(self.args, "show_filename", False),
        )
        if verbose:
            LOGGER.info("RTDETRMM OBB 推理引擎初始化成功")
