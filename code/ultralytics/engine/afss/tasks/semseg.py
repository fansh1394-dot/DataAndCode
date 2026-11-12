# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseAFSSTaskAdapter


@BaseAFSSTaskAdapter.register("semseg")
class SemSegAFSSTaskAdapter(BaseAFSSTaskAdapter):
    """No-op AFSS adapter for semseg until dense-mask sample scoring is implemented."""

    task_name = "semseg"
    required_validator_methods = ()

    def build_scoring_dataloader(self, trainer: Any, validator: Any):
        """Return an empty scoring iterable; semseg scoring is intentionally not implemented yet."""
        return ()

    def score_epoch(self, trainer: Any, validator: Any, dataloader: Any) -> List[Dict[str, Any]]:
        """TODO: implement dense semantic-mask sufficiency scoring."""
        return []

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """TODO: implement semseg per-sample score rows."""
        return []
