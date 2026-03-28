from __future__ import annotations

from typing import Any

from torch import nn

from models.upr_mvs_transformer import UPRMVSTransformerModel


class UPRMVSModel(UPRMVSTransformerModel):
    """Backward-compatible model entry that now uses the transformer pipeline."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__(model_cfg=model_cfg)


class CoarseDepthStageModel(nn.Module):
    """Deprecated shim retained for training script compatibility."""

    def __init__(self, backbone_cfg: dict[str, Any], coarse_cfg: dict[str, Any]) -> None:  # noqa: ARG002
        super().__init__()
        raise RuntimeError("Coarse-only stage was removed; use transformer model config with model.backbone=dinov3.")
