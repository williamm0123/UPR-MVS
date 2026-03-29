from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .edgeconv import EdgeConvBlock
from .knn import build_knn_graph


class PredictionHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        hidden_channels = max(in_channels, 128)
        super().__init__(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )


class EdgeConvPointRefiner(nn.Module):
    """One-shot point refinement with a static pixel-aware neighborhood graph."""

    def __init__(
        self,
        feature_dim: int,
        layer_dims: Sequence[int] = (128, 192, 256),
        knn: int = 16,
        candidate_k: int = 64,
        delta_scale: float = 0.01,
        use_checkpoint: bool = False,
        predict_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.knn = int(knn)
        self.candidate_k = int(candidate_k)
        self.delta_scale = float(delta_scale)
        self.use_checkpoint = use_checkpoint
        self.predict_uncertainty = bool(predict_uncertainty)

        block_dims = [int(feature_dim), *[int(dim) for dim in layer_dims]]
        self.blocks = nn.ModuleList(
            [EdgeConvBlock(in_channels=block_dims[idx], out_channels=block_dims[idx + 1]) for idx in range(len(block_dims) - 1)]
        )

        fusion_dim = int(feature_dim) + sum(int(dim) for dim in layer_dims)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.delta_head = PredictionHead(fusion_dim, 3)
        self.sigma_head = PredictionHead(fusion_dim, 1) if self.predict_uncertainty else None
        self.alpha_head = PredictionHead(fusion_dim, 1)

    def _run_block(self, block: nn.Module, points: Tensor, features: Tensor, neighbor_idx: Tensor) -> Tensor:
        if self.use_checkpoint and self.training and features.requires_grad:
            return checkpoint(lambda p, f, n: block(p, f, n), points, features, neighbor_idx, use_reentrant=False)
        return block(points, features, neighbor_idx)

    def forward(
        self,
        points: Tensor,
        point_feat: Tensor,
        pixel_coords: Tensor,
        point_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if point_feat.ndim != 3:
            raise ValueError(f"Expected point_feat with shape [B, N, C], got {tuple(point_feat.shape)}")

        neighbor_idx = build_knn_graph(
            points=points,
            pixel_coords=pixel_coords,
            k=self.knn,
            point_mask=point_mask,
            candidate_k=self.candidate_k,
        )

        features = point_feat
        shortcut_features = [point_feat]
        for block in self.blocks:
            features = self._run_block(block, points, features, neighbor_idx)
            shortcut_features.append(features)

        fused = self.fusion(torch.cat(shortcut_features, dim=-1))
        delta_p = torch.tanh(self.delta_head(fused)) * self.delta_scale
        if self.predict_uncertainty and self.sigma_head is not None:
            log_sigma = self.sigma_head(fused).clamp(min=-6.0, max=2.0)
            sigma = torch.exp(log_sigma)
        else:
            log_sigma = torch.zeros((*fused.shape[:2], 1), device=fused.device, dtype=fused.dtype)
            sigma = torch.ones_like(log_sigma)
        alpha = torch.sigmoid(self.alpha_head(fused))

        if point_mask is not None:
            mask = point_mask.float()
            delta_p = delta_p * mask
            log_sigma = log_sigma * mask
            sigma = torch.where(mask > 0.5, sigma, torch.ones_like(sigma))
            alpha = alpha * mask
            fused = fused * mask

        return {
            "delta_p": delta_p,
            "log_sigma": log_sigma,
            "sigma": sigma,
            "alpha": alpha,
            "refined_feat": fused,
            "neighbor_idx": neighbor_idx,
        }
