from __future__ import annotations

import torch
from torch import Tensor, nn

from upr_mvs.models.point.knn import build_knn_graph


class EdgeConvPointRefiner(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        layer_dims: tuple[int, ...] = (128, 192, 256),
        knn: int = 16,
        candidate_k: int = 64,
        delta_scale: float = 0.01,
        use_checkpoint: bool = False,
        predict_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        del use_checkpoint  # API compatibility
        self.knn = int(knn)
        self.candidate_k = int(candidate_k)
        self.delta_scale = float(delta_scale)
        self.predict_uncertainty = bool(predict_uncertainty)

        dims = [feature_dim] + [int(d) for d in layer_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.LayerNorm(dims[i + 1])])
        self.backbone = nn.Sequential(*layers)

        out_dim = dims[-1]
        self.delta_head = nn.Linear(out_dim, 3)
        self.sigma_head = nn.Linear(out_dim, 1)
        self.alpha_head = nn.Linear(out_dim, 1)

    def forward(self, points: Tensor, point_feat: Tensor, pixel_coords: Tensor, point_mask: Tensor) -> dict[str, Tensor]:
        neighbor_idx = build_knn_graph(
            points=points,
            pixel_coords=pixel_coords,
            k=self.knn,
            point_mask=point_mask,
            candidate_k=self.candidate_k,
        )
        refined_feat = self.backbone(point_feat)
        delta_p = torch.tanh(self.delta_head(refined_feat)) * self.delta_scale

        if self.predict_uncertainty:
            log_sigma = self.sigma_head(refined_feat).clamp(min=-6.0, max=2.0)
            sigma = torch.exp(log_sigma)
            alpha = torch.sigmoid(self.alpha_head(refined_feat))
        else:
            log_sigma = torch.zeros_like(point_feat[..., :1])
            sigma = torch.ones_like(point_feat[..., :1])
            alpha = torch.ones_like(point_feat[..., :1])

        mask_f = point_mask.float()
        return {
            "delta_p": delta_p * mask_f,
            "log_sigma": log_sigma * mask_f,
            "sigma": sigma * mask_f,
            "alpha": alpha * mask_f,
            "neighbor_idx": neighbor_idx,
            "refined_feat": refined_feat * mask_f,
        }
