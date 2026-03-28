from __future__ import annotations

import torch
from torch import Tensor

from upr_mvs.models.losses.consistency import masked_mean
from upr_mvs.models.point.knn import gather_by_index


def repulsion_loss(
    points: Tensor,
    neighbor_idx: Tensor,
    point_mask: Tensor,
    radius: float = 0.01,
) -> Tensor:
    neighbor_points = gather_by_index(points, neighbor_idx)  # [B, N, K, 3]
    central_points = points.unsqueeze(2).expand_as(neighbor_points)
    dist_sq = (neighbor_points - central_points).square().sum(dim=-1, keepdim=True)
    repulsion = torch.exp(-dist_sq / max(radius * radius, 1.0e-6))
    mask = point_mask.unsqueeze(2).expand_as(repulsion)
    repulsion = torch.where(mask, repulsion, torch.zeros_like(repulsion))
    return masked_mean(repulsion, mask)
