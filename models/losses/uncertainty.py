from __future__ import annotations

import torch
from torch import Tensor

from upr_mvs.models.losses.consistency import masked_mean


def uncertainty_aware_l1_loss(
    pred_points: Tensor,
    gt_points: Tensor,
    log_sigma: Tensor,
    point_mask: Tensor,
) -> Tensor:
    point_error = (pred_points - gt_points).abs().mean(dim=-1, keepdim=True)
    per_point_loss = torch.exp(-log_sigma) * point_error + log_sigma
    per_point_loss = torch.where(point_mask, per_point_loss, torch.zeros_like(per_point_loss))
    return masked_mean(per_point_loss, point_mask)
