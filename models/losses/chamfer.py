from __future__ import annotations

import torch
from torch import Tensor


def chamfer_distance_loss(
    pred_points: Tensor,
    gt_points: Tensor,
    pred_mask: Tensor,
    gt_mask: Tensor,
) -> Tensor:
    batch_losses: list[Tensor] = []
    zero = pred_points.sum() * 0.0

    for batch_idx in range(pred_points.shape[0]):
        valid_pred = pred_mask[batch_idx].squeeze(-1)
        valid_gt = gt_mask[batch_idx].squeeze(-1)
        pred_valid_points = pred_points[batch_idx][valid_pred]
        gt_valid_points = gt_points[batch_idx][valid_gt]

        if pred_valid_points.numel() == 0 or gt_valid_points.numel() == 0:
            batch_losses.append(zero)
            continue

        pairwise_dist = torch.cdist(pred_valid_points.unsqueeze(0), gt_valid_points.unsqueeze(0)).squeeze(0)
        forward = pairwise_dist.min(dim=1).values.mean()
        backward = pairwise_dist.min(dim=0).values.mean()
        batch_losses.append(forward + backward)

    if not batch_losses:
        return zero
    return torch.stack(batch_losses).mean()
