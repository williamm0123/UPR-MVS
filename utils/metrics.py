from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def tensor_dict_to_floats(metrics: dict[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}


@dataclass
class ScalarMeter:
    sums: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, metrics: dict[str, float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)

    def averages(self) -> dict[str, float]:
        if self.count == 0:
            return {key: 0.0 for key in self.sums}
        return {key: value / self.count for key, value in self.sums.items()}


def format_metrics(metrics: dict[str, float], keys: list[str] | None = None) -> str:
    if keys is None:
        keys = sorted(metrics.keys())
    return " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


def sparse_point_cloud_metrics(
    pred_points: Tensor,
    gt_points: Tensor,
    pred_mask: Tensor,
    gt_mask: Tensor,
) -> dict[str, Tensor]:
    zero = pred_points.sum() * 0.0
    accuracy_terms: list[Tensor] = []
    completeness_terms: list[Tensor] = []

    for batch_idx in range(pred_points.shape[0]):
        pred_valid = pred_mask[batch_idx].squeeze(-1)
        gt_valid = gt_mask[batch_idx].squeeze(-1)
        pred_valid_points = pred_points[batch_idx][pred_valid]
        gt_valid_points = gt_points[batch_idx][gt_valid]

        if pred_valid_points.numel() == 0 or gt_valid_points.numel() == 0:
            accuracy_terms.append(zero)
            completeness_terms.append(zero)
            continue

        pairwise_dist = torch.cdist(pred_valid_points.unsqueeze(0), gt_valid_points.unsqueeze(0)).squeeze(0)
        accuracy_terms.append(pairwise_dist.min(dim=1).values.mean())
        completeness_terms.append(pairwise_dist.min(dim=0).values.mean())

    if not accuracy_terms:
        return {"point_accuracy": zero, "point_completeness": zero, "point_overall": zero}

    accuracy = torch.stack(accuracy_terms).mean()
    completeness = torch.stack(completeness_terms).mean()
    overall = 0.5 * (accuracy + completeness)
    return {
        "point_accuracy": accuracy,
        "point_completeness": completeness,
        "point_overall": overall,
    }
