from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .chamfer import chamfer_distance_loss
from .consistency import (
    alpha_target_from_error,
    build_sparse_gt_points,
    compute_coarse_depth_loss,
    compute_point_l1_error,
    masked_mean,
)
from .repulsion import repulsion_loss
from .reprojection import sparse_feature_consistency_loss
from .uncertainty import uncertainty_aware_l1_loss


class UPRMVSLoss(nn.Module):
    """Composite loss for coarse depth and point refinement training."""

    def __init__(self, loss_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg

    def forward(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        loss_dict = compute_coarse_depth_loss(outputs, batch, self.loss_cfg)
        zero = outputs["coarse_depth"].sum() * 0.0

        if "points_refined" not in outputs:
            loss_dict.update(
                {
                    "loss_chamfer": zero,
                    "loss_uncertainty": zero,
                    "loss_feature": zero,
                    "loss_repulsion": zero,
                    "loss_alpha": zero,
                    "point_abs_error": zero,
                    "final_point_abs_error": zero,
                    "avg_sigma": zero,
                    "point_valid_ratio": zero,
                    "num_init_points": zero,
                    "num_final_points": zero,
                    "num_densified_points": zero,
                    "loss_total": loss_dict["loss_coarse"],
                }
            )
            return loss_dict

        gt_sparse = build_sparse_gt_points(outputs, batch)
        point_mask = outputs["point_mask"] & gt_sparse["point_gt_mask"]
        final_points = outputs.get("points_final", outputs["points_refined"])
        final_mask = outputs.get("point_final_mask", outputs["point_mask"])

        point_abs_error, point_error_map = compute_point_l1_error(
            pred_points=outputs["points_refined"],
            gt_points=gt_sparse["points_gt_world"],
            point_mask=point_mask,
        )
        final_point_abs_error, _ = compute_point_l1_error(
            pred_points=final_points[:, : gt_sparse["points_gt_world"].shape[1]],
            gt_points=gt_sparse["points_gt_world"],
            point_mask=gt_sparse["point_gt_mask"],
        )
        loss_chamfer = chamfer_distance_loss(
            pred_points=final_points,
            gt_points=gt_sparse["points_gt_world"],
            pred_mask=final_mask,
            gt_mask=gt_sparse["point_gt_mask"],
        ) * float(self.loss_cfg.get("chamfer_weight", 1.0))
        loss_uncertainty = uncertainty_aware_l1_loss(
            pred_points=outputs["points_refined"],
            gt_points=gt_sparse["points_gt_world"],
            log_sigma=outputs["log_sigma"],
            point_mask=point_mask,
        ) * float(self.loss_cfg.get("uncertainty_weight", 0.2))

        feature_stage = str(self.loss_cfg.get("feature_stage", "stage2"))
        loss_feature = sparse_feature_consistency_loss(
            feature_pyramid=outputs["feature_pyramid"],
            stage_key=feature_stage,
            points_world=outputs["points_refined"],
            point_mask=point_mask,
            intrinsics=batch["intrinsics"],
            extrinsics=batch["extrinsics"],
            image_hw=(batch["imgs"].shape[-2], batch["imgs"].shape[-1]),
        ) * float(self.loss_cfg.get("feature_weight", 0.5))

        if "point_final_neighbor_idx" in outputs and float(self.loss_cfg.get("repulsion_weight", 0.0)) > 0.0:
            loss_repulsion = repulsion_loss(
                points=final_points,
                neighbor_idx=outputs["point_final_neighbor_idx"],
                point_mask=final_mask,
                radius=float(self.loss_cfg.get("repulsion_radius", 0.01)),
            ) * float(self.loss_cfg.get("repulsion_weight", 0.0))
        else:
            loss_repulsion = zero

        alpha_target = alpha_target_from_error(
            point_error=point_error_map,
            point_mask=point_mask,
            threshold=float(self.loss_cfg.get("alpha_error_threshold", 2.0)),
        )
        if bool(point_mask.any()):
            alpha_loss_map = F.binary_cross_entropy(outputs["alpha"], alpha_target, reduction="none")
            loss_alpha = masked_mean(alpha_loss_map, point_mask) * float(self.loss_cfg.get("alpha_weight", 0.2))
        else:
            loss_alpha = zero

        point_valid_ratio = point_mask.float().mean()
        avg_sigma = masked_mean(outputs["sigma"], point_mask) if bool(point_mask.any()) else zero
        num_init_points = outputs["point_mask"].float().sum(dim=1).mean()
        num_final_points = outputs.get("point_final_mask", outputs["point_mask"]).float().sum(dim=1).mean()
        num_densified_points = outputs.get("densified_mask", outputs["point_mask"][:, :0]).float().sum(dim=1).mean()

        loss_dict.update(
            {
                "loss_chamfer": loss_chamfer,
                "loss_uncertainty": loss_uncertainty,
                "loss_feature": loss_feature,
                "loss_repulsion": loss_repulsion,
                "loss_alpha": loss_alpha,
                "point_abs_error": point_abs_error,
                "final_point_abs_error": final_point_abs_error,
                "avg_sigma": avg_sigma,
                "point_valid_ratio": point_valid_ratio,
                "num_init_points": num_init_points,
                "num_final_points": num_final_points,
                "num_densified_points": num_densified_points,
            }
        )
        loss_dict["loss_total"] = (
            loss_dict["loss_coarse"]
            + loss_dict["loss_chamfer"]
            + loss_dict["loss_uncertainty"]
            + loss_dict["loss_feature"]
            + loss_dict["loss_repulsion"]
            + loss_dict["loss_alpha"]
        )
        return loss_dict
