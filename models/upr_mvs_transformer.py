from __future__ import annotations

from contextlib import nullcontext
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone.depth_anything3 import DepthAnything3MetricPrior
from .point.densify import RuleBasedDensifier
from .point.feature_lifting import MultiViewFeatureLifter
from .point.knn import build_knn_graph
from .point.point_refiner import EdgeConvPointRefiner
from .point.unproject import DepthPointUnprojector
from .transformer.utils import scale_intrinsics


class UPRMVSTransformerModel(nn.Module):
    """UPR-MVS with a frozen Depth Anything 3 metric depth prior."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        backbone_name = str(model_cfg.get("backbone", "depth_anything3")).lower()
        if backbone_name != "depth_anything3":
            raise ValueError(
                "UPRMVSTransformerModel is now DA3-only. "
                f"Expected model.backbone='depth_anything3', got '{backbone_name}'."
            )

        prior_cfg = model_cfg.get("depth_anything3", {})
        self.depth_prior = DepthAnything3MetricPrior(prior_cfg)
        self.backbone = self.depth_prior.backbone
        self.backbone_name = backbone_name
        self.active_train_stage = str(model_cfg.get("train_stage", "joint")).lower()

        point_cfg = model_cfg.get("point", {})
        point_stage_keys = list(point_cfg.get("lift_stage_keys", ["stage1", "stage2", "stage3"]))
        image_feature_dim = self._resolve_uniform_feature_dim(point_stage_keys)
        configured_image_feature_dim = point_cfg.get("image_feat_dim", point_cfg.get("image_feature_dim"))
        if configured_image_feature_dim is not None and int(configured_image_feature_dim) != image_feature_dim:
            warnings.warn(
                "model.point.image_feat_dim does not match the selected stage feature width "
                f"{image_feature_dim}; using runtime feature width instead of {configured_image_feature_dim}.",
                stacklevel=2,
            )

        self.feature_lifter = MultiViewFeatureLifter(
            image_feature_dim=image_feature_dim,
            point_feature_dim=int(point_cfg.get("feat_dim", 160)),
            stage_keys=point_stage_keys,
        )
        self.point_refiner = EdgeConvPointRefiner(
            feature_dim=int(point_cfg.get("feat_dim", 160)),
            layer_dims=list(point_cfg.get("edgeconv_layers", [128, 192, 256])),
            knn=int(point_cfg.get("knn", 16)),
            candidate_k=int(point_cfg.get("candidate_k", 64)),
            delta_scale=float(point_cfg.get("delta_scale", 0.01)),
            use_checkpoint=bool(point_cfg.get("use_checkpoint", False)),
            predict_uncertainty=bool(point_cfg.get("predict_uncertainty", True)),
        )
        self.unprojector = DepthPointUnprojector(
            num_points=int(point_cfg.get("num_points", 12000)),
            use_all_if_fewer=bool(point_cfg.get("use_all_if_fewer", True)),
            sampling_strategy=str(point_cfg.get("sampling_strategy", "score_hybrid")),
            topk_ratio=float(point_cfg.get("sampling_topk_ratio", 0.7)),
            coverage_grid_size=int(point_cfg.get("sampling_coverage_grid_size", 6)),
        )

        densify_cfg = model_cfg.get("densify", {})
        self.densifier = RuleBasedDensifier(enable=bool(densify_cfg.get("enable", False)))
        self.final_knn = int(point_cfg.get("final_knn", 16))
        self.final_candidate_k = int(point_cfg.get("final_candidate_k", 64))
        self.use_gt_mask_for_sampling = bool(model_cfg.get("use_gt_mask_for_sampling", True))

    def _get_backbone_projection(self, stage_key: str) -> nn.Conv2d:
        if stage_key not in self.backbone.out_proj:
            raise KeyError(f"Backbone is missing projection layer for stage '{stage_key}'")
        proj = self.backbone.out_proj[stage_key]
        if not isinstance(proj, nn.Conv2d):
            raise TypeError(
                f"Backbone projection for stage '{stage_key}' must be nn.Conv2d, got {type(proj).__name__}"
            )
        return proj

    def _resolve_feature_channels(self, stage_key: str) -> int:
        return int(self._get_backbone_projection(stage_key).out_channels)

    def _resolve_uniform_feature_dim(self, stage_keys: list[str]) -> int:
        if not stage_keys:
            raise ValueError("point.lift_stage_keys must contain at least one stage.")
        stage_dims = [self._resolve_feature_channels(stage_key) for stage_key in stage_keys]
        if len(set(stage_dims)) != 1:
            raise ValueError(
                "MultiViewFeatureLifter currently expects all selected lift stages to share one feature width, "
                f"but got {dict(zip(stage_keys, stage_dims))}."
            )
        return stage_dims[0]

    @staticmethod
    def _module_has_trainable_params(module: nn.Module | None) -> bool:
        if module is None:
            return False
        return any(parameter.requires_grad for parameter in module.parameters())

    def _grad_context_for_modules(self, *modules: nn.Module | None):
        if any(self._module_has_trainable_params(module) for module in modules):
            return nullcontext()
        return torch.no_grad()

    def forward(self, batch: dict[str, Tensor], num_depth_bins: int | None = None) -> dict[str, Tensor]:  # noqa: ARG002
        images = batch["imgs"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        depth_range = batch["depth_range"]

        with self._grad_context_for_modules(self.depth_prior):
            prior_outputs = self.depth_prior(images, intrinsics)

        feature_pyramid = prior_outputs["feature_pyramid"]
        coarse_outputs = {key: value for key, value in prior_outputs.items() if key != "feature_pyramid"}

        coarse_depth = coarse_outputs["coarse_depth"]
        _, _, coarse_h, coarse_w = coarse_depth.shape
        img_h, img_w = images.shape[-2:]
        ref_intrinsics = intrinsics[:, 0]
        ref_extrinsics = extrinsics[:, 0]
        scaled_ref_intrinsics = scale_intrinsics(
            ref_intrinsics,
            scale_x=float(coarse_w) / float(img_w),
            scale_y=float(coarse_h) / float(img_h),
        )

        point_valid_mask = None
        if self.use_gt_mask_for_sampling and "mask" in batch:
            point_valid_mask = F.interpolate(batch["mask"].float(), size=(coarse_h, coarse_w), mode="nearest")
            if "has_depth_gt" in batch:
                has_depth_gt = batch["has_depth_gt"].view(-1, 1, 1, 1)
                point_valid_mask = torch.where(has_depth_gt, point_valid_mask, torch.ones_like(point_valid_mask))

        point_init = self.unprojector(
            depth=coarse_depth,
            intrinsics=scaled_ref_intrinsics,
            extrinsics=ref_extrinsics,
            confidence=coarse_outputs["coarse_confidence"],
            score=coarse_outputs.get("point_selection_score"),
            valid_mask=point_valid_mask,
        )
        lifted = self.feature_lifter(
            feature_pyramid=feature_pyramid,
            points_world=point_init["points_world"],
            point_confidence=point_init["point_confidence"],
            point_mask=point_init["point_mask"],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(img_h, img_w),
            depth_range=depth_range,
        )
        refined = self.point_refiner(
            points=point_init["points_world"],
            point_feat=lifted["point_feat"],
            pixel_coords=point_init["pixel_coords"],
            point_mask=point_init["point_mask"],
        )
        points_refined = point_init["points_world"] + refined["delta_p"]

        ref_camera_points_refined_h = torch.cat([points_refined, torch.ones_like(points_refined[..., :1])], dim=-1)
        ref_camera_points_refined = torch.bmm(ref_camera_points_refined_h, ref_extrinsics.transpose(1, 2))[..., :3]
        densify_outputs = self.densifier(
            points_world=points_refined,
            points_cam=ref_camera_points_refined,
            point_mask=point_init["point_mask"],
            point_pixels=point_init["pixel_coords"],
            sigma=refined["sigma"],
            alpha=refined["alpha"],
            ref_intrinsics=scaled_ref_intrinsics,
            ref_extrinsics=ref_extrinsics,
        )
        point_final_neighbor_idx = build_knn_graph(
            points=densify_outputs["points_final"],
            pixel_coords=densify_outputs["point_final_pixels"],
            k=self.final_knn,
            point_mask=densify_outputs["point_final_mask"],
            candidate_k=self.final_candidate_k,
        )

        return {
            **coarse_outputs,
            "feature_pyramid": feature_pyramid,
            "points_init": point_init["points_world"],
            "points_init_cam": point_init["points_cam"],
            "points_refined": points_refined,
            "point_pixel_coords": point_init["pixel_coords"],
            "point_indices": point_init["point_indices"],
            "point_confidence": point_init["point_confidence"],
            "point_score": point_init["point_score"],
            "point_mask": point_init["point_mask"],
            "point_selection_score": coarse_outputs.get("point_selection_score", coarse_outputs["coarse_confidence"]),
            "point_feat": lifted["point_feat"],
            "point_visibility": lifted["point_visibility"],
            "ref_camera_points": lifted["ref_camera_points"],
            "delta_p": refined["delta_p"],
            "log_sigma": refined["log_sigma"],
            "sigma": refined["sigma"],
            "alpha": refined["alpha"],
            "neighbor_idx": refined["neighbor_idx"],
            "refined_feat": refined["refined_feat"],
            "points_final": densify_outputs["points_final"],
            "point_final_mask": densify_outputs["point_final_mask"],
            "point_final_pixels": densify_outputs["point_final_pixels"],
            "point_final_sigma": densify_outputs["point_final_sigma"],
            "point_final_alpha": densify_outputs["point_final_alpha"],
            "points_densified": densify_outputs["points_densified"],
            "densified_mask": densify_outputs["densified_mask"],
            "densified_pixels": densify_outputs["densified_pixels"],
            "densified_sigma": densify_outputs["densified_sigma"],
            "densified_alpha": densify_outputs["densified_alpha"],
            "densified_parent_idx": densify_outputs["densified_parent_idx"],
            "densified_count": densify_outputs["densified_count"],
            "densify_selected_count": densify_outputs["densify_selected_count"],
            "point_final_neighbor_idx": point_final_neighbor_idx,
        }
