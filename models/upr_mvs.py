from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from upr_mvs.models.backbone.resnet_fpn import ResNetFPN
from upr_mvs.models.coarse.coarse_depth_head import CoarseDepthHead
from upr_mvs.models.coarse.cost_volume import scale_intrinsics
from upr_mvs.models.point.densify import RuleBasedDensifier
from upr_mvs.models.point.feature_lifting import MultiViewFeatureLifter
from upr_mvs.models.point.point_refiner import EdgeConvPointRefiner
from upr_mvs.models.point.knn import build_knn_graph
from upr_mvs.models.point.unproject import DepthPointUnprojector


class UPRMVSModel(nn.Module):
    """Full round-2 UPR-MVS forward graph without densification."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        backbone_cfg = model_cfg["backbone"]
        coarse_cfg = model_cfg["coarse"]
        point_cfg = model_cfg["point"]

        self.backbone = ResNetFPN(
            variant=str(backbone_cfg.get("name", "resnet34")),
            out_channels=int(backbone_cfg.get("out_channels", 128)),
            pretrained=bool(backbone_cfg.get("pretrained", False)),
            use_checkpoint=bool(backbone_cfg.get("use_checkpoint", False)),
        )
        self.coarse_feature_key = str(coarse_cfg.get("feat_key", "stage2"))
        self.coarse_head = CoarseDepthHead(
            feature_dim=int(backbone_cfg.get("out_channels", 128)),
            depth_bins_train=int(coarse_cfg.get("depth_bins_train", 48)),
            depth_bins_test=int(coarse_cfg.get("depth_bins_test", 96)),
            base_channels=int(coarse_cfg.get("base_channels", 8)),
            use_checkpoint=bool(coarse_cfg.get("use_checkpoint", False)),
        )

        self.use_gt_mask_for_sampling = bool(point_cfg.get("use_gt_mask_for_sampling", True))
        self.unprojector = DepthPointUnprojector(
            num_points=int(point_cfg.get("num_points", 12000)),
            use_all_if_fewer=bool(point_cfg.get("use_all_if_fewer", True)),
        )
        self.feature_lifter = MultiViewFeatureLifter(
            image_feature_dim=int(backbone_cfg.get("out_channels", 128)),
            point_feature_dim=int(point_cfg.get("feat_dim", 160)),
            stage_keys=tuple(point_cfg.get("lift_stage_keys", ["stage1", "stage2", "stage3"])),
        )
        self.point_refiner = EdgeConvPointRefiner(
            feature_dim=int(point_cfg.get("feat_dim", 160)),
            layer_dims=tuple(point_cfg.get("edgeconv_layers", [128, 192, 256])),
            knn=int(point_cfg.get("knn", 16)),
            candidate_k=int(point_cfg.get("candidate_k", 64)),
            delta_scale=float(point_cfg.get("delta_scale", 0.01)),
            use_checkpoint=bool(point_cfg.get("use_checkpoint", False)),
            predict_uncertainty=bool(point_cfg.get("predict_uncertainty", True)),
        )
        densify_cfg = model_cfg.get("densify", {})
        self.densifier = RuleBasedDensifier(
            enable=bool(densify_cfg.get("enable", False)),
            tau_sigma=float(densify_cfg.get("tau_sigma", 0.15)),
            tau_alpha=float(densify_cfg.get("tau_alpha", 0.6)),
            split_factor=int(densify_cfg.get("split_factor", 2)),
            offset_scale=float(densify_cfg.get("offset_scale", 0.35)),
            max_new_points=int(densify_cfg.get("max_new_points", 12000)),
            use_sigma_gate=bool(densify_cfg.get("use_sigma_gate", True)),
        )
        self.final_knn = int(densify_cfg.get("final_knn", point_cfg.get("knn", 16)))
        self.final_candidate_k = int(densify_cfg.get("final_candidate_k", point_cfg.get("candidate_k", 64)))

    def _encode_multiview_features(self, images: Tensor) -> tuple[dict[str, Tensor], tuple[int, int]]:
        batch_size, num_views, channels, img_h, img_w = images.shape
        flat_images = images.view(batch_size * num_views, channels, img_h, img_w)
        flat_pyramid = self.backbone(flat_images)

        pyramid: dict[str, Tensor] = {}
        for key, value in flat_pyramid.items():
            feat_channels = value.shape[1]
            feat_h, feat_w = value.shape[-2:]
            pyramid[key] = value.view(batch_size, num_views, feat_channels, feat_h, feat_w)
        return pyramid, (img_h, img_w)

    def forward(self, batch: dict[str, Tensor], num_depth_bins: int | None = None) -> dict[str, Tensor]:
        images = batch["imgs"]  # [B, V, 3, H, W]
        intrinsics = batch["intrinsics"]  # [B, V, 3, 3]
        extrinsics = batch["extrinsics"]  # [B, V, 4, 4]
        depth_range = batch["depth_range"]  # [B, 2]

        if images.ndim != 5:
            raise ValueError(f"Expected imgs with shape [B, V, 3, H, W], but got {tuple(images.shape)}")

        feature_pyramid, image_hw = self._encode_multiview_features(images)
        coarse_features = feature_pyramid[self.coarse_feature_key]  # [B, V, C, Hc, Wc]
        coarse_outputs = self.coarse_head(
            features=coarse_features,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_range=depth_range,
            image_hw=image_hw,
            num_depth_bins=num_depth_bins,
        )

        _, _, _, coarse_h, coarse_w = coarse_features.shape
        img_h, img_w = image_hw
        ref_intrinsics = intrinsics[:, 0]
        ref_extrinsics = extrinsics[:, 0]
        scaled_ref_intrinsics = scale_intrinsics(
            ref_intrinsics,
            scale_x=coarse_w / float(img_w),
            scale_y=coarse_h / float(img_h),
        )

        point_valid_mask = None
        if self.use_gt_mask_for_sampling and "mask" in batch:
            point_valid_mask = F.interpolate(batch["mask"].float(), size=(coarse_h, coarse_w), mode="nearest")
            if "has_depth_gt" in batch:
                has_depth_gt = batch["has_depth_gt"].view(-1, 1, 1, 1)
                point_valid_mask = torch.where(has_depth_gt, point_valid_mask, torch.ones_like(point_valid_mask))

        point_init = self.unprojector(
            depth=coarse_outputs["coarse_depth"],
            intrinsics=scaled_ref_intrinsics,
            extrinsics=ref_extrinsics,
            confidence=coarse_outputs["coarse_confidence"],
            valid_mask=point_valid_mask,
        )

        lifted = self.feature_lifter(
            feature_pyramid=feature_pyramid,
            points_world=point_init["points_world"],
            point_confidence=point_init["point_confidence"],
            point_mask=point_init["point_mask"],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=image_hw,
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
            "point_mask": point_init["point_mask"],
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
