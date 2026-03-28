from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.backbone.dinov3 import DinoV3Backbone
from models.point.densify import RuleBasedDensifier
from models.point.feature_lifting import MultiViewFeatureLifter
from models.point.knn import build_knn_graph
from models.point.point_refiner import EdgeConvPointRefiner
from models.point.unproject import DepthPointUnprojector
from models.transformer.cost_volume_transformer import CostVolumeTransformer
from models.transformer.utils import scale_intrinsics


class UPRMVSTransformerModel(nn.Module):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        point_cfg = model_cfg["point"]
        cvt_cfg = model_cfg["cvt"]

        self.backbone = DinoV3Backbone(
            name=str(model_cfg.get("dinov3_name", "dinov3_vitb16")),
            pretrained=model_cfg.get("dinov3_pretrained") or None,
            sva_layers=tuple(model_cfg.get("sva_layers", [3, 6, 9])),
            sva_dropout=float(model_cfg.get("sva_dropout", 0.0)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        )
        self.coarse_feature_key = str(cvt_cfg.get("feat_key", "stage2"))
        self.cvt = CostVolumeTransformer(
            feature_dim=int(cvt_cfg.get("feature_dim", 256)),
            num_layers=int(cvt_cfg.get("num_layers", 6)),
            nhead=int(cvt_cfg.get("nhead", 8)),
            d_bins=int(cvt_cfg.get("d_bins", 64)),
            scales=tuple(cvt_cfg.get("scales", [4, 8, 16])),
            num_groups=int(cvt_cfg.get("num_groups", 8)),
            aas_enable=bool(cvt_cfg.get("aas_enable", True)),
            fpe_enable=bool(cvt_cfg.get("fpe_enable", True)),
            use_checkpoint=bool(cvt_cfg.get("use_checkpoint", True)),
            share_across_scales=bool(cvt_cfg.get("share_across_scales", False)),
        )

        self.use_gt_mask_for_sampling = bool(point_cfg.get("use_gt_mask_for_sampling", True))
        self.unprojector = DepthPointUnprojector(
            num_points=int(point_cfg.get("num_points", 12000)),
            use_all_if_fewer=bool(point_cfg.get("use_all_if_fewer", True)),
        )
        self.feature_lifter = MultiViewFeatureLifter(
            image_feature_dim=int(cvt_cfg.get("feature_dim", 256)),
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
        feats = self.backbone.forward_multiview(images)
        return feats, (images.shape[-2], images.shape[-1])

    def forward(self, batch: dict[str, Tensor], num_depth_bins: int | None = None) -> dict[str, Tensor]:
        images = batch["imgs"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        depth_range = batch["depth_range"]

        feature_pyramid, image_hw = self._encode_multiview_features(images)
        coarse_features = feature_pyramid[self.coarse_feature_key]
        coarse_outputs = self.cvt(
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
        scaled_ref_intrinsics = scale_intrinsics(ref_intrinsics, scale_x=coarse_w / float(img_w), scale_y=coarse_h / float(img_h))

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
