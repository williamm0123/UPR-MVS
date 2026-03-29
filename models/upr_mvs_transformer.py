from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 修复导入路径：使用单点相对路径（同级模块）或三点绝对路径
from .backbone.dinov3 import DinoV3Backbone
from .transformer.cost_volume_transformer import CostVolumeTransformer
from .point.feature_lifting import MultiViewFeatureLifter
from .point.point_refiner import EdgeConvPointRefiner
from .point.unproject import DepthPointUnprojector
from .point.densify import RuleBasedDensifier
from .point.knn import build_knn_graph
from .ccff import CCFF
from .transformer.depth_refinement_head import DepthRefinementHead
from .transformer.utils import scale_intrinsics


class UPRMVSTransformerModel(nn.Module):
    """
    Transformer-based UPR-MVS model with DINOv3 backbone.
    
    Architecture:
        DINOv3 Backbone (3 层输出)
            ├── stage1: 高分辨率细节特征
            ├── stage2: 中等分辨率主特征
            └── stage3: 低分辨率全局特征
        
        CCFF Feature Fusion (可选)
            └── 融合三层特征 → fused_feature
        
        CostVolumeTransformer
            ├── 输入：fused_feature 或 stage2 特征
            ├── 构建 cost volume (d_bins=64/256)
            ├── 多尺度处理 (scales: [4, 8, 16])
            └── 输出：coarse_depth
        
        Depth Refinement Head (可选)
            └── 细化 coarse_depth → refined_depth
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone_name = str(model_cfg.get("backbone", "dinov3")).lower()
        self.coarse_feature_key = str(model_cfg.get("feat_key", "stage2"))
        
        # Check if CCFF is enabled
        self.use_ccff = bool(model_cfg.get("ccff", {}).get("enable", False))
        
        # Check if depth refinement is enabled
        self.use_depth_refinement = bool(model_cfg.get("depth_refinement", {}).get("enable", False))

        # Initialize backbone
        dinov3_name = str(model_cfg.get("dinov3_name", "dinov3_vitb16"))
        dinov3_pretrained = model_cfg.get("dinov3_pretrained")
        dinov3_output_layers = list(model_cfg.get("dinov3_output_layers", [3, 7, 11]))
        attention_backend = str(model_cfg.get("attention_backend", "sdpa"))
        sva_layers = list(model_cfg.get("sva_layers", [3, 6, 9]))
        sva_dropout = float(model_cfg.get("sva_dropout", 0.0))
        use_checkpoint = bool(model_cfg.get("use_checkpoint", False))

        self.backbone = DinoV3Backbone(
            name=dinov3_name,
            pretrained=dinov3_pretrained,                  # 修复参数名：pretrained_path -> pretrained
            output_layers=dinov3_output_layers,
            attention_backend=attention_backend,
            sva_layers=sva_layers,
            sva_dropout=sva_dropout,
            use_checkpoint=use_checkpoint,
        )

        # Initialize CCFF if enabled
        if self.use_ccff:
            ccff_cfg = model_cfg["ccff"]
            in_channels = list(ccff_cfg.get("in_channels", [768, 768, 768]))
            hidden_dim = int(ccff_cfg.get("hidden_dim", 256))
            depth_mult = float(ccff_cfg.get("depth_mult", 1.0))
            expansion = float(ccff_cfg.get("expansion", 1.0))
            act = str(ccff_cfg.get("act", "silu"))
            
            self.ccff = CCFF(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                depth_mult=depth_mult,
                expansion=expansion,
                act=act,
                out_index=0,
                return_all=False,
            )
            # Update feature key to use fused features
            self.coarse_feature_key = "ccff_output"
        
        # Initialize CVT
        cvt_cfg = model_cfg.get("cvt", {})
        self.cvt = CostVolumeTransformer(
            feature_dim=int(cvt_cfg.get("feature_dim", 256)),
            num_layers=int(cvt_cfg.get("num_layers", 6)),
            nhead=int(cvt_cfg.get("nhead", 8)),
            fpe_enable=bool(cvt_cfg.get("fpe_enable", True)),
            aas_enable=bool(cvt_cfg.get("aas_enable", True)),
            d_bins=int(cvt_cfg.get("d_bins", 64)),
            scales=list(cvt_cfg.get("scales", [4, 8, 16])),
            num_groups=int(cvt_cfg.get("num_groups", 8)),
            attention_backend=str(cvt_cfg.get("attention_backend", "sdpa")),
            use_checkpoint=bool(cvt_cfg.get("use_checkpoint", False)),
            share_across_scales=bool(cvt_cfg.get("share_across_scales", False)),
        )

        # Initialize Depth Refinement Head if enabled
        if self.use_depth_refinement:
            ref_cfg = model_cfg["depth_refinement"]
            self.depth_refinement_head = DepthRefinementHead(
                in_channels=int(ref_cfg.get("in_channels", 1)),
                hidden_channels=int(ref_cfg.get("hidden_channels", 16)),
                out_channels=int(ref_cfg.get("out_channels", 1)),
            )
        else:
            self.depth_refinement_head = None

        # Initialize other modules
        point_cfg = model_cfg.get("point", {})
        self.feature_lifter = MultiViewFeatureLifter(
            image_feature_dim=int(point_cfg.get("feat_dim", 160)),     # 修复参数名
            point_feature_dim=int(point_cfg.get("feat_dim", 160)),    # 新增参数
            stage_keys=list(point_cfg.get("lift_stage_keys", ["stage1", "stage2", "stage3"])),  # 修复参数名
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
        
        densify_cfg = model_cfg.get("densify", {})
        self.densifier = RuleBasedDensifier(
            enable=bool(densify_cfg.get("enable", False)),
        )
        
        # Initialize final KNN parameters
        self.final_knn = int(point_cfg.get("final_knn", 16))
        self.final_candidate_k = int(point_cfg.get("final_candidate_k", 64))
        
        self.unprojector = DepthPointUnprojector()
        self.use_gt_mask_for_sampling = bool(model_cfg.get("use_gt_mask_for_sampling", True))

    def _encode_multiview_features(self, images: Tensor) -> tuple[dict[str, Tensor], tuple[int, int]]:
        feats = self.backbone.forward_multiview(images)
        
        # Apply CCFF fusion if enabled
        if self.use_ccff:
            # Extract stage1, stage2, stage3 features
            stage1 = feats.get("stage1", None)
            stage2 = feats.get("stage2", None)
            stage3 = feats.get("stage3", None)
            
            if stage1 is not None and stage2 is not None and stage3 is not None:
                # CCFF expects (low, mid, high) = (high_res, mid_res, low_res)
                fused = self.ccff(stage1, stage2, stage3)
                feats["ccff_output"] = fused
        
        return feats, (images.shape[-2], images.shape[-1])

    def forward(self, batch: dict[str, Tensor], num_depth_bins: int | None = None) -> dict[str, Tensor]:
        images = batch["imgs"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        depth_range = batch["depth_range"]

        feature_pyramid, image_hw = self._encode_multiview_features(images)
        coarse_features = feature_pyramid[self.coarse_feature_key]
        
        # Get coarse depth from CVT
        coarse_outputs = self.cvt(
            features=coarse_features,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_range=depth_range,
            image_hw=image_hw,
            num_depth_bins=num_depth_bins,
        )
        
        # Apply depth refinement if enabled
        if self.use_depth_refinement and self.depth_refinement_head is not None:
            coarse_depth = coarse_outputs["coarse_depth"]
            refined_depth = self.depth_refinement_head(coarse_depth)
            
            # Ensure refined depth has correct shape
            if refined_depth.shape != coarse_depth.shape:
                refined_depth = F.interpolate(refined_depth, size=coarse_depth.shape[-2:], mode="bilinear", align_corners=False)
            
            # Update outputs with refined depth
            coarse_outputs["coarse_depth"] = coarse_depth + refined_depth
            coarse_outputs["depth_refined"] = True

        batch_size, num_views, channels, coarse_h, coarse_w = coarse_features.shape
        img_h, img_w = image_hw
        ref_intrinsics = intrinsics[:, 0]
        ref_extrinsics = extrinsics[:, 0]
        
        # Scale intrinsics to match the coarse depth map resolution
        scale_x = float(coarse_w) / float(img_w)
        scale_y = float(coarse_h) / float(img_h)
        scaled_ref_intrinsics = scale_intrinsics(ref_intrinsics, scale_x=scale_x, scale_y=scale_y)

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

        # 构建输出字典
        outputs = {
            **coarse_outputs,
            "feature_pyramid": feature_pyramid,  # type: ignore[dict-item]
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
        
        return outputs  # type: ignore[return-value]
