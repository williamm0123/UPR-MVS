from __future__ import annotations

import torch
from torch import Tensor

from models.transformer.utils import scale_intrinsics
from models.losses.consistency import masked_mean
from models.point.feature_lifting import project_world_to_view, sample_feature_map


def sparse_feature_consistency_loss(
    feature_pyramid: dict[str, Tensor],
    stage_key: str,
    points_world: Tensor,
    point_mask: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor,
    image_hw: tuple[int, int],
) -> Tensor:
    if stage_key not in feature_pyramid:
        raise KeyError(f"Feature pyramid missing stage '{stage_key}'")

    stage_features = feature_pyramid[stage_key]  # [B, V, C, Hs, Ws]
    batch_size, num_views, _, feat_h, feat_w = stage_features.shape
    image_h, image_w = image_hw
    scaled_intrinsics = scale_intrinsics(
        intrinsics.view(batch_size * num_views, 3, 3),
        scale_x=feat_w / float(image_w),
        scale_y=feat_h / float(image_h),
    ).view(batch_size, num_views, 3, 3)

    ref_xy, ref_depth = project_world_to_view(points_world, scaled_intrinsics[:, 0], extrinsics[:, 0])
    ref_feat, ref_vis = sample_feature_map(stage_features[:, 0], ref_xy)
    ref_depth_vis = (ref_depth > 0.0).unsqueeze(-1)

    zero = points_world.sum() * 0.0
    total_loss = zero
    total_count = zero

    for view_idx in range(1, num_views):
        src_xy, src_depth = project_world_to_view(points_world, scaled_intrinsics[:, view_idx], extrinsics[:, view_idx])
        src_feat, src_vis = sample_feature_map(stage_features[:, view_idx], src_xy)
        src_depth_vis = (src_depth > 0.0).unsqueeze(-1)

        valid_mask = point_mask & ref_vis & src_vis & ref_depth_vis & src_depth_vis
        per_point_error = (ref_feat - src_feat).abs().mean(dim=-1, keepdim=True)
        total_loss = total_loss + masked_mean(per_point_error, valid_mask)
        total_count = total_count + 1.0

    if total_count.item() == 0:
        return zero
    return total_loss / total_count
