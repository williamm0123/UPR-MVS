from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from upr_mvs.models.transformer.utils import scale_intrinsics


def project_world_to_view(points_world: Tensor, intrinsics: Tensor, extrinsics: Tensor) -> tuple[Tensor, Tensor]:
    if points_world.ndim != 3 or points_world.shape[-1] != 3:
        raise ValueError(f"Expected points_world with shape [B, N, 3], got {tuple(points_world.shape)}")

    ones = torch.ones_like(points_world[..., :1])
    points_world_h = torch.cat([points_world, ones], dim=-1)  # [B, N, 4]
    camera_points_h = torch.bmm(points_world_h, extrinsics.transpose(1, 2))  # [B, N, 4]
    camera_points = camera_points_h[..., :3]  # [B, N, 3]
    pixel_h = torch.bmm(camera_points, intrinsics.transpose(1, 2))  # [B, N, 3]
    z = camera_points[..., 2].clamp(min=1.0e-6)
    xy = pixel_h[..., :2] / z.unsqueeze(-1)
    return xy, camera_points[..., 2]


def sample_feature_map(feature_map: Tensor, pixel_coords: Tensor) -> tuple[Tensor, Tensor]:
    """
    Args:
        feature_map: [B, C, H, W]
        pixel_coords: [B, N, 2] in feature-map pixel coordinates
    Returns:
        sampled_features: [B, N, C]
        visibility: [B, N, 1]
    """
    batch_size, _, height, width = feature_map.shape
    x = pixel_coords[..., 0]
    y = pixel_coords[..., 1]
    visibility = (
        (x >= 0.0)
        & (x <= float(max(width - 1, 0)))
        & (y >= 0.0)
        & (y <= float(max(height - 1, 0)))
    ).unsqueeze(-1)

    norm_x = (x / max(width - 1, 1)) * 2.0 - 1.0
    norm_y = (y / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).view(batch_size, -1, 1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(-1).transpose(1, 2).contiguous(), visibility


class PointFeatureProjector(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        hidden_channels = max(out_channels * 2, 256)
        super().__init__(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels),
        )


class MultiViewFeatureLifter(nn.Module):
    """Lift 2D multi-view FPN features onto the point set and aggregate them."""

    def __init__(
        self,
        image_feature_dim: int,
        point_feature_dim: int,
        stage_keys: Sequence[str] = ("stage1", "stage2", "stage3"),
    ) -> None:
        super().__init__()
        self.stage_keys = list(stage_keys)
        self.image_feature_dim = int(image_feature_dim)
        self.point_feature_dim = int(point_feature_dim)
        concat_feature_dim = self.image_feature_dim * len(self.stage_keys)
        raw_dim = 3 + concat_feature_dim * 3 + 1 + 1 + 1
        self.projector = PointFeatureProjector(raw_dim, point_feature_dim)

    def forward(
        self,
        feature_pyramid: dict[str, Tensor],
        points_world: Tensor,
        point_confidence: Tensor,
        point_mask: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        image_hw: tuple[int, int],
        depth_range: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, num_views = intrinsics.shape[:2]
        image_h, image_w = image_hw

        if point_mask.ndim != 3 or point_mask.shape[-1] != 1:
            raise ValueError("point_mask must have shape [B, N, 1].")

        per_view_feature_chunks = [[] for _ in range(num_views)]
        per_view_visibility_chunks = [[] for _ in range(num_views)]

        for stage_key in self.stage_keys:
            if stage_key not in feature_pyramid:
                raise KeyError(f"Feature pyramid missing stage '{stage_key}'")
            stage_features = feature_pyramid[stage_key]  # [B, V, C, Hs, Ws]
            _, _, stage_channels, feat_h, feat_w = stage_features.shape
            if stage_channels != self.image_feature_dim:
                raise ValueError(
                    f"Expected {self.image_feature_dim} channels for stage '{stage_key}', got {stage_channels}"
                )

            scaled_intrinsics = scale_intrinsics(
                intrinsics.view(batch_size * num_views, 3, 3),
                scale_x=feat_w / float(image_w),
                scale_y=feat_h / float(image_h),
            ).view(batch_size, num_views, 3, 3)

            for view_index in range(num_views):
                projected_xy, projected_depth = project_world_to_view(
                    points_world=points_world,
                    intrinsics=scaled_intrinsics[:, view_index],
                    extrinsics=extrinsics[:, view_index],
                )
                sampled_features, sampled_visibility = sample_feature_map(
                    feature_map=stage_features[:, view_index],
                    pixel_coords=projected_xy,
                )
                depth_visibility = (projected_depth > 0.0).unsqueeze(-1)
                per_view_feature_chunks[view_index].append(sampled_features)
                per_view_visibility_chunks[view_index].append(sampled_visibility & depth_visibility)

        per_view_features = [torch.cat(feature_chunks, dim=-1) for feature_chunks in per_view_feature_chunks]
        per_view_visibility = [
            torch.stack(visibility_chunks, dim=0).all(dim=0) for visibility_chunks in per_view_visibility_chunks
        ]

        ref_features = per_view_features[0]
        ref_visibility = per_view_visibility[0]
        source_features = torch.stack(per_view_features[1:], dim=1)  # [B, Vs, N, Ctotal]
        source_visibility = torch.stack(per_view_visibility[1:], dim=1).float()  # [B, Vs, N, 1]

        visible_count = source_visibility.sum(dim=1)  # [B, N, 1]
        safe_visible_count = visible_count.clamp_min(1.0)
        source_mean = (source_features * source_visibility).sum(dim=1) / safe_visible_count
        source_var = ((source_features - source_mean.unsqueeze(1)).square() * source_visibility).sum(dim=1) / safe_visible_count

        cosine = F.cosine_similarity(ref_features.unsqueeze(1), source_features, dim=-1)  # [B, Vs, N]
        cosine = (cosine * source_visibility.squeeze(-1)).sum(dim=1) / safe_visible_count.squeeze(-1)
        cosine = cosine.unsqueeze(-1)  # [B, N, 1]

        points_world_h = torch.cat([points_world, torch.ones_like(points_world[..., :1])], dim=-1)
        ref_camera_points = torch.bmm(points_world_h, extrinsics[:, 0].transpose(1, 2))[..., :3]  # [B, N, 3]
        depth_span = (depth_range[:, 1] - depth_range[:, 0]).clamp_min(1.0e-6).view(batch_size, 1, 1)
        depth_norm = ((ref_camera_points[..., 2:3] - depth_range[:, 0].view(batch_size, 1, 1)) / depth_span).clamp(0.0, 1.0)
        xyz_norm = ref_camera_points / depth_span

        raw_features = torch.cat(
            [
                xyz_norm,
                ref_features,
                source_mean,
                source_var,
                cosine,
                point_confidence,
                depth_norm,
            ],
            dim=-1,
        )
        point_features = self.projector(raw_features)
        combined_visibility = point_mask.float() * ref_visibility.float() * (visible_count > 0.0).float()
        point_features = point_features * combined_visibility

        return {
            "point_feat": point_features,
            "ref_feat": ref_features,
            "src_mean": source_mean,
            "src_var": source_var,
            "cosine": cosine,
            "ref_camera_points": ref_camera_points,
            "point_visibility": combined_visibility,
        }
