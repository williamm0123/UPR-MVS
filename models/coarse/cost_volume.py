from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sample_depth_planes(depth_range: Tensor, num_depth_bins: int) -> Tensor:
    """
    Args:
        depth_range: [B, 2] where [:, 0] is min depth and [:, 1] is max depth.
        num_depth_bins: number of depth planes.

    Returns:
        Tensor of shape [B, D].
    """
    if depth_range.ndim != 2 or depth_range.shape[1] != 2:
        raise ValueError(f"Expected depth_range with shape [B, 2], but got {tuple(depth_range.shape)}")
    if num_depth_bins <= 1:
        raise ValueError("num_depth_bins must be greater than 1.")
    depth_min = depth_range[:, 0]
    depth_max = depth_range[:, 1]
    if torch.any(depth_max <= depth_min):
        raise ValueError("Each sample must satisfy depth_max > depth_min.")

    linear_steps = torch.linspace(0.0, 1.0, num_depth_bins, device=depth_range.device, dtype=depth_range.dtype).view(1, -1)
    return depth_min.unsqueeze(1) + linear_steps * (depth_max - depth_min).unsqueeze(1)


def scale_intrinsics(intrinsics: Tensor, scale_x: float, scale_y: float) -> Tensor:
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics with shape [B, 3, 3], but got {tuple(intrinsics.shape)}")
    scaled = intrinsics.clone()
    scaled[:, 0, :] *= scale_x
    scaled[:, 1, :] *= scale_y
    return scaled


def intrinsics_to_projection(intrinsics: Tensor, extrinsics: Tensor) -> Tensor:
    """
    Args:
        intrinsics: [B, 3, 3]
        extrinsics: [B, 4, 4] world-to-camera

    Returns:
        projection: [B, 4, 4]
    """
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"Expected extrinsics with shape [B, 4, 4], but got {tuple(extrinsics.shape)}")
    projection = intrinsics.new_zeros((intrinsics.shape[0], 4, 4))
    projection[:, 3, 3] = 1.0
    projection[:, :3, :4] = torch.matmul(intrinsics, extrinsics[:, :3, :4])
    return projection


def homo_warping(src_features: Tensor, src_projection: Tensor, ref_projection: Tensor, depth_values: Tensor) -> Tensor:
    """
    Args:
        src_features: [B, C, H, W]
        src_projection: [B, 4, 4]
        ref_projection: [B, 4, 4]
        depth_values: [B, D]

    Returns:
        Warped source volume with shape [B, C, D, H, W].
    """
    if src_features.ndim != 4:
        raise ValueError(f"Expected src_features with shape [B, C, H, W], got {tuple(src_features.shape)}")

    batch_size, channels, height, width = src_features.shape
    num_depth_bins = depth_values.shape[1]

    with torch.no_grad():
        projection = torch.matmul(src_projection, torch.linalg.inv(ref_projection))
        rotation = projection[:, :3, :3]
        translation = projection[:, :3, 3:4]

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=src_features.device, dtype=src_features.dtype),
            torch.arange(width, device=src_features.device, dtype=src_features.dtype),
            indexing="ij",
        )
        ones = torch.ones_like(x_coords)
        reference_grid = torch.stack((x_coords, y_coords, ones), dim=0).view(1, 3, height * width).repeat(batch_size, 1, 1)

        rotated = torch.matmul(rotation, reference_grid)
        rotated_depth = rotated.unsqueeze(2) * depth_values.view(batch_size, 1, num_depth_bins, 1)
        projected = rotated_depth + translation.view(batch_size, 3, 1, 1)
        projected_z = projected[:, 2:3].clamp(min=1e-6)
        projected_xy = projected[:, :2] / projected_z

        normalized_x = (projected_xy[:, 0].view(batch_size, num_depth_bins, height, width) / max(width - 1, 1)) * 2.0 - 1.0
        normalized_y = (projected_xy[:, 1].view(batch_size, num_depth_bins, height, width) / max(height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack((normalized_x, normalized_y), dim=-1)

    warped = F.grid_sample(
        src_features,
        grid.view(batch_size, num_depth_bins * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.view(batch_size, channels, num_depth_bins, height, width)


class VarianceCostVolumeBuilder(nn.Module):
    """Standard MVSNet-style variance cost volume aggregation."""

    def forward(
        self,
        ref_features: Tensor,
        src_features: Sequence[Tensor],
        ref_projection: Tensor,
        src_projections: Sequence[Tensor],
        depth_values: Tensor,
    ) -> Tensor:
        if len(src_features) != len(src_projections):
            raise ValueError("src_features and src_projections must have the same length.")
        if len(src_features) == 0:
            raise ValueError("At least one source view is required to build a cost volume.")

        num_depth_bins = depth_values.shape[1]
        ref_volume = ref_features.unsqueeze(2).expand(-1, -1, num_depth_bins, -1, -1)
        volume_sum = ref_volume.clone()
        volume_sq_sum = ref_volume.square()

        for feature_map, projection in zip(src_features, src_projections):
            warped = homo_warping(feature_map, projection, ref_projection, depth_values)
            volume_sum = volume_sum + warped
            volume_sq_sum = volume_sq_sum + warped.square()

        num_views = float(len(src_features) + 1)
        mean = volume_sum / num_views
        return volume_sq_sum / num_views - mean.square()
