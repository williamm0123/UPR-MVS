from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

_GRID_SAMPLE_FALLBACK_WARNED = False


def sample_depth_planes(depth_range: Tensor, num_depth_bins: int) -> Tensor:
    depth_min = depth_range[:, 0]
    depth_max = depth_range[:, 1]
    t = torch.linspace(0.0, 1.0, num_depth_bins, device=depth_range.device, dtype=depth_range.dtype).view(1, -1)
    return depth_min[:, None] + t * (depth_max - depth_min)[:, None]


def scale_intrinsics(intrinsics: Tensor, scale_x: float, scale_y: float) -> Tensor:
    scaled = intrinsics.clone()
    scaled[:, 0, :] *= scale_x
    scaled[:, 1, :] *= scale_y
    return scaled


def intrinsics_to_projection(intrinsics: Tensor, extrinsics: Tensor) -> Tensor:
    projection = intrinsics.new_zeros((intrinsics.shape[0], 4, 4))
    projection[:, 3, 3] = 1.0
    projection[:, :3, :4] = torch.matmul(intrinsics, extrinsics[:, :3, :4])
    return projection


def safe_grid_sample(
    input_tensor: Tensor,
    grid: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
    input_tensor = input_tensor.contiguous()
    grid = grid.contiguous()

    try:
        return F.grid_sample(
            input_tensor,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "CUDNN_STATUS_NOT_SUPPORTED" not in message:
            raise

        global _GRID_SAMPLE_FALLBACK_WARNED
        if not _GRID_SAMPLE_FALLBACK_WARNED:
            warnings.warn(
                "grid_sample hit a cuDNN unsupported path; falling back to the native CUDA kernel for stability.",
                stacklevel=2,
            )
            _GRID_SAMPLE_FALLBACK_WARNED = True

        with torch.backends.cudnn.flags(enabled=False):
            return F.grid_sample(
                input_tensor,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )


def homo_warping(src_features: Tensor, src_projection: Tensor, ref_projection: Tensor, depth_values: Tensor) -> Tensor:
    b, c, h, w = src_features.shape
    d = depth_values.shape[1]
    projection = torch.matmul(src_projection, torch.linalg.inv(ref_projection))
    rotation = projection[:, :3, :3]
    translation = projection[:, :3, 3:4]

    y, x = torch.meshgrid(
        torch.arange(h, device=src_features.device, dtype=src_features.dtype),
        torch.arange(w, device=src_features.device, dtype=src_features.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    ref_grid = torch.stack((x, y, ones), dim=0).view(1, 3, h * w).repeat(b, 1, 1)
    rot = torch.matmul(rotation, ref_grid)
    rot_depth = rot.unsqueeze(2) * depth_values.view(b, 1, d, 1)
    proj = rot_depth + translation.view(b, 3, 1, 1)
    proj_z = proj[:, 2:3].clamp(min=1e-6)
    proj_xy = proj[:, :2] / proj_z

    norm_x = (proj_xy[:, 0].view(b, d, h, w) / max(w - 1, 1)) * 2.0 - 1.0
    norm_y = (proj_xy[:, 1].view(b, d, h, w) / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)

    warped = safe_grid_sample(
        src_features,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.view(b, c, d, h, w)


def group_wise_correlation(ref_volume: Tensor, src_volume: Tensor, num_groups: int) -> Tensor:
    b, c, d, h, w = ref_volume.shape
    g = max(1, min(num_groups, c))
    group_channels = c // g
    ref = ref_volume.view(b, g, group_channels, d, h, w)
    src = src_volume.view(b, g, group_channels, d, h, w)
    return (ref * src).mean(dim=2)
