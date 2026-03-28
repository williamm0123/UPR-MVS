from __future__ import annotations

import torch
from torch import Tensor, nn


def make_pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x_coords, y_coords), dim=-1).view(1, height * width, 2)


def gather_by_index(values: Tensor, index: Tensor) -> Tensor:
    if values.ndim < 2:
        raise ValueError(f"Expected values with batch dimension and point dimension, got {tuple(values.shape)}")
    if index.shape[0] != values.shape[0]:
        raise ValueError("Batch size mismatch between values and index.")

    batch_size = values.shape[0]
    batch_index_shape = [batch_size] + [1] * (index.ndim - 1)
    batch_indices = torch.arange(batch_size, device=values.device).view(*batch_index_shape).expand_as(index)
    return values[batch_indices, index]


def select_point_indices(valid_mask: Tensor, scores: Tensor, num_points: int) -> tuple[Tensor, Tensor]:
    batch_size, num_candidates = valid_mask.shape
    target = min(max(num_points, 1), num_candidates)

    if target == num_candidates:
        indices = torch.arange(num_candidates, device=valid_mask.device).view(1, num_candidates).expand(batch_size, -1)
        return indices, gather_by_index(valid_mask, indices)

    masked_scores = torch.where(valid_mask, scores, torch.full_like(scores, -1.0e8))
    indices = torch.topk(masked_scores, k=target, dim=1, largest=True, sorted=True).indices
    selected_valid = gather_by_index(valid_mask, indices)

    if bool(selected_valid.all()):
        return indices, selected_valid

    fallback = torch.argmax(masked_scores, dim=1, keepdim=True).expand(-1, target)
    indices = torch.where(selected_valid, indices, fallback)
    selected_valid = gather_by_index(valid_mask, indices)
    return indices, selected_valid


def depth_to_world_points(depth: Tensor, intrinsics: Tensor, extrinsics: Tensor, pixel_coords: Tensor) -> tuple[Tensor, Tensor]:
    """
    Args:
        depth: [B, N]
        intrinsics: [B, 3, 3] at the same resolution as pixel_coords
        extrinsics: [B, 4, 4] world-to-camera
        pixel_coords: [B, N, 2]
    """
    ones = torch.ones_like(depth).unsqueeze(-1)
    pixels_h = torch.cat([pixel_coords, ones], dim=-1)  # [B, N, 3]
    intrinsics_inv = torch.linalg.inv(intrinsics)  # [B, 3, 3]
    camera_rays = torch.bmm(pixels_h, intrinsics_inv.transpose(1, 2))  # [B, N, 3]
    camera_points = camera_rays * depth.unsqueeze(-1)  # [B, N, 3]

    camera_to_world = torch.linalg.inv(extrinsics)  # [B, 4, 4]
    camera_points_h = torch.cat([camera_points, torch.ones_like(depth).unsqueeze(-1)], dim=-1)  # [B, N, 4]
    world_points_h = torch.bmm(camera_points_h, camera_to_world.transpose(1, 2))  # [B, N, 4]
    return world_points_h[..., :3], camera_points


class DepthPointUnprojector(nn.Module):
    """Convert a coarse depth map into a fixed-size point set in world coordinates."""

    def __init__(self, num_points: int, use_all_if_fewer: bool = True) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.use_all_if_fewer = use_all_if_fewer

    def forward(
        self,
        depth: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        confidence: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            depth: [B, 1, H, W]
            intrinsics: [B, 3, 3] matched to depth resolution
            extrinsics: [B, 4, 4] world-to-camera for the reference view
            confidence: [B, 1, H, W]
            valid_mask: optional [B, 1, H, W]
        """
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"Expected depth with shape [B, 1, H, W], got {tuple(depth.shape)}")

        batch_size, _, height, width = depth.shape
        num_candidates = height * width
        pixel_grid = make_pixel_grid(height, width, depth.device, depth.dtype).expand(batch_size, -1, -1)
        depth_flat = depth.view(batch_size, num_candidates)

        if confidence is None:
            confidence_flat = torch.ones_like(depth_flat)
        else:
            confidence_flat = confidence.view(batch_size, num_candidates)

        valid = torch.isfinite(depth_flat) & (depth_flat > 0.0)
        if valid_mask is not None:
            valid = valid & (valid_mask.view(batch_size, num_candidates) > 0.5)

        num_points = num_candidates if self.use_all_if_fewer and num_candidates <= self.num_points else self.num_points
        indices, selected_valid = select_point_indices(valid_mask=valid, scores=confidence_flat, num_points=num_points)

        selected_depth = gather_by_index(depth_flat, indices)  # [B, N]
        selected_conf = gather_by_index(confidence_flat, indices).unsqueeze(-1)  # [B, N, 1]
        selected_pixels = gather_by_index(pixel_grid, indices)  # [B, N, 2]
        world_points, camera_points = depth_to_world_points(
            depth=selected_depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            pixel_coords=selected_pixels,
        )

        point_mask = selected_valid.unsqueeze(-1)
        selected_conf = torch.where(point_mask, selected_conf, torch.zeros_like(selected_conf))
        world_points = torch.where(point_mask, world_points, torch.zeros_like(world_points))
        camera_points = torch.where(point_mask, camera_points, torch.zeros_like(camera_points))

        return {
            "points_world": world_points,
            "points_cam": camera_points,
            "pixel_coords": selected_pixels,
            "point_confidence": selected_conf,
            "point_mask": point_mask,
            "point_indices": indices,
        }
