from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..transformer.utils import scale_intrinsics
from ..point.unproject import depth_to_world_points, gather_by_index


def masked_mean(values: Tensor, mask: Tensor, eps: float = 1.0e-6) -> Tensor:
    mask_float = mask.float()
    numerator = (values * mask_float).sum()
    denominator = mask_float.sum().clamp_min(eps)
    return numerator / denominator


def downsample_depth_and_mask(depth_gt: Tensor, mask: Tensor, target_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    depth_small = F.interpolate(depth_gt, size=target_hw, mode="bilinear", align_corners=False)
    mask_small = F.interpolate(mask.float(), size=target_hw, mode="nearest") > 0.5
    return depth_small, mask_small


def compute_coarse_depth_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor], loss_cfg: dict[str, float]) -> dict[str, Tensor]:
    coarse_depth = outputs["coarse_depth"]  # [B, 1, Hc, Wc]
    depth_gt = batch["depth_gt"]
    mask = batch["mask"]
    has_depth_gt = batch["has_depth_gt"].view(-1, 1, 1, 1)

    target_depth, target_mask = downsample_depth_and_mask(depth_gt, mask, coarse_depth.shape[-2:])
    valid_mask = target_mask & has_depth_gt & torch.isfinite(target_depth) & (target_depth > 0.0)

    if valid_mask.any():
        loss_coarse = F.smooth_l1_loss(coarse_depth[valid_mask], target_depth[valid_mask])
        abs_error = (coarse_depth - target_depth).abs()
        depth_abs_error = masked_mean(abs_error, valid_mask)
        thres_2mm = masked_mean((abs_error < 2.0).float(), valid_mask)
        thres_4mm = masked_mean((abs_error < 4.0).float(), valid_mask)
        thres_8mm = masked_mean((abs_error < 8.0).float(), valid_mask)
    else:
        zero = coarse_depth.sum() * 0.0
        loss_coarse = zero
        depth_abs_error = zero
        thres_2mm = zero
        thres_4mm = zero
        thres_8mm = zero

    return {
        "loss_coarse": loss_coarse * float(loss_cfg.get("coarse_weight", 1.0)),
        "depth_abs_error": depth_abs_error,
        "thres_2mm": thres_2mm,
        "thres_4mm": thres_4mm,
        "thres_8mm": thres_8mm,
    }


def build_sparse_gt_points(outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
    coarse_depth = outputs["coarse_depth"]  # [B, 1, Hc, Wc]
    point_indices = outputs["point_indices"]  # [B, N]
    point_pixels = outputs["point_pixel_coords"]  # [B, N, 2]

    batch_size, _, coarse_h, coarse_w = coarse_depth.shape
    img_h, img_w = batch["imgs"].shape[-2], batch["imgs"].shape[-1]
    depth_gt_small, mask_small = downsample_depth_and_mask(batch["depth_gt"], batch["mask"], (coarse_h, coarse_w))
    has_depth_gt = batch["has_depth_gt"].view(-1, 1)

    gt_depth_flat = depth_gt_small.view(batch_size, -1)
    gt_mask_flat = mask_small.view(batch_size, -1) & has_depth_gt
    point_gt_depth = gather_by_index(gt_depth_flat, point_indices)  # [B, N]
    point_gt_mask = gather_by_index(gt_mask_flat, point_indices).unsqueeze(-1)  # [B, N, 1]

    ref_intrinsics = batch["intrinsics"][:, 0]
    ref_extrinsics = batch["extrinsics"][:, 0]
    scaled_ref_intrinsics = scale_intrinsics(
        ref_intrinsics,
        scale_x=coarse_w / float(img_w),
        scale_y=coarse_h / float(img_h),
    )
    gt_points_world, gt_points_cam = depth_to_world_points(
        depth=point_gt_depth,
        intrinsics=scaled_ref_intrinsics,
        extrinsics=ref_extrinsics,
        pixel_coords=point_pixels,
    )
    gt_points_world = torch.where(point_gt_mask, gt_points_world, torch.zeros_like(gt_points_world))
    gt_points_cam = torch.where(point_gt_mask, gt_points_cam, torch.zeros_like(gt_points_cam))
    return {
        "point_gt_depth": point_gt_depth.unsqueeze(-1),
        "point_gt_mask": point_gt_mask,
        "points_gt_world": gt_points_world,
        "points_gt_cam": gt_points_cam,
    }


def compute_point_l1_error(pred_points: Tensor, gt_points: Tensor, point_mask: Tensor) -> tuple[Tensor, Tensor]:
    point_error = (pred_points - gt_points).abs().mean(dim=-1, keepdim=True)  # [B, N, 1]
    point_error = torch.where(point_mask, point_error, torch.zeros_like(point_error))
    mean_error = masked_mean(point_error, point_mask)
    return mean_error, point_error


def alpha_target_from_error(point_error: Tensor, point_mask: Tensor, threshold: float) -> Tensor:
    target = (point_error < threshold).float()
    return target * point_mask.float()
