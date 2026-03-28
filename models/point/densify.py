from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def gather_by_index(values: Tensor, index: Tensor) -> Tensor:
    batch_size = values.shape[0]
    batch_index_shape = [batch_size] + [1] * (index.ndim - 1)
    batch_indices = torch.arange(batch_size, device=values.device).view(*batch_index_shape).expand_as(index)
    return values[batch_indices, index]


def make_split_templates(split_factor: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if split_factor <= 1:
        return torch.zeros((1, 2), device=device, dtype=dtype)
    if split_factor == 2:
        return torch.tensor([[1.0, 0.0], [-1.0, 0.0]], device=device, dtype=dtype)
    if split_factor == 4:
        return torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=device, dtype=dtype)

    angles = torch.linspace(0.0, 2.0 * math.pi, steps=split_factor + 1, device=device, dtype=dtype)[:-1]
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


class RuleBasedDensifier(nn.Module):
    """Rule-based split densification along the reference camera image plane."""

    def __init__(
        self,
        enable: bool = True,
        tau_sigma: float = 0.15,
        tau_alpha: float = 0.6,
        split_factor: int = 2,
        offset_scale: float = 0.35,
        max_new_points: int = 12000,
        use_sigma_gate: bool = True,
    ) -> None:
        super().__init__()
        self.enable = bool(enable)
        self.tau_sigma = float(tau_sigma)
        self.tau_alpha = float(tau_alpha)
        self.split_factor = int(split_factor)
        self.offset_scale = float(offset_scale)
        self.max_new_points = int(max_new_points)
        self.use_sigma_gate = bool(use_sigma_gate)

    def forward(
        self,
        points_world: Tensor,
        points_cam: Tensor,
        point_mask: Tensor,
        point_pixels: Tensor,
        sigma: Tensor,
        alpha: Tensor,
        ref_intrinsics: Tensor,
        ref_extrinsics: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, num_points, _ = points_world.shape
        zero_count = torch.zeros((batch_size, 1), device=points_world.device, dtype=torch.long)
        if (not self.enable) or self.split_factor <= 1 or self.max_new_points <= 0:
            return {
                "points_final": points_world,
                "point_final_mask": point_mask,
                "point_final_pixels": point_pixels,
                "point_final_sigma": sigma,
                "point_final_alpha": alpha,
                "points_densified": points_world[:, :0],
                "densified_mask": point_mask[:, :0],
                "densified_pixels": point_pixels[:, :0],
                "densified_sigma": sigma[:, :0],
                "densified_alpha": alpha[:, :0],
                "densified_parent_idx": torch.zeros((batch_size, 0), device=points_world.device, dtype=torch.long),
                "densified_count": zero_count,
                "densify_selected_count": zero_count,
            }

        template = make_split_templates(self.split_factor, points_world.device, points_world.dtype)  # [M, 2]
        candidate_mask = point_mask.bool()
        if self.use_sigma_gate:
            candidate_mask = candidate_mask & (sigma < self.tau_sigma)
        candidate_mask = candidate_mask & (alpha > self.tau_alpha)
        candidate_mask_flat = candidate_mask.squeeze(-1)

        score = alpha.squeeze(-1)
        if self.use_sigma_gate:
            score = score - sigma.squeeze(-1)
        score = torch.where(candidate_mask_flat, score, torch.full_like(score, -1.0e8))

        max_parent_points = max(self.max_new_points // self.split_factor, 0)
        if max_parent_points <= 0:
            max_parent_points = 1
        topk = min(max_parent_points, num_points)
        parent_idx = torch.topk(score, k=topk, dim=1, largest=True, sorted=True).indices  # [B, P]
        parent_valid = gather_by_index(candidate_mask_flat, parent_idx).unsqueeze(-1)  # [B, P, 1]

        selected_points_world = gather_by_index(points_world, parent_idx)  # [B, P, 3]
        selected_points_cam = gather_by_index(points_cam, parent_idx)  # [B, P, 3]
        selected_pixels = gather_by_index(point_pixels, parent_idx)  # [B, P, 2]
        selected_sigma = gather_by_index(sigma, parent_idx)  # [B, P, 1]
        selected_alpha = gather_by_index(alpha, parent_idx)  # [B, P, 1]

        camera_to_world = torch.linalg.inv(ref_extrinsics)[:, :3, :3]  # [B, 3, 3]
        basis_x_cam = torch.tensor([1.0, 0.0, 0.0], device=points_world.device, dtype=points_world.dtype).view(1, 1, 3)
        basis_y_cam = torch.tensor([0.0, 1.0, 0.0], device=points_world.device, dtype=points_world.dtype).view(1, 1, 3)
        basis_x_world = torch.bmm(basis_x_cam.expand(batch_size, -1, -1), camera_to_world.transpose(1, 2)).squeeze(1)  # [B, 3]
        basis_y_world = torch.bmm(basis_y_cam.expand(batch_size, -1, -1), camera_to_world.transpose(1, 2)).squeeze(1)  # [B, 3]

        depth = selected_points_cam[..., 2:3].clamp_min(1.0e-6)  # [B, P, 1]
        fx = ref_intrinsics[:, 0, 0].view(batch_size, 1, 1).clamp_min(1.0e-6)
        fy = ref_intrinsics[:, 1, 1].view(batch_size, 1, 1).clamp_min(1.0e-6)
        basis_x_scaled = basis_x_world.unsqueeze(1) * (depth / fx) * self.offset_scale  # [B, P, 3]
        basis_y_scaled = basis_y_world.unsqueeze(1) * (depth / fy) * self.offset_scale  # [B, P, 3]

        template_world = (
            template.view(1, 1, self.split_factor, 2)[..., :1] * basis_x_scaled.unsqueeze(2)
            + template.view(1, 1, self.split_factor, 2)[..., 1:] * basis_y_scaled.unsqueeze(2)
        )  # [B, P, M, 3]
        densified_points = selected_points_world.unsqueeze(2) + template_world
        densified_pixels = selected_pixels.unsqueeze(2) + self.offset_scale * template.view(1, 1, self.split_factor, 2)

        densified_sigma = selected_sigma.unsqueeze(2).expand(-1, -1, self.split_factor, -1)
        densified_alpha = selected_alpha.unsqueeze(2).expand(-1, -1, self.split_factor, -1)
        densified_mask = parent_valid.unsqueeze(2).expand(-1, -1, self.split_factor, -1)
        densified_parent_idx = parent_idx.unsqueeze(2).expand(-1, -1, self.split_factor).reshape(batch_size, -1)

        densified_points = densified_points.reshape(batch_size, -1, 3)
        densified_pixels = densified_pixels.reshape(batch_size, -1, 2)
        densified_sigma = densified_sigma.reshape(batch_size, -1, 1)
        densified_alpha = densified_alpha.reshape(batch_size, -1, 1)
        densified_mask = densified_mask.reshape(batch_size, -1, 1)

        points_final = torch.cat([points_world, densified_points], dim=1)
        point_final_mask = torch.cat([point_mask, densified_mask], dim=1)
        point_final_pixels = torch.cat([point_pixels, densified_pixels], dim=1)
        point_final_sigma = torch.cat([sigma, densified_sigma], dim=1)
        point_final_alpha = torch.cat([alpha, densified_alpha], dim=1)

        densified_count = densified_mask.squeeze(-1).sum(dim=1, keepdim=True).to(torch.long)
        densify_selected_count = parent_valid.squeeze(-1).sum(dim=1, keepdim=True).to(torch.long)

        return {
            "points_final": points_final,
            "point_final_mask": point_final_mask,
            "point_final_pixels": point_final_pixels,
            "point_final_sigma": point_final_sigma,
            "point_final_alpha": point_final_alpha,
            "points_densified": densified_points,
            "densified_mask": densified_mask,
            "densified_pixels": densified_pixels,
            "densified_sigma": densified_sigma,
            "densified_alpha": densified_alpha,
            "densified_parent_idx": densified_parent_idx,
            "densified_count": densified_count,
            "densify_selected_count": densify_selected_count,
        }
