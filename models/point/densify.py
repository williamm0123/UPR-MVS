from __future__ import annotations

import torch
from torch import Tensor, nn


class RuleBasedDensifier(nn.Module):
    def __init__(
        self,
        enable: bool = False,
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
        del ref_intrinsics, ref_extrinsics
        b, n, _ = points_world.shape
        device = points_world.device

        base = {
            "points_final": points_world,
            "point_final_mask": point_mask,
            "point_final_pixels": point_pixels,
            "point_final_sigma": sigma,
            "point_final_alpha": alpha,
            "points_densified": points_world.new_zeros((b, 0, 3)),
            "densified_mask": point_mask.new_zeros((b, 0, 1)),
            "densified_pixels": point_pixels.new_zeros((b, 0, 2)),
            "densified_sigma": sigma.new_zeros((b, 0, 1)),
            "densified_alpha": alpha.new_zeros((b, 0, 1)),
            "densified_parent_idx": torch.zeros((b, 0), device=device, dtype=torch.long),
            "densified_count": torch.zeros((b,), device=device, dtype=torch.long),
            "densify_selected_count": torch.zeros((b,), device=device, dtype=torch.long),
        }
        if not self.enable or n == 0:
            return base

        gate = (alpha.squeeze(-1) >= self.tau_alpha) & (point_mask.squeeze(-1) > 0)
        if self.use_sigma_gate:
            gate = gate & (sigma.squeeze(-1) <= self.tau_sigma)

        all_new_points = []
        all_new_mask = []
        all_new_pixels = []
        all_new_sigma = []
        all_new_alpha = []
        all_parent_idx = []
        densified_count = []
        selected_count = []

        for i in range(b):
            sel_idx = torch.nonzero(gate[i], as_tuple=False).squeeze(-1)
            selected_count.append(torch.tensor(sel_idx.numel(), device=device, dtype=torch.long))
            if sel_idx.numel() == 0:
                all_new_points.append(points_world.new_zeros((0, 3)))
                all_new_mask.append(point_mask.new_zeros((0, 1)))
                all_new_pixels.append(point_pixels.new_zeros((0, 2)))
                all_new_sigma.append(sigma.new_zeros((0, 1)))
                all_new_alpha.append(alpha.new_zeros((0, 1)))
                all_parent_idx.append(torch.zeros((0,), device=device, dtype=torch.long))
                densified_count.append(torch.tensor(0, device=device, dtype=torch.long))
                continue

            parent_world = points_world[i, sel_idx]
            parent_cam = points_cam[i, sel_idx]
            parent_pixels = point_pixels[i, sel_idx]
            parent_sigma = sigma[i, sel_idx]
            parent_alpha = alpha[i, sel_idx]

            rep = self.split_factor
            new_world = parent_world.repeat_interleave(rep, dim=0)
            new_cam = parent_cam.repeat_interleave(rep, dim=0)
            new_pixels = parent_pixels.repeat_interleave(rep, dim=0)
            new_sigma = parent_sigma.repeat_interleave(rep, dim=0)
            new_alpha = parent_alpha.repeat_interleave(rep, dim=0)
            parent_rep = sel_idx.repeat_interleave(rep)

            noise = torch.randn_like(new_cam) * self.offset_scale
            noise[..., 2] = 0.0
            new_world = new_world + noise
            new_pixels = new_pixels + noise[..., :2]

            if new_world.shape[0] > self.max_new_points:
                keep = torch.randperm(new_world.shape[0], device=device)[: self.max_new_points]
                new_world = new_world[keep]
                new_pixels = new_pixels[keep]
                new_sigma = new_sigma[keep]
                new_alpha = new_alpha[keep]
                parent_rep = parent_rep[keep]

            all_new_points.append(new_world)
            all_new_mask.append(torch.ones((new_world.shape[0], 1), device=device, dtype=point_mask.dtype))
            all_new_pixels.append(new_pixels)
            all_new_sigma.append(new_sigma)
            all_new_alpha.append(new_alpha)
            all_parent_idx.append(parent_rep)
            densified_count.append(torch.tensor(new_world.shape[0], device=device, dtype=torch.long))

        max_new = max((x.shape[0] for x in all_new_points), default=0)
        if max_new == 0:
            return base

        def pad(t: Tensor, out_shape: tuple[int, int], fill: float = 0.0) -> Tensor:
            if t.shape[0] == out_shape[0]:
                return t
            pad_rows = out_shape[0] - t.shape[0]
            fill_tensor = t.new_full((pad_rows, out_shape[1]), fill)
            return torch.cat([t, fill_tensor], dim=0)

        new_points = torch.stack([pad(x, (max_new, 3)) for x in all_new_points], dim=0)
        new_mask = torch.stack([pad(x.float(), (max_new, 1)) for x in all_new_mask], dim=0)
        new_pixels = torch.stack([pad(x, (max_new, 2)) for x in all_new_pixels], dim=0)
        new_sigma = torch.stack([pad(x, (max_new, 1)) for x in all_new_sigma], dim=0)
        new_alpha = torch.stack([pad(x, (max_new, 1)) for x in all_new_alpha], dim=0)
        new_parent = torch.stack(
            [
                torch.cat([
                    x,
                    x.new_zeros((max_new - x.shape[0],), dtype=torch.long),
                ])
                for x in all_parent_idx
            ],
            dim=0,
        )

        return {
            "points_final": torch.cat([points_world, new_points], dim=1),
            "point_final_mask": torch.cat([point_mask.float(), new_mask], dim=1),
            "point_final_pixels": torch.cat([point_pixels, new_pixels], dim=1),
            "point_final_sigma": torch.cat([sigma, new_sigma], dim=1),
            "point_final_alpha": torch.cat([alpha, new_alpha], dim=1),
            "points_densified": new_points,
            "densified_mask": new_mask,
            "densified_pixels": new_pixels,
            "densified_sigma": new_sigma,
            "densified_alpha": new_alpha,
            "densified_parent_idx": new_parent,
            "densified_count": torch.stack(densified_count, dim=0),
            "densify_selected_count": torch.stack(selected_count, dim=0),
        }
