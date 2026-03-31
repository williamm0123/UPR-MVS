from __future__ import annotations

import torch
from torch import Tensor, nn


class DepthPointUnprojector(nn.Module):
    """Sample points from depth map and unproject to world/camera coordinates."""

    def __init__(
        self,
        num_points: int = 12000,
        use_all_if_fewer: bool = True,
        sampling_strategy: str = "score_hybrid",
        topk_ratio: float = 0.7,
        coverage_grid_size: int = 6,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.use_all_if_fewer = bool(use_all_if_fewer)
        self.sampling_strategy = str(sampling_strategy).lower()
        self.topk_ratio = float(topk_ratio)
        self.coverage_grid_size = int(coverage_grid_size)

    def _sample_with_score(self, valid_idx: Tensor, valid_score: Tensor, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if valid_idx.numel() == 0:
            raise ValueError("valid_idx must not be empty when score-based sampling is used.")

        target_count = self.num_points
        sorted_order = torch.argsort(valid_score, descending=True)
        sorted_idx = valid_idx[sorted_order]

        topk_count = min(max(int(round(target_count * self.topk_ratio)), 1), target_count, sorted_idx.numel())
        selected = sorted_idx[:topk_count]

        remaining = target_count - selected.numel()
        if remaining > 0:
            candidate_pool = sorted_idx[topk_count:]
            if candidate_pool.numel() > 0:
                y = candidate_pool // width
                x = candidate_pool % width
                grid_h = max(self.coverage_grid_size, 1)
                grid_w = max(self.coverage_grid_size, 1)
                cell_h = max(int(y.max().item() + 1) // grid_h, 1)
                cell_w = max(int(x.max().item() + 1) // grid_w, 1)
                cell_ids = (y // cell_h) * grid_w + (x // cell_w)

                chosen_from_cells: list[Tensor] = []
                seen_cells: set[int] = set()
                for order_idx in range(candidate_pool.numel()):
                    cell_id = int(cell_ids[order_idx].item())
                    if cell_id in seen_cells:
                        continue
                    chosen_from_cells.append(candidate_pool[order_idx : order_idx + 1])
                    seen_cells.add(cell_id)
                    if len(chosen_from_cells) >= remaining:
                        break
                if chosen_from_cells:
                    selected = torch.cat([selected, torch.cat(chosen_from_cells, dim=0)], dim=0)

            if selected.numel() < target_count:
                fallback = sorted_idx[: min(sorted_idx.numel(), target_count - selected.numel())]
                selected = torch.cat([selected, fallback], dim=0)

        if selected.numel() > target_count:
            selected = selected[:target_count]

        valid_count = selected.numel()
        if valid_count < target_count:
            pad_count = target_count - valid_count
            pad_source = selected if valid_count > 0 else valid_idx[:1]
            pad = pad_source[torch.randint(0, pad_source.numel(), (pad_count,), device=device)]
            selected = torch.cat([selected, pad], dim=0)
            point_mask = torch.cat(
                [
                    torch.ones(valid_count, device=device, dtype=torch.bool),
                    torch.zeros(pad_count, device=device, dtype=torch.bool),
                ],
                dim=0,
            )
        else:
            point_mask = torch.ones(target_count, device=device, dtype=torch.bool)

        return selected, point_mask

    def forward(
        self,
        depth: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        confidence: Tensor,
        score: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"Expected depth [B,1,H,W], got {tuple(depth.shape)}")
        b, _, h, w = depth.shape
        device = depth.device

        depth_flat = depth.view(b, -1)
        conf_flat = confidence.view(b, -1)
        score_flat = score.view(b, -1) if score is not None else conf_flat
        if valid_mask is not None:
            mask_flat = valid_mask.view(b, -1) > 0
        else:
            mask_flat = torch.ones_like(depth_flat, dtype=torch.bool)

        all_idx = torch.arange(h * w, device=device).view(1, -1).expand(b, -1)
        sampled_idx = []
        sampled_mask = []
        for i in range(b):
            valid_idx = all_idx[i][mask_flat[i]]
            if valid_idx.numel() == 0:
                valid_idx = all_idx[i]
            valid_score = score_flat[i][mask_flat[i]]
            if valid_score.numel() == 0:
                valid_score = score_flat[i]

            if self.sampling_strategy in {"score", "score_hybrid", "importance"}:
                pick, m = self._sample_with_score(valid_idx, valid_score, width=w, device=device)
            elif valid_idx.numel() >= self.num_points:
                pick = valid_idx[torch.randperm(valid_idx.numel(), device=device)[: self.num_points]]
                m = torch.ones(self.num_points, device=device, dtype=torch.bool)
            elif self.use_all_if_fewer:
                pad_count = self.num_points - valid_idx.numel()
                if pad_count > 0:
                    pad = valid_idx[torch.randint(0, valid_idx.numel(), (pad_count,), device=device)]
                    pick = torch.cat([valid_idx, pad], dim=0)
                    m = torch.cat(
                        [
                            torch.ones(valid_idx.numel(), device=device, dtype=torch.bool),
                            torch.zeros(pad_count, device=device, dtype=torch.bool),
                        ],
                        dim=0,
                    )
                else:
                    pick = valid_idx
                    m = torch.ones(valid_idx.numel(), device=device, dtype=torch.bool)
            else:
                pick = valid_idx[torch.randint(0, valid_idx.numel(), (self.num_points,), device=device)]
                m = torch.ones(self.num_points, device=device, dtype=torch.bool)
            sampled_idx.append(pick)
            sampled_mask.append(m)

        point_indices = torch.stack(sampled_idx, dim=0)
        point_mask = torch.stack(sampled_mask, dim=0).unsqueeze(-1)

        y = (point_indices // w).float()
        x = (point_indices % w).float()
        pixel_coords = torch.stack([x, y], dim=-1)

        z = depth_flat.gather(1, point_indices).unsqueeze(-1).clamp_min(1e-6)
        fx = intrinsics[:, 0, 0].view(b, 1, 1)
        fy = intrinsics[:, 1, 1].view(b, 1, 1)
        cx = intrinsics[:, 0, 2].view(b, 1, 1)
        cy = intrinsics[:, 1, 2].view(b, 1, 1)
        x_cam = (pixel_coords[..., 0:1] - cx) * z / fx.clamp_min(1e-6)
        y_cam = (pixel_coords[..., 1:2] - cy) * z / fy.clamp_min(1e-6)
        points_cam = torch.cat([x_cam, y_cam, z], dim=-1)

        cam_h = torch.cat([points_cam, torch.ones_like(points_cam[..., :1])], dim=-1)
        extr_inv = torch.linalg.inv(extrinsics)
        points_world = torch.bmm(cam_h, extr_inv.transpose(1, 2))[..., :3]

        point_conf = conf_flat.gather(1, point_indices).unsqueeze(-1)
        point_conf = point_conf * point_mask.float()
        point_score = score_flat.gather(1, point_indices).unsqueeze(-1) * point_mask.float()

        return {
            "points_world": points_world,
            "points_cam": points_cam,
            "pixel_coords": pixel_coords,
            "point_indices": point_indices,
            "point_confidence": point_conf,
            "point_score": point_score,
            "point_mask": point_mask,
        }



def gather_by_index(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected values [B,M], got {tuple(values.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"Expected indices [B,N], got {tuple(indices.shape)}")
    return values.gather(1, indices)


def depth_to_world_points(
    depth: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor,
    pixel_coords: Tensor,
) -> tuple[Tensor, Tensor]:
    if depth.ndim != 2:
        raise ValueError(f"Expected depth [B,N], got {tuple(depth.shape)}")
    b = depth.shape[0]
    z = depth.unsqueeze(-1).clamp_min(1e-6)
    fx = intrinsics[:, 0, 0].view(b, 1, 1)
    fy = intrinsics[:, 1, 1].view(b, 1, 1)
    cx = intrinsics[:, 0, 2].view(b, 1, 1)
    cy = intrinsics[:, 1, 2].view(b, 1, 1)

    x = (pixel_coords[..., 0:1] - cx) * z / fx.clamp_min(1e-6)
    y = (pixel_coords[..., 1:2] - cy) * z / fy.clamp_min(1e-6)
    points_cam = torch.cat([x, y, z], dim=-1)

    cam_h = torch.cat([points_cam, torch.ones_like(points_cam[..., :1])], dim=-1)
    extr_inv = torch.linalg.inv(extrinsics)
    points_world = torch.bmm(cam_h, extr_inv.transpose(1, 2))[..., :3]
    return points_world, points_cam
