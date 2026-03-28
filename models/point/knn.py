from __future__ import annotations

import torch
from torch import Tensor


def gather_by_index(values: Tensor, index: Tensor) -> Tensor:
    if values.ndim < 2:
        raise ValueError(f"Expected values with shape [B, N, ...], got {tuple(values.shape)}")
    if index.shape[0] != values.shape[0]:
        raise ValueError("Batch size mismatch between values and index.")

    batch_size = values.shape[0]
    batch_index_shape = [batch_size] + [1] * (index.ndim - 1)
    batch_indices = torch.arange(batch_size, device=values.device).view(*batch_index_shape).expand_as(index)
    return values[batch_indices, index]


def build_knn_graph(
    points: Tensor,
    pixel_coords: Tensor,
    k: int,
    point_mask: Tensor | None = None,
    candidate_k: int = 64,
    chunk_size: int = 1024,
) -> Tensor:
    """
    Build a pixel-aware kNN graph in chunks to avoid allocating a full NxN matrix.

    Args:
        points: [B, N, 3]
        pixel_coords: [B, N, 2]
        k: number of neighbors.
        point_mask: optional [B, N, 1] or [B, N]
    Returns:
        neighbor_idx: [B, N, k]
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [B, N, 3], got {tuple(points.shape)}")
    if pixel_coords.shape[:2] != points.shape[:2] or pixel_coords.shape[-1] != 2:
        raise ValueError("pixel_coords must have shape [B, N, 2] and align with points.")

    batch_size, num_points, _ = points.shape
    if num_points <= 1:
        return torch.zeros((batch_size, num_points, 1), device=points.device, dtype=torch.long)

    effective_k = min(max(k, 1), max(num_points - 1, 1))
    candidate_k = min(max(candidate_k, effective_k + 1), num_points)

    if point_mask is None:
        valid_mask = torch.ones((batch_size, num_points), device=points.device, dtype=torch.bool)
    else:
        valid_mask = point_mask.squeeze(-1).bool() if point_mask.ndim == 3 else point_mask.bool()

    all_neighbor_chunks: list[Tensor] = []
    all_indices = torch.arange(num_points, device=points.device).view(1, num_points).expand(batch_size, -1)

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        query_pixels = pixel_coords[:, start:end]  # [B, Q, 2]
        query_points = points[:, start:end]  # [B, Q, 3]
        query_valid = valid_mask[:, start:end]  # [B, Q]

        pixel_dist = torch.cdist(query_pixels, pixel_coords)  # [B, Q, N]
        pixel_dist = pixel_dist.masked_fill(~valid_mask.unsqueeze(1), float("inf"))

        candidate_indices = torch.topk(pixel_dist, k=candidate_k, dim=-1, largest=False, sorted=False).indices
        candidate_points = gather_by_index(points, candidate_indices)  # [B, Q, C, 3]
        point_dist = (candidate_points - query_points.unsqueeze(2)).square().sum(dim=-1)  # [B, Q, C]

        candidate_valid = gather_by_index(valid_mask.float().unsqueeze(-1), candidate_indices).squeeze(-1) > 0.5
        point_dist = point_dist.masked_fill(~candidate_valid, float("inf"))

        query_indices = all_indices[:, start:end].unsqueeze(-1)  # [B, Q, 1]
        point_dist = point_dist.masked_fill(candidate_indices == query_indices, float("inf"))
        point_dist = point_dist.masked_fill(~query_valid.unsqueeze(-1), float("inf"))

        local_neighbor = torch.topk(point_dist, k=effective_k, dim=-1, largest=False, sorted=False).indices
        neighbor_indices = torch.gather(candidate_indices, dim=-1, index=local_neighbor)

        if effective_k < k:
            pad = neighbor_indices[..., :1].expand(-1, -1, k - effective_k)
            neighbor_indices = torch.cat([neighbor_indices, pad], dim=-1)

        invalid_queries = ~query_valid.unsqueeze(-1)
        neighbor_indices = torch.where(invalid_queries, query_indices.expand_as(neighbor_indices), neighbor_indices)
        all_neighbor_chunks.append(neighbor_indices)

    return torch.cat(all_neighbor_chunks, dim=1)
