from __future__ import annotations

import torch
from torch import Tensor


def build_knn_graph(
    points: Tensor,
    pixel_coords: Tensor,
    k: int,
    point_mask: Tensor,
    candidate_k: int = 64,
) -> Tensor:
    """Build simple masked KNN graph using Euclidean distance in xyz+pixel space."""
    del candidate_k  # kept for API compatibility
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points [B,N,3], got {tuple(points.shape)}")

    b, n, _ = points.shape
    feat = torch.cat([points, pixel_coords], dim=-1)
    dist = torch.cdist(feat, feat, p=2)
    mask = point_mask.squeeze(-1) > 0

    # Invalidate masked nodes both as src and dst.
    src_valid = mask.unsqueeze(-1)
    dst_valid = mask.unsqueeze(1)
    valid_pair = src_valid & dst_valid
    dist = dist.masked_fill(~valid_pair, float("inf"))

    eye = torch.eye(n, device=points.device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
    dist = dist.masked_fill(eye, float("inf"))
    k_eff = max(1, min(int(k), n - 1))
    idx = torch.topk(dist, k=k_eff, dim=-1, largest=False).indices

    # For invalid source points, return self indices to keep gather safe downstream.
    if not mask.all():
        self_idx = torch.arange(n, device=points.device).view(1, n, 1).expand(b, -1, k_eff)
        idx = torch.where(mask.unsqueeze(-1), idx, self_idx)

    return idx
