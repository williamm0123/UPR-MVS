from __future__ import annotations

import torch
from torch import Tensor, nn

from upr_mvs.models.point.knn import gather_by_index


class EdgeConvMLP(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
        )


class EdgeConvBlock(nn.Module):
    """DGCNN-style EdgeConv block with max aggregation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mlp = EdgeConvMLP(in_channels * 2 + 3, out_channels)

    def forward(self, points: Tensor, features: Tensor, neighbor_idx: Tensor) -> Tensor:
        """
        Args:
            points: [B, N, 3]
            features: [B, N, C]
            neighbor_idx: [B, N, K]
        Returns:
            updated_features: [B, N, Cout]
        """
        num_neighbors = neighbor_idx.shape[-1]
        central_feat = features.unsqueeze(2).expand(-1, -1, num_neighbors, -1)
        central_points = points.unsqueeze(2).expand(-1, -1, num_neighbors, -1)
        neighbor_feat = gather_by_index(features, neighbor_idx)  # [B, N, K, C]
        neighbor_points = gather_by_index(points, neighbor_idx)  # [B, N, K, 3]

        edge_input = torch.cat(
            [central_feat, neighbor_feat - central_feat, neighbor_points - central_points],
            dim=-1,
        )  # [B, N, K, 2C+3]
        edge_features = self.mlp(edge_input)
        return edge_features.max(dim=2).values
