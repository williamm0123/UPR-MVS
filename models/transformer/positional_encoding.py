from __future__ import annotations

import torch
from torch import Tensor, nn


class Normalized2DPositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2, dim)

    def forward(self, batch_size: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        pos = torch.stack([xx, yy], dim=-1).view(1, h * w, 2).repeat(batch_size, 1, 1)
        return self.proj(pos)


class FrustoconicalPositionalEncoding3D(nn.Module):
    """3D Frustoconical positional encoding projected to volume channel dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, depth_values: Tensor, h: int, w: int) -> Tensor:
        """Returns [B, D, H, W, C]."""
        b, d = depth_values.shape
        device, dtype = depth_values.device, depth_values.dtype
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        z = (depth_values - depth_values[:, :1]) / (depth_values[:, -1:] - depth_values[:, :1]).clamp_min(1e-6)
        z = z[:, :, None, None].expand(-1, -1, h, w)
        x = xx[None, None].expand(b, d, -1, -1)
        y = yy[None, None].expand(b, d, -1, -1)
        coord = torch.stack([x, y, z], dim=-1)
        return self.proj(coord)


class AdaptiveAttentionScaling(nn.Module):
    def __init__(self, init: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale
