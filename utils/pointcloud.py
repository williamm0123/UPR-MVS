from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def filter_valid_points(points: Tensor, mask: Tensor | None = None) -> Tensor:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {tuple(points.shape)}")
    valid = torch.isfinite(points).all(dim=-1)
    if mask is not None:
        mask_flat = mask.squeeze(-1) if mask.ndim == 2 else mask
        valid = valid & mask_flat.bool()
    return points[valid]


def write_ply_ascii(path: str | Path, points: Tensor) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points_cpu = points.detach().cpu().float()
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points_cpu.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points_cpu:
            handle.write(f"{point[0].item():.6f} {point[1].item():.6f} {point[2].item():.6f}\n")
