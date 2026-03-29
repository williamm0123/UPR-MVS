from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .ddp_utils import save_on_master, unwrap_model


def resolve_resume_path(work_dir: Path, explicit_resume: str | Path | None) -> str:
    if explicit_resume:
        return str(explicit_resume)
    latest_checkpoint = work_dir / "latest.pth"
    return str(latest_checkpoint) if latest_checkpoint.is_file() else ""


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    device: torch.device,
    load_training_state: bool = True,
) -> tuple[int, float]:
    if not checkpoint_path:
        return 0, float("inf")

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    if load_training_state and optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if load_training_state and scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if load_training_state and scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1 if load_training_state else 0
    best_metric = float(checkpoint.get("best_metric", checkpoint.get("best_depth_abs_error", float("inf"))))
    return start_epoch, best_metric


def save_checkpoint(
    work_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    best_metric: float,
    monitor_key: str,
    tag: str,
) -> None:
    payload = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "monitor_key": monitor_key,
    }
    save_on_master(payload, work_dir / f"{tag}.pth")
