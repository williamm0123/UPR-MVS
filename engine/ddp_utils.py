from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn


@dataclass(frozen=True)
class DistributedConfig:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_dict(input_dict: dict[str, Tensor], average: bool = True) -> dict[str, Tensor]:
    if not is_dist_available_and_initialized():
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[key] for key in keys], dim=0)
        dist.all_reduce(values)
        if average:
            values /= float(get_world_size())
        return {key: value for key, value in zip(keys, values)}


def move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def init_distributed_mode(launcher: str = "none", backend: str = "nccl") -> DistributedConfig:
    launcher = launcher.lower()
    if launcher not in {"none", "pytorch"}:
        raise ValueError(f"Unsupported launcher: {launcher}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = launcher == "pytorch" and world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not is_dist_available_and_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()

    return DistributedConfig(
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def save_on_master(payload: Any, path: str | os.PathLike[str]) -> None:
    if is_main_process():
        torch.save(payload, path)
