from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from upr_mvs.datasets.dtu import build_dtu_dataset
from upr_mvs.engine.checkpoint_io import load_checkpoint, resolve_resume_path
from upr_mvs.engine.ddp_utils import cleanup_distributed, init_distributed_mode, is_main_process, move_to_device, reduce_dict, synchronize
from upr_mvs.engine.trainer import autocast_context, configure_trainable_modules
from upr_mvs.models.losses import UPRMVSLoss
from upr_mvs.models.losses.consistency import build_sparse_gt_points
from upr_mvs.models.upr_mvs import UPRMVSModel
from upr_mvs.models.upr_mvs_transformer import UPRMVSTransformerModel
from upr_mvs.utils.metrics import ScalarMeter, format_metrics, sparse_point_cloud_metrics, tensor_dict_to_floats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UPR-MVS checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument("--launcher", type=str, default="none", choices=["none", "pytorch"], help="Distributed launcher.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path. Defaults to latest.pth in work_dir.")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory used for checkpoints and metrics.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping, but got {type(config)!r}")
    return config


def set_seed(seed: int, rank: int) -> None:
    effective_seed = seed + rank
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    backbone = str(config["model"].get("backbone", "dinov3")).lower()
    if backbone == "dinov3":
        return UPRMVSTransformerModel(model_cfg=config["model"])
    return UPRMVSModel(model_cfg=config["model"])


def build_dataloader(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    distributed: bool,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader[Any],
    device: torch.device,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    meter = ScalarMeter()

    for batch in data_loader:
        batch = move_to_device(batch, device)
        with autocast_context(device, amp_dtype):
            outputs = model(batch)
            metric_dict = criterion(outputs, batch)

            if "points_final" in outputs:
                gt_sparse = build_sparse_gt_points(outputs, batch)
                point_metrics = sparse_point_cloud_metrics(
                    pred_points=outputs["points_final"],
                    gt_points=gt_sparse["points_gt_world"],
                    pred_mask=outputs.get("point_final_mask", outputs["point_mask"]),
                    gt_mask=gt_sparse["point_gt_mask"],
                )
                metric_dict.update(point_metrics)

        reduced = reduce_dict({key: value.detach() for key, value in metric_dict.items()}, average=True)
        meter.update(tensor_dict_to_floats(reduced))

    return meter.averages()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    use_ddp = bool(config.get("train", {}).get("use_ddp", False))
    launcher = args.launcher if use_ddp else "none"
    ddp_cfg = init_distributed_mode(
        launcher=launcher,
        backend=str(config.get("ddp", {}).get("backend", "nccl")),
    )
    set_seed(int(config["train"].get("seed", 42)), ddp_cfg.rank)

    dataset = build_dtu_dataset(config["data"], split=args.split)
    data_loader, _ = build_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=int(config["train"].get("num_workers", 4)),
        distributed=ddp_cfg.distributed,
    )

    model = build_model(config).to(ddp_cfg.device)
    configure_trainable_modules(model, str(config["train"].get("stage", "coarse_only")).lower())
    if ddp_cfg.distributed:
        model = DDP(
            model,
            device_ids=[ddp_cfg.device.index] if ddp_cfg.device.type == "cuda" else None,
            broadcast_buffers=bool(config.get("ddp", {}).get("broadcast_buffers", False)),
            find_unused_parameters=bool(config.get("ddp", {}).get("find_unused_parameters", False)),
        )

    criterion = UPRMVSLoss(config["loss"]).to(ddp_cfg.device)
    checkpoint_path = resolve_resume_path(work_dir, args.checkpoint)
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint specified and latest.pth was not found in work_dir.")
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device=ddp_cfg.device,
        load_training_state=False,
    )

    metrics = evaluate(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        device=ddp_cfg.device,
        amp_dtype=str(config["train"].get("amp_dtype", "bf16")),
    )

    if is_main_process():
        print(f"[test:{args.split}] {format_metrics(metrics)}")
        metrics_path = work_dir / f"metrics_{args.split}.yaml"
        with metrics_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metrics, handle, sort_keys=True)

    synchronize()
    cleanup_distributed()


if __name__ == "__main__":
    main()
