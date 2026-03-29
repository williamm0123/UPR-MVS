from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
# Make sibling packages importable when running `python train.py`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dtu import build_dtu_dataset
from engine.checkpoint_io import load_checkpoint, resolve_resume_path
from engine.ddp_utils import cleanup_distributed, init_distributed_mode, is_main_process, synchronize
from engine.trainer import (
    UPRMVSTrainer,
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    configure_trainable_modules,
)
from models.losses import UPRMVSLoss
from models.upr_mvs import UPRMVSModel
from models.upr_mvs_transformer import UPRMVSTransformerModel
from utils.metrics import format_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UPR-MVS with DDP/AMP and stage-aware optimization.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument("--launcher", type=str, default="none", choices=["none", "pytorch"], help="Distributed launcher.")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from.")
    parser.add_argument(
        "--resume_mode",
        type=str,
        default="auto",
        choices=["auto", "full", "model_only"],
        help="Whether to restore optimizer/scheduler/scaler states together with model weights.",
    )
    parser.add_argument("--work_dir", type=str, required=True, help="Directory for logs and checkpoints.")
    parser.add_argument("--eval_only", action="store_true", help="Run validation only.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file() and not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
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


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    backbone = str(model_cfg.get("backbone", "dinov3")).lower()
    supported_backbones = {"dinov3", "upr_mvs_legacy"}
    if backbone not in supported_backbones:
        raise ValueError(
            f"Unsupported model.backbone='{backbone}'. Supported values: {sorted(supported_backbones)}"
        )
    if backbone == "dinov3":
        return UPRMVSTransformerModel(model_cfg=model_cfg)
    return UPRMVSModel(model_cfg=model_cfg)


def resolve_train_batch_size(train_cfg: dict[str, Any]) -> int:
    stage = str(train_cfg.get("stage", "coarse_only")).lower()
    if stage == "point_refine":
        return int(train_cfg.get("batch_size_point_per_gpu", train_cfg["batch_size_per_gpu"]))
    if stage == "joint":
        return int(train_cfg.get("batch_size_joint_per_gpu", train_cfg["batch_size_per_gpu"]))
    return int(train_cfg["batch_size_per_gpu"])


def build_dataloader(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def resolve_resume_mode(resume_path: str, work_dir: Path, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if not resume_path:
        return "full"

    resume_parent = Path(resume_path).resolve().parent
    work_root = work_dir.resolve()
    try:
        in_work_tree = resume_parent.is_relative_to(work_root)
    except AttributeError:
        in_work_tree = str(resume_parent).startswith(str(work_root))
    return "full" if in_work_tree else "model_only"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
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

    if is_main_process():
        resolved_config_path = work_dir / "resolved_config.yaml"
        with resolved_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    train_dataset = build_dtu_dataset(
        config["data"],
        split="train",
        project_root=PROJECT_ROOT,
        config_dir=config_path.parent,
    )
    val_split = "val" if "val_list" in config["data"] else "test"
    val_dataset = build_dtu_dataset(
        config["data"],
        split=val_split,
        project_root=PROJECT_ROOT,
        config_dir=config_path.parent,
    )

    batch_size = resolve_train_batch_size(config["train"])
    num_workers = int(config["train"].get("num_workers", 4))
    train_loader, train_sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ddp_cfg.distributed,
        shuffle=True,
    )
    val_loader, _ = build_dataloader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=num_workers,
        distributed=ddp_cfg.distributed,
        shuffle=False,
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
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_fp16_scaler = str(config["train"].get("amp_dtype", "bf16")).lower() == "fp16" and ddp_cfg.device.type == "cuda"
    scaler = build_grad_scaler(enabled=use_fp16_scaler)

    resume_path = resolve_resume_path(work_dir, args.resume)
    resume_mode = resolve_resume_mode(resume_path, work_dir, args.resume_mode)
    start_epoch, best_metric = load_checkpoint(
        checkpoint_path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=ddp_cfg.device,
        load_training_state=resume_mode == "full",
    )

    trainer = UPRMVSTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=ddp_cfg.device,
        train_cfg=config["train"],
        work_dir=work_dir,
    )

    if args.eval_only:
        metrics = trainer.validate(val_loader)
        if is_main_process():
            print(f"[val-only] {format_metrics(metrics)}")
        trainer.close()
        synchronize()
        cleanup_distributed()
        return

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        start_epoch=start_epoch,
        max_epochs=int(config["train"]["epochs"]),
        best_metric=best_metric,
    )
    trainer.close()
    synchronize()
    cleanup_distributed()


if __name__ == "__main__":
    main()
