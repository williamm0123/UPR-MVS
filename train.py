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
    parser.add_argument(
        "--stage",
        type=str,
        default="auto",
        choices=["auto", "stage_a", "stage_b", "stage_c"],
        help="Training stage to run. Use 'auto' for automatic multi-stage training.",
    )
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


def apply_stage_config(model: nn.Module, config: dict[str, Any], stage_name: str, device: torch.device) -> None:
    """Apply stage-specific configuration to the model."""
    if "training_stages" not in config:
        return
    
    stage_cfg = config["training_stages"].get(stage_name, {})
    if not stage_cfg:
        return
    
    # Apply model modifications for this stage
    if "model" in stage_cfg:
        model_stage_cfg = stage_cfg["model"]
        model_unwrapped = unwrap_model(model)
        
        if isinstance(model_unwrapped, (UPRMVSModel, UPRMVSTransformerModel)):
            # Apply point module settings
            if "point" in model_stage_cfg:
                point_cfg = model_stage_cfg["point"]
                if "use_checkpoint" in point_cfg:
                    model_unwrapped.point_refiner.use_checkpoint = bool(point_cfg.get("use_checkpoint", False))
            
            # Apply densify settings
            if "densify" in model_stage_cfg:
                densify_cfg = model_stage_cfg["densify"]
                if "enable" in densify_cfg:
                    model_unwrapped.densifier.enable = bool(densify_cfg.get("enable", False))


def update_config_for_stage(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """Update config dictionary with stage-specific settings."""
    if "training_stages" not in config:
        return config
    
    stage_cfg = config["training_stages"].get(stage_name, {})
    if not stage_cfg:
        return config
    
    # Update train section
    if "name" in stage_cfg:
        config["train"]["stage"] = stage_cfg["name"]
    if "epochs" in stage_cfg:
        config["train"]["epochs"] = int(stage_cfg["epochs"])
    if "batch_size_per_gpu" in stage_cfg:
        config["train"]["batch_size_per_gpu"] = int(stage_cfg["batch_size_per_gpu"])
    if "grad_accum_steps" in stage_cfg:
        config["train"]["grad_accum_steps"] = int(stage_cfg["grad_accum_steps"])
    
    # Update optimizer section
    if "lr_backbone" in stage_cfg:
        config["optim"]["lr_backbone"] = float(stage_cfg["lr_backbone"])
    if "lr_coarse" in stage_cfg:
        config["optim"]["lr_coarse"] = float(stage_cfg["lr_coarse"])
    if "lr_point" in stage_cfg:
        config["optim"]["lr_point"] = float(stage_cfg["lr_point"])
    if "warmup_epochs" in stage_cfg:
        config["optim"]["warmup_epochs"] = int(stage_cfg["warmup_epochs"])
    
    # Update loss weights
    if "loss_weights" in stage_cfg:
        loss_weights = stage_cfg["loss_weights"]
        for key, value in loss_weights.items():
            if key in config["loss"]:
                config["loss"][key] = float(value)
    
    return config


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = load_config(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine training stages to run
    if args.stage == "auto" and "training_stages" in config:
        stages_to_run = list(config["training_stages"].keys())
    elif args.stage != "auto":
        stages_to_run = [args.stage.replace("_", "_")]  # e.g., "stage_a" -> ["stage_a"]
    else:
        stages_to_run = ["default"]  # Use default config without stage-specific settings
    
    use_ddp = bool(config.get("train", {}).get("use_ddp", False))
    launcher = args.launcher if use_ddp else "none"
    
    # Run training stages sequentially
    for stage_idx, stage_name in enumerate(stages_to_run):
        print(f"\n{'='*60}")
        print(f"Starting Stage {stage_idx + 1}/{len(stages_to_run)}: {stage_name.upper()}")
        print(f"{'='*60}\n")
        
        # Create stage-specific work directory
        if len(stages_to_run) > 1:
            stage_work_dir = work_dir / stage_name
            stage_work_dir.mkdir(parents=True, exist_ok=True)
        else:
            stage_work_dir = work_dir
        
        # Update config for current stage
        stage_config = update_config_for_stage(config.copy(), stage_name)
        
        # Initialize distributed mode
        ddp_cfg = init_distributed_mode(
            launcher=launcher,
            backend=str(stage_config.get("ddp", {}).get("backend", "nccl")),
        )
        set_seed(int(stage_config["train"].get("seed", 42)), ddp_cfg.rank)

        if is_main_process():
            resolved_config_path = stage_work_dir / "resolved_config.yaml"
            with resolved_config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(stage_config, handle, sort_keys=False)

        train_dataset = build_dtu_dataset(
            stage_config["data"],
            split="train",
            project_root=PROJECT_ROOT,
            config_dir=config_path.parent,
        )
        val_split = "val" if "val_list" in stage_config["data"] else "test"
        val_dataset = build_dtu_dataset(
            stage_config["data"],
            split=val_split,
            project_root=PROJECT_ROOT,
            config_dir=config_path.parent,
        )

        batch_size = resolve_train_batch_size(stage_config["train"])
        num_workers = int(stage_config["train"].get("num_workers", 4))
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

        model = build_model(stage_config).to(ddp_cfg.device)
        configure_trainable_modules(model, str(stage_config["train"].get("stage", "coarse_only")).lower())
        
        # Apply stage-specific model configurations
        apply_stage_config(model, stage_config, stage_name, ddp_cfg.device)
        
        if ddp_cfg.distributed:
            model = DDP(
                model,
                device_ids=[ddp_cfg.device.index] if ddp_cfg.device.type == "cuda" else None,
                broadcast_buffers=bool(stage_config.get("ddp", {}).get("broadcast_buffers", False)),
                find_unused_parameters=bool(stage_config.get("ddp", {}).get("find_unused_parameters", False)),
            )

        criterion = UPRMVSLoss(stage_config["loss"]).to(ddp_cfg.device)
        optimizer = build_optimizer(model, stage_config)
        scheduler = build_scheduler(optimizer, stage_config)
        use_fp16_scaler = str(stage_config["train"].get("amp_dtype", "bf16")).lower() == "fp16" and ddp_cfg.device.type == "cuda"
        scaler = build_grad_scaler(enabled=use_fp16_scaler)

        # Determine resume path for this stage
        if args.resume and stage_idx == 0:
            # User provided resume path for first stage
            resume_path = args.resume
        elif stage_idx > 0:
            # Resume from previous stage's best checkpoint
            prev_stage_name = stages_to_run[stage_idx - 1]
            prev_stage_dir = work_dir / prev_stage_name
            resume_path = str(prev_stage_dir / "checkpoints" / "best.pth")
            if not Path(resume_path).exists():
                print(f"[WARNING] Previous stage checkpoint not found: {resume_path}")
                print(f"Starting {stage_name} from scratch.")
                resume_path = ""
        else:
            resume_path = args.resume
        
        resume_mode = resolve_resume_mode(resume_path, stage_work_dir, args.resume_mode if stage_idx == 0 else "model_only")
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
            train_cfg=stage_config["train"],
            work_dir=stage_work_dir,
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
            max_epochs=int(stage_config["train"]["epochs"]),
            best_metric=best_metric,
        )
        trainer.close()
        
        # Cleanup distributed environment for this stage
        synchronize()
        cleanup_distributed()
        
        print(f"\n{'='*60}")
        print(f"Stage {stage_name.upper()} completed!")
        print(f"{'='*60}\n")
        
        # Re-initialize for next stage if needed
        if stage_idx < len(stages_to_run) - 1:
            print(f"Preparing for next stage: {stages_to_run[stage_idx + 1].upper()}...\n")
            # Re-initialize distributed mode for next stage
            if len(stages_to_run) > 1:
                ddp_cfg = init_distributed_mode(
                    launcher=launcher,
                    backend=str(stage_config.get("ddp", {}).get("backend", "nccl")),
                )

if __name__ == "__main__":
    main()
