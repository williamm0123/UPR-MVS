from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
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
from engine.ddp_utils import cleanup_distributed, init_distributed_mode, is_main_process, synchronize, unwrap_model
from engine.trainer import (
    UPRMVSTrainer,
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    configure_trainable_modules,
    update_optimizer_lrs,
)
from models.losses import UPRMVSLoss
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
        choices=["auto", "curriculum", "stage_a", "stage_b"],
        help="Training stage to run. Use 'curriculum' for a single-run staged schedule.",
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
    return UPRMVSTransformerModel(model_cfg=model_cfg)


def resolve_train_batch_size(train_cfg: dict[str, Any]) -> int:
    stage = str(train_cfg.get("stage", "point_refine")).lower()
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
        if not isinstance(model_unwrapped, UPRMVSTransformerModel):
            return

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
    if "lr_prior" in stage_cfg:
        config["optim"]["lr_prior"] = float(stage_cfg["lr_prior"])
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


def resolve_training_plan(config: dict[str, Any], requested_stage: str) -> tuple[list[str], str]:
    available_stages = list(config.get("training_stages", {}).keys())
    run_mode = str(config.get("train", {}).get("run_mode", "multi_stage")).lower()

    if requested_stage == "curriculum":
        if not available_stages:
            raise ValueError("Requested --stage curriculum, but config.training_stages is empty.")
        return available_stages, "curriculum"

    if requested_stage == "auto":
        if available_stages and run_mode in {"curriculum", "single_run", "single_run_curriculum"}:
            return available_stages, "curriculum"
        if available_stages:
            return available_stages, "multi_stage"
        return ["default"], "single_stage"

    return [requested_stage], "single_stage"


def build_curriculum_phases(config: dict[str, Any], stage_names: list[str]) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    epoch_cursor = 0
    for stage_name in stage_names:
        phase_config = update_config_for_stage(deepcopy(config), stage_name)
        phase_epochs = int(phase_config["train"]["epochs"])
        if phase_epochs <= 0:
            raise ValueError(f"Stage '{stage_name}' must have a positive epoch count, got {phase_epochs}.")
        phases.append(
            {
                "stage_name": stage_name,
                "config": phase_config,
                "start_epoch": epoch_cursor,
                "end_epoch": epoch_cursor + phase_epochs,
                "phase_epochs": phase_epochs,
            }
        )
        epoch_cursor += phase_epochs
    return phases


def find_curriculum_phase(phases: list[dict[str, Any]], epoch: int) -> tuple[int, int]:
    if not phases:
        raise ValueError("Curriculum phase list must not be empty.")
    for index, phase in enumerate(phases):
        start_epoch = int(phase["start_epoch"])
        end_epoch = int(phase["end_epoch"])
        if epoch < end_epoch:
            return index, max(epoch - start_epoch, 0)
    return len(phases) - 1, int(phases[-1]["phase_epochs"])


def advance_scheduler(scheduler: Any, steps: int) -> None:
    for _ in range(max(int(steps), 0)):
        scheduler.step()


def dump_resolved_config(config: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    total_start = time.time()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    
    print(f"\n{'='*60}")
    print(f"Starting UPR-MVS Training")
    print(f"{'='*60}")
    
    # Detect GPU and recommend configuration
    print(f"[0/12] Environment Detection...")
    step_start = time.time()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"           ✓ Detected GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Auto-detect environment
        if "5060Ti" in gpu_name or "5060 Ti" in gpu_name or gpu_memory < 20:
            print(f"           ⚠️  Local development environment detected")
            print(f"           💡 Recommendation: Use configs/local_training.config")
            print(f"           📊 Expected memory usage: ~12-15GB")
        elif "A100" in gpu_name and gpu_memory >= 70:
            print(f"           ✅ Server production environment detected")
            print(f"           💡 Recommendation: Use configs/server_training.config")
            print(f"           📊 Expected memory usage: DA3 prior frozen + point branch")
    else:
        print(f"           ⚠️  CUDA not available, running on CPU")
    
    print(f"      ✓ Environment check completed ({time.time() - step_start:.2f}s)\n")
    
    print(f"[1/12] Loading configuration...")
    step_start = time.time()
    config = load_config(args.config)
    print(f"      ✓ Config loaded ({time.time() - step_start:.2f}s)")
    
    # Print optimization info
    print(f"\n🔧 Optimization Settings:")
    print(f"   - Depth Prior: {config['model'].get('backbone', 'depth_anything3')}")
    print(f"   - Prior Trainable: {config['model'].get('depth_anything3', {}).get('trainable', False)}")
    print(f"   - AMP Dtype: {config['train'].get('amp_dtype', 'fp16')}")
    print(f"   - Num Workers: {config['train'].get('num_workers', 4)}")
    print(f"   - Image Size: {config['data'].get('img_h', 1024)}x{config['data'].get('img_w', 1280)}")
    print(f"   - Views: {config['data'].get('n_views', 5)}")
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    stage_names, training_mode = resolve_training_plan(config, args.stage)
    print(f"\n📚 Training Plan: mode={training_mode}, stages={stage_names}")

    use_ddp = bool(config.get("train", {}).get("use_ddp", False))
    launcher = args.launcher if use_ddp else "none"

    if training_mode == "curriculum":
        phases = build_curriculum_phases(config, stage_names)
        print(f"   - Total epochs: {sum(int(phase['phase_epochs']) for phase in phases)}")

        print(f"[curriculum] Initializing distributed mode...")
        step_start = time.time()
        ddp_cfg = init_distributed_mode(
            launcher=launcher,
            backend=str(config.get("ddp", {}).get("backend", "nccl")),
        )
        print(f"           ✓ Distributed initialized ({time.time() - step_start:.2f}s)")
        set_seed(int(config["train"].get("seed", 42)), ddp_cfg.rank)

        if is_main_process():
            dump_resolved_config(deepcopy(config), work_dir / "resolved_config_curriculum_base.yaml")
            curriculum_summary = {
                "mode": "curriculum",
                "stages": [
                    {
                        "stage_name": phase["stage_name"],
                        "start_epoch": phase["start_epoch"],
                        "end_epoch": phase["end_epoch"],
                        "train_stage": phase["config"]["train"]["stage"],
                        "batch_size_per_gpu": resolve_train_batch_size(phase["config"]["train"]),
                        "grad_accum_steps": int(phase["config"]["train"].get("grad_accum_steps", 1)),
                    }
                    for phase in phases
                ],
            }
            with (work_dir / "curriculum_plan.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(curriculum_summary, handle, sort_keys=False)

        print(f"[curriculum] Building datasets once for all phases...")
        step_start = time.time()
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
        print(f"           ✓ Datasets built ({time.time() - step_start:.2f}s)")

        print(f"[curriculum] Building model once...")
        step_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = build_model(config).to(ddp_cfg.device)
        configure_trainable_modules(model, "joint", config.get("loss"))
        print(f"           ✓ Model built ({time.time() - step_start:.2f}s)")

        if ddp_cfg.distributed:
            print(f"[curriculum] Wrapping with DDP...")
            step_start = time.time()
            model = DDP(
                model,
                device_ids=[ddp_cfg.device.index] if ddp_cfg.device.type == "cuda" else None,
                broadcast_buffers=bool(config.get("ddp", {}).get("broadcast_buffers", False)),
                find_unused_parameters=bool(config.get("ddp", {}).get("find_unused_parameters", False)),
            )
            print(f"           ✓ DDP wrapped ({time.time() - step_start:.2f}s)")

        criterion = UPRMVSLoss(deepcopy(config["loss"])).to(ddp_cfg.device)
        optimizer = build_optimizer(model, config, include_all_params=True)
        use_fp16_scaler = str(config["train"].get("amp_dtype", "bf16")).lower() == "fp16" and ddp_cfg.device.type == "cuda"
        scaler = build_grad_scaler(enabled=use_fp16_scaler)

        resume_path = resolve_resume_path(work_dir, args.resume)
        print(f"[curriculum] Loading checkpoint (resume_path={resume_path[:80] if resume_path else 'None'})...")
        step_start = time.time()
        resume_mode = resolve_resume_mode(resume_path, work_dir, args.resume_mode)
        start_epoch, best_metric = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            device=ddp_cfg.device,
            load_training_state=resume_mode == "full",
        )
        print(f"           ✓ Checkpoint loaded ({time.time() - step_start:.2f}s)")

        if start_epoch >= int(phases[-1]["end_epoch"]):
            print("[curriculum] Resume checkpoint is already past the configured curriculum. Nothing to do.")
            synchronize()
            cleanup_distributed()
            return

        start_phase_idx, local_epoch_offset = find_curriculum_phase(phases, start_epoch)
        trainer: UPRMVSTrainer | None = None

        for phase_idx in range(start_phase_idx, len(phases)):
            phase = phases[phase_idx]
            stage_name = str(phase["stage_name"])
            phase_config = deepcopy(phase["config"])
            phase_start_epoch = int(phase["start_epoch"])
            phase_end_epoch = int(phase["end_epoch"])
            phase_local_start_epoch = local_epoch_offset if phase_idx == start_phase_idx else 0
            phase_best_metric = best_metric if phase_idx == start_phase_idx and phase_local_start_epoch > 0 else float("inf")
            phase_time_start = time.time()

            print(f"\n{'='*60}")
            print(
                f"Starting Curriculum Phase {phase_idx + 1}/{len(phases)}: {stage_name.upper()} "
                f"(epochs {phase_start_epoch}-{phase_end_epoch - 1})"
            )
            print(f"{'='*60}\n")

            configure_trainable_modules(
                model,
                str(phase_config["train"].get("stage", "point_refine")).lower(),
                phase_config.get("loss"),
            )
            apply_stage_config(model, phase_config, stage_name, ddp_cfg.device)
            criterion.loss_cfg = deepcopy(phase_config["loss"])
            update_optimizer_lrs(optimizer, phase_config)

            scheduler = build_scheduler(optimizer, phase_config)
            if phase_local_start_epoch > 0:
                advance_scheduler(scheduler, phase_local_start_epoch)

            batch_size = resolve_train_batch_size(phase_config["train"])
            num_workers = int(phase_config["train"].get("num_workers", 4))
            print(f"[{stage_name}] Building dataloaders (batch_size={batch_size}, workers={num_workers})...")
            step_start = time.time()
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
            print(f"           ✓ Dataloaders built ({time.time() - step_start:.2f}s)")

            if is_main_process():
                dump_resolved_config(phase_config, work_dir / f"resolved_config_{stage_name}.yaml")

            if trainer is None:
                trainer = UPRMVSTrainer(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=ddp_cfg.device,
                    train_cfg=phase_config["train"],
                    work_dir=work_dir,
                )
            else:
                trainer.update_runtime(
                    train_cfg=phase_config["train"],
                    scheduler=scheduler,
                    work_dir=work_dir,
                )

            if args.eval_only:
                metrics = trainer.validate(val_loader)
                if is_main_process():
                    print(f"[val-only:{stage_name}] {format_metrics(metrics)}")
                trainer.close()
                synchronize()
                cleanup_distributed()
                return

            print(f"\n[{stage_name}] Starting curriculum training loop...")
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                start_epoch=phase_start_epoch + phase_local_start_epoch,
                max_epochs=phase_end_epoch,
                best_metric=phase_best_metric,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                current_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"           ✓ Peak memory for {stage_name}: {current_memory:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

            print(f"\n{'='*60}")
            print(f"Curriculum phase {stage_name.upper()} completed in {time.time() - phase_time_start:.2f}s!")
            print(f"{'='*60}\n")

        if trainer is not None:
            trainer.close()
        synchronize()
        cleanup_distributed()
    else:
        for stage_idx, stage_name in enumerate(stage_names):
            print(f"\n{'='*60}")
            print(f"Starting Stage {stage_idx + 1}/{len(stage_names)}: {stage_name.upper()}")
            print(f"{'='*60}\n")

            stage_start = time.time()
            stage_work_dir = work_dir / stage_name if len(stage_names) > 1 else work_dir
            stage_work_dir.mkdir(parents=True, exist_ok=True)
            stage_config = update_config_for_stage(deepcopy(config), stage_name)

            print(f"[{stage_name}] Initializing distributed mode...")
            step_start = time.time()
            ddp_cfg = init_distributed_mode(
                launcher=launcher,
                backend=str(stage_config.get("ddp", {}).get("backend", "nccl")),
            )
            print(f"           ✓ Distributed initialized ({time.time() - step_start:.2f}s)")
            set_seed(int(stage_config["train"].get("seed", 42)), ddp_cfg.rank)

            if is_main_process():
                dump_resolved_config(stage_config, stage_work_dir / "resolved_config.yaml")

            print(f"[{stage_name}] Building datasets...")
            step_start = time.time()
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
            print(f"           ✓ Datasets built ({time.time() - step_start:.2f}s)")

            batch_size = resolve_train_batch_size(stage_config["train"])
            num_workers = int(stage_config["train"].get("num_workers", 4))
            print(f"[{stage_name}] Building dataloaders (batch_size={batch_size}, workers={num_workers})...")
            step_start = time.time()
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
            print(f"           ✓ Dataloaders built ({time.time() - step_start:.2f}s)")

            print(f"[{stage_name}] Building model...")
            step_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = build_model(stage_config).to(ddp_cfg.device)
            print(f"           ✓ Model built ({time.time() - step_start:.2f}s)")

            configure_trainable_modules(
                model,
                str(stage_config["train"].get("stage", "point_refine")).lower(),
                stage_config.get("loss"),
            )
            apply_stage_config(model, stage_config, stage_name, ddp_cfg.device)

            if ddp_cfg.distributed:
                print(f"[{stage_name}] Wrapping with DDP...")
                step_start = time.time()
                model = DDP(
                    model,
                    device_ids=[ddp_cfg.device.index] if ddp_cfg.device.type == "cuda" else None,
                    broadcast_buffers=bool(stage_config.get("ddp", {}).get("broadcast_buffers", False)),
                    find_unused_parameters=bool(stage_config.get("ddp", {}).get("find_unused_parameters", False)),
                )
                print(f"           ✓ DDP wrapped ({time.time() - step_start:.2f}s)")

            print(f"[{stage_name}] Building criterion...")
            criterion = UPRMVSLoss(stage_config["loss"]).to(ddp_cfg.device)
            print(f"[{stage_name}] Building optimizer...")
            optimizer = build_optimizer(model, stage_config)
            print(f"[{stage_name}] Building scheduler...")
            scheduler = build_scheduler(optimizer, stage_config)
            print(f"[{stage_name}] Building GradScaler...")
            use_fp16_scaler = str(stage_config["train"].get("amp_dtype", "bf16")).lower() == "fp16" and ddp_cfg.device.type == "cuda"
            scaler = build_grad_scaler(enabled=use_fp16_scaler)

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

            if args.resume and stage_idx == 0:
                resume_path = args.resume
            elif stage_idx > 0:
                prev_stage_name = stage_names[stage_idx - 1]
                prev_stage_dir = work_dir / prev_stage_name
                resume_path = str(prev_stage_dir / "best.pth")
                if not Path(resume_path).exists():
                    print(f"[WARNING] Previous stage checkpoint not found: {resume_path}")
                    print(f"Starting {stage_name} from scratch.")
                    resume_path = ""
            else:
                resume_path = args.resume

            print(f"[{stage_name}] Loading checkpoint (resume_path={resume_path[:80] if resume_path else 'None'})...")
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

            if args.eval_only:
                metrics = trainer.validate(val_loader)
                if is_main_process():
                    print(f"[val-only] {format_metrics(metrics)}")
                trainer.close()
                synchronize()
                cleanup_distributed()
                return

            print(f"\n[{stage_name}] Starting training loop...")
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                start_epoch=start_epoch,
                max_epochs=int(stage_config["train"]["epochs"]),
                best_metric=best_metric,
            )
            trainer.close()

            synchronize()
            if torch.cuda.is_available():
                print(f"\n[{stage_name}] Cleaning up GPU memory...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                current_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"           ✓ Peak memory for {stage_name}: {current_memory:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

            cleanup_distributed()
            print(f"\n{'='*60}")
            print(f"Stage {stage_name.upper()} completed in {time.time() - stage_start:.2f}s!")
            print(f"{'='*60}\n")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All stages completed! Total time: {total_time:.2f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
