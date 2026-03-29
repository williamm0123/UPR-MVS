from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from engine.checkpoint_io import save_checkpoint
from engine.ddp_utils import is_main_process, move_to_device, reduce_dict, unwrap_model
from models.upr_mvs import UPRMVSModel
from models.upr_mvs_transformer import UPRMVSTransformerModel
from utils.metrics import ScalarMeter, format_metrics, tensor_dict_to_floats


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def configure_trainable_modules(model: nn.Module, train_stage: str) -> None:
    model_unwrapped = unwrap_model(model)
    if not isinstance(model_unwrapped, (UPRMVSModel, UPRMVSTransformerModel)):
        return

    set_requires_grad(model_unwrapped.backbone, True)
    set_requires_grad(model_unwrapped.cvt, True)
    set_requires_grad(model_unwrapped.feature_lifter, True)
    set_requires_grad(model_unwrapped.point_refiner, True)

    if train_stage == "point_refine":
        set_requires_grad(model_unwrapped.backbone, False)
        set_requires_grad(model_unwrapped.cvt, False)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    model_unwrapped = unwrap_model(model)
    optim_cfg = config["optim"]
    train_stage = str(config["train"].get("stage", "coarse_only")).lower()
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    betas = tuple(float(beta) for beta in optim_cfg.get("betas", [0.9, 0.999]))

    def collect_params(module: nn.Module) -> list[Tensor]:
        return [parameter for parameter in module.parameters() if parameter.requires_grad]

    param_groups: list[dict[str, Any]] = []
    if isinstance(model_unwrapped, (UPRMVSModel, UPRMVSTransformerModel)):
        coarse_scale = float(optim_cfg.get("joint_coarse_lr_scale", 0.2)) if train_stage == "joint" else 1.0
        backbone_params = collect_params(model_unwrapped.backbone)
        coarse_params = collect_params(model_unwrapped.cvt)
        point_params = collect_params(model_unwrapped.feature_lifter) + collect_params(model_unwrapped.point_refiner)
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": float(optim_cfg.get("lr_backbone", 1.0e-4))})
        if coarse_params:
            param_groups.append({"params": coarse_params, "lr": float(optim_cfg.get("lr_coarse", 2.0e-4)) * coarse_scale})
        if point_params:
            param_groups.append({"params": point_params, "lr": float(optim_cfg.get("lr_point", 4.0e-4))})

    if not param_groups:
        raise RuntimeError("No trainable parameters found when building the optimizer.")

    name = str(optim_cfg.get("name", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    scheduler_cfg = config.get("scheduler", {"name": "cosine", "min_lr": 1.0e-6})
    name = str(scheduler_cfg.get("name", "cosine")).lower()
    if name == "cosine":
        epochs = int(config["train"]["epochs"])
        min_lr = float(scheduler_cfg.get("min_lr", 1.0e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    if name == "multistep":
        milestones = [int(step) for step in scheduler_cfg.get("milestones", [int(config["train"]["epochs"]) // 2])]
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {name}")


def build_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, amp_dtype: str) -> Any:
    if device.type != "cuda":
        return nullcontext()
    normalized = amp_dtype.lower()
    if normalized == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    if normalized == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if normalized in {"fp32", "none"}:
        return torch.autocast(device_type="cuda", enabled=False)
    raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")


class UPRMVSTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        device: torch.device,
        train_cfg: dict[str, Any],
        work_dir: Path,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.train_cfg = train_cfg
        self.work_dir = work_dir
        default_monitor_key = "depth_abs_error" if str(train_cfg.get("stage", "coarse_only")).lower() == "coarse_only" else "point_abs_error"
        monitor_key = str(train_cfg.get("monitor_key", "auto"))
        self.monitor_key = default_monitor_key if monitor_key == "auto" else monitor_key

    def _maybe_step_optimizer(self, grad_clip: float | None) -> None:
        if grad_clip is not None and grad_clip > 0.0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader[Any], epoch: int) -> dict[str, float]:
        self.model.train()
        accum_steps = int(self.train_cfg.get("grad_accum_steps", 1))
        log_interval = int(self.train_cfg.get("log_interval", 20))
        grad_clip = float(self.train_cfg.get("grad_clip", 0.0))
        amp_dtype = str(self.train_cfg.get("amp_dtype", "bf16"))

        self.optimizer.zero_grad(set_to_none=True)
        meter = ScalarMeter()
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            batch = move_to_device(batch, self.device)
            with autocast_context(self.device, amp_dtype):
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)
                scaled_loss = loss_dict["loss_total"] / accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = ((step + 1) % accum_steps == 0) or (step + 1 == len(train_loader))
            if should_step:
                self._maybe_step_optimizer(grad_clip)

            reduced = reduce_dict({key: value.detach() for key, value in loss_dict.items()}, average=True)
            meter.update(tensor_dict_to_floats(reduced))

            if is_main_process() and ((step + 1) % log_interval == 0 or step + 1 == len(train_loader)):
                elapsed = time.time() - start_time
                averages = meter.averages()
                print(
                    f"[train] epoch={epoch:03d} step={step + 1:04d}/{len(train_loader):04d} "
                    f"{format_metrics(averages, ['loss_total', 'loss_coarse', 'loss_chamfer', 'point_abs_error', 'avg_sigma', 'num_final_points'])} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.6e} time={elapsed:.2f}s"
                )

        return meter.averages()

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader[Any]) -> dict[str, float]:
        self.model.eval()
        amp_dtype = str(self.train_cfg.get("amp_dtype", "bf16"))
        meter = ScalarMeter()

        for batch in val_loader:
            batch = move_to_device(batch, self.device)
            with autocast_context(self.device, amp_dtype):
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)
            reduced = reduce_dict({key: value.detach() for key, value in loss_dict.items()}, average=True)
            meter.update(tensor_dict_to_floats(reduced))

        return meter.averages()

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader[Any],
        val_loader: torch.utils.data.DataLoader[Any],
        train_sampler: torch.utils.data.Sampler[Any] | None,
        start_epoch: int,
        max_epochs: int,
        best_metric: float,
    ) -> None:
        val_interval = int(self.train_cfg.get("val_interval", 1))
        save_interval = int(self.train_cfg.get("save_interval", 1))

        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics: dict[str, float] | None = None

            if (epoch + 1) % val_interval == 0:
                val_metrics = self.validate(val_loader)
                current_metric = val_metrics.get(self.monitor_key, float("inf"))
                if current_metric < best_metric:
                    best_metric = current_metric
                    save_checkpoint(
                        work_dir=self.work_dir,
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        best_metric=best_metric,
                        monitor_key=self.monitor_key,
                        tag="best",
                    )

            self.scheduler.step()

            if is_main_process():
                print(f"[epoch {epoch:03d}] train {format_metrics(train_metrics)}")
                if val_metrics is not None:
                    print(f"[epoch {epoch:03d}] val   {format_metrics(val_metrics)}")

            if (epoch + 1) % save_interval == 0:
                save_checkpoint(
                    work_dir=self.work_dir,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    best_metric=best_metric,
                    monitor_key=self.monitor_key,
                    tag="latest",
                )
                save_checkpoint(
                    work_dir=self.work_dir,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    best_metric=best_metric,
                    monitor_key=self.monitor_key,
                    tag=f"epoch_{epoch:03d}",
                )
