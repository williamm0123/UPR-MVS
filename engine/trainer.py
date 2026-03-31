from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any, cast
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None

from engine.checkpoint_io import save_checkpoint
from engine.ddp_utils import is_main_process, move_to_device, reduce_dict, unwrap_model
from models.upr_mvs_transformer import UPRMVSTransformerModel
from utils.metrics import ScalarMeter, format_metrics, tensor_dict_to_floats

# 修复 GradScaler 导入警告 - 使用新的 API
# PyTorch >= 2.0 使用 torch.amp，但 GradScaler 实际在 torch.cuda.amp 中
# 为了兼容性和避免警告，使用条件导入
try:
    # 尝试从新位置导入（PyTorch >= 2.4+）
    from torch.amp import GradScaler as AmpGradScaler  # type: ignore
except (ImportError, TypeError):
    # 回退到旧位置（PyTorch < 2.4）
    from torch.cuda.amp import GradScaler as AmpGradScaler  # type: ignore


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def configure_trainable_modules(
    model: nn.Module,
    train_stage: str,
    loss_cfg: dict[str, Any] | None = None,
) -> None:
    model_unwrapped = unwrap_model(model)
    if not isinstance(model_unwrapped, UPRMVSTransformerModel):
        return

    model_unwrapped.active_train_stage = train_stage
    loss_cfg = loss_cfg or {}

    set_requires_grad(model_unwrapped.depth_prior, True)
    set_requires_grad(model_unwrapped.feature_lifter, True)
    set_requires_grad(model_unwrapped.point_refiner, True)
    set_requires_grad(model_unwrapped.densifier, True)

    if train_stage not in {"point_refine", "joint"}:
        raise ValueError(f"Unsupported train stage for DA3-only pipeline: {train_stage}")

    depth_anything_cfg = {}
    if isinstance(getattr(model_unwrapped, "model_cfg", None), dict):
        depth_anything_cfg = getattr(model_unwrapped, "model_cfg", {}).get("depth_anything3", {})
    if not bool(depth_anything_cfg.get("trainable", False)):
        set_requires_grad(model_unwrapped.depth_prior, False)

    point_refiner = getattr(model_unwrapped, "point_refiner", None)
    if point_refiner is not None:
        sigma_head = getattr(point_refiner, "sigma_head", None)
        if sigma_head is not None and float(loss_cfg.get("uncertainty_weight", 0.0)) <= 0.0:
            set_requires_grad(sigma_head, False)

        alpha_head = getattr(point_refiner, "alpha_head", None)
        if alpha_head is not None and float(loss_cfg.get("alpha_weight", 0.0)) <= 0.0:
            set_requires_grad(alpha_head, False)


def build_optimizer(
    model: nn.Module,
    config: dict[str, Any],
    *,
    include_all_params: bool = False,
) -> torch.optim.Optimizer:
    model_unwrapped = unwrap_model(model)
    optim_cfg = config["optim"]
    train_stage = str(config["train"].get("stage", "point_refine")).lower()
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    
    # 修复类型错误：显式转换为 tuple[float, float]
    betas_list = optim_cfg.get("betas", [0.9, 0.999])
    betas: tuple[float, float] = (float(betas_list[0]), float(betas_list[1]))

    def collect_params(module: nn.Module) -> list[Tensor]:
        return [
            parameter
            for parameter in module.parameters()
            if include_all_params or parameter.requires_grad
        ]

    param_groups: list[dict[str, Any]] = []
    if not isinstance(model_unwrapped, UPRMVSTransformerModel):
        raise TypeError(
            "build_optimizer expects a UPRMVSTransformerModel in the DA3-only pipeline, "
            f"got {type(model_unwrapped).__name__}."
        )

    prior_scale = float(optim_cfg.get("joint_prior_lr_scale", 1.0)) if train_stage == "joint" else 1.0
    prior_params = collect_params(model_unwrapped.depth_prior)
    point_params = collect_params(model_unwrapped.feature_lifter) + collect_params(model_unwrapped.point_refiner)
    point_params.extend(collect_params(model_unwrapped.densifier))
    if prior_params:
        param_groups.append(
            {
                "name": "depth_prior",
                "params": prior_params,
                "lr": float(optim_cfg.get("lr_prior", 0.0)) * prior_scale,
            }
        )
    if point_params:
        param_groups.append(
            {
                "name": "point",
                "params": point_params,
                "lr": float(optim_cfg.get("lr_point", 4.0e-4)),
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters found when building the optimizer.")

    name = str(optim_cfg.get("name", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unsupported optimizer: {name}")


def update_optimizer_lrs(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> None:
    optim_cfg = config["optim"]
    train_stage = str(config["train"].get("stage", "point_refine")).lower()
    prior_scale = float(optim_cfg.get("joint_prior_lr_scale", 1.0)) if train_stage == "joint" else 1.0
    lr_by_group = {
        "depth_prior": float(optim_cfg.get("lr_prior", 0.0)) * prior_scale,
        "point": float(optim_cfg.get("lr_point", 4.0e-4)),
    }
    fallback_order = ("depth_prior", "point")
    for idx, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", fallback_order[min(idx, len(fallback_order) - 1)]))
        if group_name in lr_by_group:
            group["lr"] = lr_by_group[group_name]


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    scheduler_cfg = config.get("scheduler", {"name": "cosine", "min_lr": 1.0e-6})
    optim_cfg = config.get("optim", {})  # 新增：从 config 读取 optim 配置
    train_cfg = config.get("train", {})
    
    name = str(scheduler_cfg.get("name", "cosine")).lower()
    
    # 新增：warmup 支持
    warmup_epochs = int(optim_cfg.get("warmup_epochs", 0))
    warmup_lr_init = float(optim_cfg.get("warmup_lr_init", 1.0e-6))
    
    if name == "cosine":
        epochs = int(train_cfg.get("epochs", 1))
        min_lr = float(scheduler_cfg.get("min_lr", 1.0e-6))
        
        # 如果有 warmup，使用 SequentialLR
        if warmup_epochs > 0:
            # 使用最大的学习率作为基准
            max_lr = max(
                float(optim_cfg.get("lr_prior", 0.0)),
                float(optim_cfg.get("lr_point", 4.0e-4))
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_lr_init / max_lr,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(epochs - warmup_epochs, 1),
                eta_min=min_lr
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    
    if name == "multistep":
        milestones_list = scheduler_cfg.get("milestones", [int(train_cfg.get("epochs", 1)) // 2])
        milestones = [int(step) for step in milestones_list]
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    raise ValueError(f"Unsupported scheduler: {name}")


def build_grad_scaler(enabled: bool) -> Any:
    # 修复 GradScaler API 警告
    # PyTorch >= 2.4: 使用 torch.amp.GradScaler('cuda', ...)
    # PyTorch < 2.4: 使用 torch.cuda.amp.GradScaler(...)
    try:
        # 尝试新 API（需要检查是否接受 device_type 参数）
        import inspect
        sig = inspect.signature(AmpGradScaler.__init__)
        params = list(sig.parameters.keys())
        
        if 'device_type' in params:
            # 新 API: torch.amp.GradScaler(device_type='cuda', enabled=...)
            return AmpGradScaler(device_type='cuda', enabled=enabled)  # type: ignore
        else:
            # 旧 API: torch.cuda.amp.GradScaler(enabled=...)
            return AmpGradScaler(enabled=enabled)  # type: ignore
    except Exception:
        # 如果出错，回退到最简单的用法
        return AmpGradScaler(enabled=enabled)  # type: ignore


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
        self.monitor_key = self._resolve_monitor_key(train_cfg)
        self.global_step = 0

        tensorboard_cfg = train_cfg.get("tensorboard", {}) if isinstance(train_cfg.get("tensorboard", {}), dict) else {}
        self.tb_enabled = bool(tensorboard_cfg.get("enable", True))
        self.tb_scalar_interval = int(tensorboard_cfg.get("scalar_interval", train_cfg.get("log_interval", 20)))
        self.tb_image_interval = int(tensorboard_cfg.get("image_interval", 200))
        self.tb_feature_stage = str(tensorboard_cfg.get("feature_stage", "stage2"))
        tb_log_dir = str(tensorboard_cfg.get("log_dir", "tensorboard")).strip()
        self.tb_log_dir = work_dir if tb_log_dir in {"", ".", "./"} else work_dir / tb_log_dir
        self.tb_writer = None
        if self.tb_enabled and is_main_process():
            if SummaryWriter is None:
                warnings.warn("TensorBoard is enabled but the tensorboard package is not installed.", stacklevel=2)
            else:
                self.tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir))

    @staticmethod
    def _resolve_monitor_key(train_cfg: dict[str, Any]) -> str:
        default_monitor_key = "point_abs_error"
        monitor_key = str(train_cfg.get("monitor_key", "auto"))
        return default_monitor_key if monitor_key == "auto" else monitor_key

    def update_runtime(
        self,
        *,
        train_cfg: dict[str, Any] | None = None,
        scheduler: Any | None = None,
        work_dir: Path | None = None,
    ) -> None:
        if train_cfg is not None:
            self.train_cfg = train_cfg
            self.monitor_key = self._resolve_monitor_key(train_cfg)
            tensorboard_cfg = (
                train_cfg.get("tensorboard", {})
                if isinstance(train_cfg.get("tensorboard", {}), dict)
                else {}
            )
            self.tb_scalar_interval = int(tensorboard_cfg.get("scalar_interval", train_cfg.get("log_interval", 20)))
            self.tb_image_interval = int(tensorboard_cfg.get("image_interval", 200))
            self.tb_feature_stage = str(tensorboard_cfg.get("feature_stage", "stage2"))
        if scheduler is not None:
            self.scheduler = scheduler
        if work_dir is not None:
            self.work_dir = work_dir

    @staticmethod
    def _normalize_image(image: Tensor) -> Tensor:
        image = image.detach().float().cpu()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with 2 or 3 dims, got {tuple(image.shape)}")
        return image.clamp(0.0, 1.0)

    @staticmethod
    def _normalize_map(vis_map: Tensor) -> Tensor:
        vis_map = vis_map.detach().float().cpu()
        if vis_map.ndim == 2:
            vis_map = vis_map.unsqueeze(0)
        if vis_map.ndim != 3:
            raise ValueError(f"Expected visualization tensor with 2 or 3 dims, got {tuple(vis_map.shape)}")
        finite = torch.isfinite(vis_map)
        if not finite.any():
            return torch.zeros_like(vis_map)
        vis_map = torch.where(finite, vis_map, torch.zeros_like(vis_map))
        min_value = vis_map.min()
        max_value = vis_map.max()
        if (max_value - min_value).abs() < 1.0e-6:
            return torch.zeros_like(vis_map)
        return (vis_map - min_value) / (max_value - min_value)

    def _log_scalars(self, prefix: str, metrics: dict[str, float], step: int) -> None:
        if self.tb_writer is None:
            return
        for key, value in metrics.items():
            self.tb_writer.add_scalar(f"{prefix}/{key}", value, step)

    def _log_visuals(self, prefix: str, batch: dict[str, Tensor], outputs: dict[str, Tensor], step: int) -> None:
        if self.tb_writer is None:
            return
        if "imgs" in batch:
            self.tb_writer.add_image(f"{prefix}/ref_image", self._normalize_image(batch["imgs"][0, 0]), step)
        if "depth_gt" in batch:
            self.tb_writer.add_image(f"{prefix}/depth_gt", self._normalize_map(batch["depth_gt"][0]), step)
        if "coarse_depth" in outputs:
            coarse_depth = outputs["coarse_depth"][0]
            if "depth_gt" in batch:
                coarse_depth = F.interpolate(
                    coarse_depth.unsqueeze(0),
                    size=batch["depth_gt"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            self.tb_writer.add_image(f"{prefix}/coarse_depth", self._normalize_map(coarse_depth), step)
        feature_pyramid = outputs.get("feature_pyramid")
        if isinstance(feature_pyramid, dict):
            # 修复类型错误：使用 .get() 方法避免类型检查问题
            feature_tensor = feature_pyramid.get(self.tb_feature_stage)
            if feature_tensor is not None:
                feature_stage = feature_tensor[0, 0]
                feature_map = feature_stage.abs().mean(dim=0, keepdim=True)
                if "imgs" in batch:
                    feature_map = F.interpolate(
                        feature_map.unsqueeze(0),
                        size=batch["imgs"].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                self.tb_writer.add_image(f"{prefix}/feature_{self.tb_feature_stage}", self._normalize_map(feature_map), step)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

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
        
        # 新增：NaN 检查配置
        use_nan_check = bool(self.train_cfg.get("use_loss_nan_check", False))
        early_stop_on_nan = bool(self.train_cfg.get("early_stop_on_nan", False))

        self.optimizer.zero_grad(set_to_none=True)
        meter = ScalarMeter()
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            batch = move_to_device(batch, self.device)
            with autocast_context(self.device, amp_dtype):
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)
                scaled_loss = loss_dict["loss_total"] / accum_steps

            # 新增：NaN/Inf 检查
            if use_nan_check:
                if not torch.isfinite(scaled_loss):
                    print(f"[WARNING] Loss is not finite at step {step}. Skipping batch.")
                    if early_stop_on_nan:
                        print("[ERROR] Early stopping due to NaN loss.")
                        break
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = ((step + 1) % accum_steps == 0) or (step + 1 == len(train_loader))
            if should_step:
                self._maybe_step_optimizer(grad_clip)

            reduced = reduce_dict({key: value.detach() for key, value in loss_dict.items()}, average=True)
            meter.update(tensor_dict_to_floats(reduced))
            reduced_floats = tensor_dict_to_floats(reduced)
            if self.tb_writer is not None and (self.global_step % max(self.tb_scalar_interval, 1) == 0):
                self._log_scalars("train", reduced_floats, self.global_step)
                self.tb_writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            if self.tb_writer is not None and self.tb_image_interval > 0 and (self.global_step % self.tb_image_interval == 0):
                self._log_visuals("train", batch, outputs, self.global_step)

            if is_main_process() and ((step + 1) % log_interval == 0 or step + 1 == len(train_loader)):
                elapsed = time.time() - start_time
                averages = meter.averages()
                print(
                    f"[train] epoch={epoch:03d} step={step + 1:04d}/{len(train_loader):04d} "
                    f"{format_metrics(averages, ['loss_total', 'loss_coarse', 'loss_chamfer', 'point_abs_error', 'avg_sigma', 'num_final_points'])} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.6e} time={elapsed:.2f}s"
                )
            self.global_step += 1

        return meter.averages()

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader[Any]) -> dict[str, float]:
        self.model.eval()
        amp_dtype = str(self.train_cfg.get("amp_dtype", "bf16"))
        meter = ScalarMeter()
        logged_visuals = False

        for batch in val_loader:
            batch = move_to_device(batch, self.device)
            with autocast_context(self.device, amp_dtype):
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)
            reduced = reduce_dict({key: value.detach() for key, value in loss_dict.items()}, average=True)
            meter.update(tensor_dict_to_floats(reduced))
            if self.tb_writer is not None and not logged_visuals:
                self._log_visuals("val", batch, outputs, self.global_step)
                logged_visuals = True

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
            # 修复类型检查：使用 cast 断言为 DistributedSampler
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                cast(DistributedSampler, train_sampler).set_epoch(epoch)

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics: dict[str, float] | None = None

            if (epoch + 1) % val_interval == 0:
                val_metrics = self.validate(val_loader)
                if self.tb_writer is not None:
                    self._log_scalars("val", val_metrics, self.global_step)
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
