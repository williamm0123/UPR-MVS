from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from upr_mvs.datasets.dtu import build_dtu_dataset
from upr_mvs.engine.checkpoint_io import load_checkpoint, resolve_resume_path
from upr_mvs.engine.ddp_utils import move_to_device
from upr_mvs.engine.trainer import configure_trainable_modules
from upr_mvs.models.coarse.coarse_depth_head import CoarseDepthStageModel
from upr_mvs.models.upr_mvs import UPRMVSModel
from upr_mvs.utils.pointcloud import filter_valid_points, write_ply_ascii


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export UPR-MVS point clouds to PLY.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory used for checkpoints.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path. Defaults to latest.pth in work_dir.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save exported point clouds.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of samples to export.")
    parser.add_argument("--sigma_threshold", type=float, default=-1.0, help="Optional sigma filter. Negative disables it.")
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


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    train_stage = str(config.get("train", {}).get("stage", "coarse_only")).lower()
    if train_stage == "coarse_only":
        return CoarseDepthStageModel(backbone_cfg=config["model"]["backbone"], coarse_cfg=config["model"]["coarse"])
    return UPRMVSModel(model_cfg=config["model"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dtu_dataset(config["data"], split=args.split)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(config["train"].get("num_workers", 4)) > 0,
    )

    model = build_model(config).to(device)
    configure_trainable_modules(model, str(config["train"].get("stage", "coarse_only")).lower())
    checkpoint_path = resolve_resume_path(work_dir, args.checkpoint)
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint specified and latest.pth was not found in work_dir.")
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device=device,
        load_training_state=False,
    )

    model.eval()
    exported = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            outputs = model(batch)

            sample_name_field = batch["sample_name"]
            sample_name = sample_name_field[0] if isinstance(sample_name_field, list) else str(sample_name_field)
            points = outputs.get("points_final", outputs.get("points_refined"))
            if points is None:
                raise RuntimeError("Model outputs do not contain points_final or points_refined.")

            point_mask = outputs.get("point_final_mask", outputs.get("point_mask"))
            sigma = outputs.get("point_final_sigma", outputs.get("sigma"))
            points_sample = points[0]
            mask_sample = point_mask[0] if point_mask is not None else None
            if mask_sample is not None and args.sigma_threshold >= 0.0 and sigma is not None:
                sigma_sample = sigma[0]
                sigma_mask = sigma_sample < args.sigma_threshold
                mask_sample = mask_sample & sigma_mask

            valid_points = filter_valid_points(points_sample, mask_sample)
            write_ply_ascii(output_dir / f"{sample_name}.ply", valid_points)

            exported += 1
            if args.max_samples > 0 and exported >= args.max_samples:
                break


if __name__ == "__main__":
    main()
