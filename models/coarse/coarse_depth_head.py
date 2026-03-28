from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from upr_mvs.models.backbone.resnet_fpn import ResNetFPN
from upr_mvs.models.coarse.cost_volume import (
    VarianceCostVolumeBuilder,
    intrinsics_to_projection,
    sample_depth_planes,
    scale_intrinsics,
)


def _group_count(num_channels: int) -> int:
    for group_count in (16, 8, 4, 2, 1):
        if num_channels % group_count == 0:
            return group_count
    return 1


class ConvNormAct3d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = _group_count(out_channels)
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class Tiny3DUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        stem_channels = base_channels * 4
        bottleneck_channels = stem_channels * 2

        self.stem = nn.Sequential(
            ConvNormAct3d(in_channels, stem_channels),
            Residual3DBlock(stem_channels),
        )
        self.down = nn.Sequential(
            nn.Conv3d(stem_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(bottleneck_channels), bottleneck_channels),
            nn.SiLU(inplace=True),
            Residual3DBlock(bottleneck_channels),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose3d(bottleneck_channels, stem_channels, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvNormAct3d(stem_channels * 2, stem_channels),
            Residual3DBlock(stem_channels),
        )
        self.head = nn.Conv3d(stem_channels, 1, kernel_size=3, padding=1)

    def _run_block(self, module: nn.Module, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(lambda tensor: module(tensor), x, use_reentrant=False)
        return module(x)

    def forward(self, volume: Tensor) -> Tensor:
        """
        Args:
            volume: [B, C, D, H, W]

        Returns:
            logits: [B, D, H, W]
        """
        stem = self._run_block(self.stem, volume)
        down = self._run_block(self.down, stem)
        up = self._run_block(self.up, down)

        if up.shape[-3:] != stem.shape[-3:]:
            up = F.interpolate(up, size=stem.shape[-3:], mode="trilinear", align_corners=False)

        decoded = self._run_block(self.decoder, torch.cat([stem, up], dim=1))
        return self.head(decoded).squeeze(1)


class CoarseDepthHead(nn.Module):
    """Low-resolution variance cost volume + lightweight 3D UNet regularizer."""

    def __init__(
        self,
        feature_dim: int,
        depth_bins_train: int = 48,
        depth_bins_test: int = 96,
        base_channels: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.depth_bins_train = depth_bins_train
        self.depth_bins_test = depth_bins_test
        self.volume_builder = VarianceCostVolumeBuilder()
        self.regularizer = Tiny3DUNet(
            in_channels=feature_dim,
            base_channels=base_channels,
            use_checkpoint=use_checkpoint,
        )

    def forward(
        self,
        features: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        depth_range: Tensor,
        image_hw: tuple[int, int],
        num_depth_bins: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            features: [B, V, C, Hf, Wf]
            intrinsics: [B, V, 3, 3]
            extrinsics: [B, V, 4, 4]
            depth_range: [B, 2]
            image_hw: (input_image_height, input_image_width)
        """
        if features.ndim != 5:
            raise ValueError(f"Expected features with shape [B, V, C, Hf, Wf], got {tuple(features.shape)}")
        if intrinsics.shape[:2] != features.shape[:2]:
            raise ValueError("intrinsics and features must agree on [B, V].")
        if extrinsics.shape[:2] != features.shape[:2]:
            raise ValueError("extrinsics and features must agree on [B, V].")
        if features.shape[1] < 2:
            raise ValueError("CoarseDepthHead requires at least one source view.")

        batch_size, num_views, _, feat_h, feat_w = features.shape
        img_h, img_w = image_hw
        scale_x = feat_w / float(img_w)
        scale_y = feat_h / float(img_h)
        num_depth_bins = (
            num_depth_bins
            if num_depth_bins is not None
            else (self.depth_bins_train if self.training else self.depth_bins_test)
        )
        depth_values = sample_depth_planes(depth_range, num_depth_bins)

        scaled_intrinsics = scale_intrinsics(
            intrinsics.view(batch_size * num_views, 3, 3),
            scale_x=scale_x,
            scale_y=scale_y,
        ).view(batch_size, num_views, 3, 3)

        ref_projection = intrinsics_to_projection(scaled_intrinsics[:, 0], extrinsics[:, 0])
        src_projections = [
            intrinsics_to_projection(scaled_intrinsics[:, view_index], extrinsics[:, view_index])
            for view_index in range(1, num_views)
        ]

        ref_features = features[:, 0]
        src_features = [features[:, view_index] for view_index in range(1, num_views)]
        cost_volume = self.volume_builder(
            ref_features=ref_features,
            src_features=src_features,
            ref_projection=ref_projection,
            src_projections=src_projections,
            depth_values=depth_values,
        )

        logits = self.regularizer(cost_volume)
        probability = F.softmax(logits, dim=1)
        coarse_depth = torch.sum(probability * depth_values.view(batch_size, num_depth_bins, 1, 1), dim=1, keepdim=True)
        coarse_confidence = probability.max(dim=1, keepdim=True).values

        return {
            "coarse_depth": coarse_depth,
            "coarse_prob": probability,
            "coarse_confidence": coarse_confidence,
            "coarse_logits": logits,
            "depth_values": depth_values,
        }


class CoarseDepthStageModel(nn.Module):
    """Round-1 UPR-MVS model: backbone + coarse depth head."""

    def __init__(self, backbone_cfg: dict[str, Any], coarse_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = ResNetFPN(
            variant=str(backbone_cfg.get("name", "resnet34")),
            out_channels=int(backbone_cfg.get("out_channels", 128)),
            pretrained=bool(backbone_cfg.get("pretrained", False)),
            use_checkpoint=bool(backbone_cfg.get("use_checkpoint", False)),
        )
        self.feature_key = str(coarse_cfg.get("feat_key", "stage2"))
        self.coarse_head = CoarseDepthHead(
            feature_dim=int(backbone_cfg.get("out_channels", 128)),
            depth_bins_train=int(coarse_cfg.get("depth_bins_train", 48)),
            depth_bins_test=int(coarse_cfg.get("depth_bins_test", 96)),
            base_channels=int(coarse_cfg.get("base_channels", 8)),
            use_checkpoint=bool(coarse_cfg.get("use_checkpoint", False)),
        )

    def forward(self, batch: dict[str, Tensor], num_depth_bins: int | None = None) -> dict[str, Tensor]:
        images = batch["imgs"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        depth_range = batch["depth_range"]

        if images.ndim != 5:
            raise ValueError(f"Expected imgs with shape [B, V, 3, H, W], but got {tuple(images.shape)}")

        batch_size, num_views, channels, img_h, img_w = images.shape
        flat_images = images.view(batch_size * num_views, channels, img_h, img_w)
        pyramid = self.backbone(flat_images)
        if self.feature_key not in pyramid:
            available_keys = ", ".join(sorted(pyramid.keys()))
            raise KeyError(f"Unknown coarse feature key '{self.feature_key}'. Available keys: {available_keys}")

        feature_map = pyramid[self.feature_key]
        feat_channels = feature_map.shape[1]
        feat_h, feat_w = feature_map.shape[-2:]
        feature_map = feature_map.view(batch_size, num_views, feat_channels, feat_h, feat_w)

        outputs = self.coarse_head(
            features=feature_map,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_range=depth_range,
            image_hw=(img_h, img_w),
            num_depth_bins=num_depth_bins,
        )
        outputs["coarse_feat"] = feature_map[:, 0]
        return outputs
