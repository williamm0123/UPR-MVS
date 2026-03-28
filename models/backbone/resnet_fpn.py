from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


class ConvNormAct2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResNetFPN(nn.Module):
    """ResNet-FPN backbone that exposes feature maps at H/4, H/8, and H/16."""

    def __init__(
        self,
        variant: str = "resnet34",
        out_channels: int = 128,
        pretrained: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if variant not in {"resnet18", "resnet34"}:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        if variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            stage_channels = (64, 128, 256)
        else:
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            stage_channels = (64, 128, 256)

        self.use_checkpoint = use_checkpoint

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.lateral3 = nn.Conv2d(stage_channels[2], out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(stage_channels[1], out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(stage_channels[0], out_channels, kernel_size=1)

        self.out3 = ConvNormAct2d(out_channels, out_channels)
        self.out2 = ConvNormAct2d(out_channels, out_channels)
        self.out1 = ConvNormAct2d(out_channels, out_channels)

    def _forward_stage(self, module: nn.Module, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(lambda tensor: module(tensor), x, use_reentrant=False)
        return module(x)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            Dict[str, Tensor]:
                - stage1: [B, C, H/4, W/4]
                - stage2: [B, C, H/8, W/8]
                - stage3: [B, C, H/16, W/16]
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images with shape [B, 3, H, W], but got {tuple(images.shape)}")

        x = self._forward_stage(self.stem, images)  # [B, 64, H/4, W/4]
        c1 = self._forward_stage(self.layer1, x)  # [B, 64, H/4, W/4]
        c2 = self._forward_stage(self.layer2, c1)  # [B, 128, H/8, W/8]
        c3 = self._forward_stage(self.layer3, c2)  # [B, 256, H/16, W/16]

        p3 = self.out3(self.lateral3(c3))  # [B, C, H/16, W/16]
        p2 = self.out2(self.lateral2(c2) + torch.nn.functional.interpolate(p3, size=c2.shape[-2:], mode="nearest"))
        p1 = self.out1(self.lateral1(c1) + torch.nn.functional.interpolate(p2, size=c1.shape[-2:], mode="nearest"))

        return {"stage1": p1, "stage2": p2, "stage3": p3}
