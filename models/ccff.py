from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act: Optional[Union[str, nn.Module]], inplace: bool = True) -> nn.Module:
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act == "silu":
        module = nn.SiLU()
    elif act == "relu":
        module = nn.ReLU()
    elif act == "leaky_relu":
        module = nn.LeakyReLU()
    elif act == "gelu":
        module = nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {act}")

    if hasattr(module, "inplace"):
        module.inplace = inplace
    return module


class CBS(nn.Module):
    """Conv + BN + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        bias: bool = False,
        act: Optional[Union[str, nn.Module]] = "silu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = "silu") -> None:
        super().__init__()
        self.branch3x3 = CBS(in_channels, out_channels, 3, 1, padding=1, act=None)
        self.branch1x1 = CBS(in_channels, out_channels, 1, 1, padding=0, act=None)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.branch3x3(x) + self.branch1x1(x))


class FusionBlock(nn.Module):
    """
    RT-DETR CCFF fusion block.
    Equivalent to the original CSPRepLayer implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = CBS(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = CBS(in_channels, hidden_channels, 1, 1, act=act)
        self.blocks = nn.Sequential(
            *[RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = CBS(hidden_channels, out_channels, 1, 1, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.blocks(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(x1 + x2)


class CCFF(nn.Module):
    """
    Standalone CCFF extracted from RT-DETR's HybridEncoder.

    Input order:
    - x_low:  highest spatial resolution / lowest semantic level
    - x_mid:  middle spatial resolution
    - x_high: lowest spatial resolution / highest semantic level

    By default this module returns one fused feature map (`out_index=0`).
    Set `return_all=True` to get the original three-scale outputs from RT-DETR.
    """

    def __init__(
        self,
        in_channels: Union[Tuple[int, int, int], List[int]],
        hidden_dim: int = 256,
        depth_mult: float = 1.0,
        expansion: float = 1.0,
        act: str = "silu",
        out_index: int = 0,
        return_all: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"CCFF expects exactly 3 inputs, got {len(in_channels)}")
        if out_index not in (0, 1, 2):
            raise ValueError(f"out_index must be 0, 1 or 2, got {out_index}")

        self.in_channels = list(in_channels)
        self.hidden_dim = hidden_dim
        self.out_index = out_index
        self.return_all = return_all
        self.use_pan = self.return_all or self.out_index > 0

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
                for ch in self.in_channels
            ]
        )

        num_blocks = round(3 * depth_mult)

        self.lateral_convs = nn.ModuleList(
            [CBS(hidden_dim, hidden_dim, 1, 1, act=act) for _ in range(2)]
        )
        self.fpn_blocks = nn.ModuleList(
            [FusionBlock(hidden_dim * 2, hidden_dim, num_blocks, expansion, act) for _ in range(2)]
        )

        if self.use_pan:
            self.downsample_convs = nn.ModuleList(
                [CBS(hidden_dim, hidden_dim, 3, 2, act=act) for _ in range(2)]
            )
            self.pan_blocks = nn.ModuleList(
                [FusionBlock(hidden_dim * 2, hidden_dim, num_blocks, expansion, act) for _ in range(2)]
            )

    def _parse_inputs(
        self,
        x_low: Union[torch.Tensor, Sequence[torch.Tensor]],
        x_mid: Optional[torch.Tensor] = None,
        x_high: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_mid is None and x_high is None:
            if not isinstance(x_low, (list, tuple)) or len(x_low) != 3:
                raise ValueError("Pass either (x_low, x_mid, x_high) or a sequence of 3 tensors.")
            return x_low[0], x_low[1], x_low[2]

        if x_mid is None or x_high is None:
            raise ValueError("x_low, x_mid and x_high must all be provided.")
        return x_low, x_mid, x_high

    def forward(
        self,
        x_low: Union[torch.Tensor, Sequence[torch.Tensor]],
        x_mid: Optional[torch.Tensor] = None,
        x_high: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        feats = self._parse_inputs(x_low, x_mid, x_high)
        proj_feats = [proj(feat) for proj, feat in zip(self.input_proj, feats)]

        inner_outs = [proj_feats[-1]]
        for idx in range(2, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            block_idx = 2 - idx

            feat_high = self.lateral_convs[block_idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[-2:], mode="nearest")
            inner_out = self.fpn_blocks[block_idx](torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        if not self.use_pan:
            return inner_outs[0]

        outs = [inner_outs[0]]
        for idx in range(2):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if downsample_feat.shape[-2:] != feat_high.shape[-2:]:
                downsample_feat = F.interpolate(downsample_feat, size=feat_high.shape[-2:], mode="nearest")
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        if self.return_all:
            return outs
        return outs[self.out_index]
