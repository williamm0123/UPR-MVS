from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from models.transformer.positional_encoding import Normalized2DPositionalEncoding
from models.transformer.side_view_attention import SideViewAttention


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 384, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (h, w)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        name: str = "dinov3_vitb16",
        pretrained: str | None = None,
        sva_layers: Sequence[int] = (3, 6, 9),
        sva_dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        spec = {
            "dinov3_vits16": {"dim": 384, "depth": 12, "heads": 6},
            "dinov3_vitb16": {"dim": 768, "depth": 12, "heads": 12},
            "dinov3_vitl16": {"dim": 1024, "depth": 24, "heads": 16},
        }
        if name not in spec:
            raise ValueError(f"Unsupported DINOv3 variant: {name}")
        cfg = spec[name]
        self.embed_dim = cfg["dim"]
        self.use_checkpoint = use_checkpoint
        self.sva_layers = set(int(x) for x in sva_layers)

        self.patch_embed = PatchEmbed(embed_dim=self.embed_dim, patch_size=16)
        self.pos_enc = Normalized2DPositionalEncoding(self.embed_dim)
        self.blocks = nn.ModuleList([ViTBlock(self.embed_dim, cfg["heads"]) for _ in range(cfg["depth"])])
        self.sva = nn.ModuleDict({str(i): SideViewAttention(self.embed_dim, dropout=sva_dropout) for i in self.sva_layers})

        self.out_proj = nn.ModuleDict(
            {
                "stage1": nn.Conv2d(self.embed_dim, 256, kernel_size=1),
                "stage2": nn.Conv2d(self.embed_dim, 256, kernel_size=1),
                "stage3": nn.Conv2d(self.embed_dim, 256, kernel_size=1),
            }
        )
        self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained: str | None) -> None:
        if not pretrained:
            return
        path = Path(pretrained)
        if not path.is_file():
            return
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=False)

    def _maybe_checkpoint(self, module: nn.Module, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(lambda t: module(t), x, use_reentrant=False)
        return module(x)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim == 4:
            b, _, h, w = images.shape
            tokens, (ph, pw) = self.patch_embed(images)
            pos = self.pos_enc(b, ph, pw, images.device, tokens.dtype)
            x = tokens + pos
            snaps: dict[int, Tensor] = {}
            for i, block in enumerate(self.blocks):
                x = self._maybe_checkpoint(block, x)
                if i in {len(self.blocks) // 3, (2 * len(self.blocks)) // 3, len(self.blocks) - 1}:
                    snaps[i] = x
            snap_list = sorted(snaps.items(), key=lambda kv: kv[0])
            features = {}
            for key, (_, t) in zip(("stage1", "stage2", "stage3"), snap_list):
                fmap = t.transpose(1, 2).reshape(b, self.embed_dim, ph, pw)
                projected = self.out_proj[key](fmap)
                target = (max(1, h // (8 if key == "stage1" else 16 if key == "stage2" else 32)), max(1, w // (8 if key == "stage1" else 16 if key == "stage2" else 32)))
                features[key] = F.interpolate(projected, size=target, mode="bilinear", align_corners=False)
            return features
        raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")

    def extract_feature(self, images: Tensor) -> dict[str, Tensor]:
        return self.forward(images)

    def forward_multiview(self, images: Tensor) -> dict[str, Tensor]:
        """images: [B,V,3,H,W]. Applies SVA at configured layers."""
        b, v, _, h, w = images.shape
        flat = images.view(b * v, 3, h, w)
        tokens, (ph, pw) = self.patch_embed(flat)
        pos = self.pos_enc(b * v, ph, pw, images.device, tokens.dtype)
        x = tokens + pos
        x = x.view(b, v, ph * pw, self.embed_dim)

        snaps: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            x = x.view(b * v, ph * pw, self.embed_dim)
            x = self._maybe_checkpoint(block, x)
            x = x.view(b, v, ph * pw, self.embed_dim)
            if i in self.sva_layers and v > 1:
                ref = x[:, 0]
                src = x[:, 1:].reshape(b, -1, self.embed_dim)
                x[:, 0] = self.sva[str(i)](ref, src)
            if i in {len(self.blocks) // 3, (2 * len(self.blocks)) // 3, len(self.blocks) - 1}:
                snaps[i] = x

        out: dict[str, Tensor] = {}
        snap_list = sorted(snaps.items(), key=lambda kv: kv[0])
        for key, (_, t) in zip(("stage1", "stage2", "stage3"), snap_list):
            fmap = t.view(b * v, ph * pw, self.embed_dim).transpose(1, 2).reshape(b * v, self.embed_dim, ph, pw)
            projected = self.out_proj[key](fmap)
            target = (max(1, h // (8 if key == "stage1" else 16 if key == "stage2" else 32)), max(1, w // (8 if key == "stage1" else 16 if key == "stage2" else 32)))
            projected = F.interpolate(projected, size=target, mode="bilinear", align_corners=False)
            c = projected.shape[1]
            out[key] = projected.view(b, v, c, projected.shape[-2], projected.shape[-1])
        return out
