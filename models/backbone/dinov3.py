from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..dinov3.vision_transformer import (
    DinoVisionTransformer,
    vit_7b,
    vit_base,
    vit_giant2,
    vit_huge2,
    vit_large,
    vit_small,
    vit_so400m,
)
from ..transformer.side_view_attention import SideViewAttention


class DinoV3Backbone(nn.Module):
    STAGE_NAMES = ("stage1", "stage2", "stage3")
    STAGE_SCALES = {"stage1": 8, "stage2": 16, "stage3": 32}
    VARIANT_SPECS = {
        "dinov3_vits16": (vit_small, 384),
        "dinov3_vitb16": (vit_base, 768),
        "dinov3_vitl16": (vit_large, 1024),
        "dinov3_vitso400m16": (vit_so400m, 1152),
        "dinov3_vith16": (vit_huge2, 1280),
        "dinov3_vitg16": (vit_giant2, 1536),
        "dinov3_vit7b16": (vit_7b, 4096),
        "vit_small": (vit_small, 384),
        "vit_base": (vit_base, 768),
        "vit_large": (vit_large, 1024),
        "vit_so400m": (vit_so400m, 1152),
        "vit_huge2": (vit_huge2, 1280),
        "vit_giant2": (vit_giant2, 1536),
        "vit_7b": (vit_7b, 4096),
    }

    def __init__(
        self,
        name: str = "dinov3_vitb16",
        pretrained: str | None = None,
        output_layers: Sequence[int] = (3, 7, 11),
        sva_layers: Sequence[int] = (3, 6, 9),
        sva_dropout: float = 0.0,
        use_checkpoint: bool = False,
        attention_backend: str = "auto",
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        builder, embed_dim = self._resolve_variant(name)
        self.encoder: DinoVisionTransformer = builder(patch_size=16, attention_backend=attention_backend)
        self.encoder.init_weights()
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.output_block_indices = tuple(
            self._resolve_one_based_block_indices(output_layers, depth=self.encoder.n_blocks, name="output_layers")
        )
        if len(self.output_block_indices) != len(self.STAGE_NAMES):
            raise ValueError(
                f"Expected exactly {len(self.STAGE_NAMES)} output layers, got {len(self.output_block_indices)}."
            )

        # Keep SVA on the repo's historical zero-based convention to avoid
        # silently shifting old configs by one block.
        self.sva_block_indices = set(self._resolve_zero_based_block_indices(sva_layers, depth=self.encoder.n_blocks))
        self.token_offset = self.encoder.n_storage_tokens + 1
        self.sva = nn.ModuleDict(
            {str(idx): SideViewAttention(self.embed_dim, dropout=sva_dropout) for idx in sorted(self.sva_block_indices)}
        )
        self.out_proj = nn.ModuleDict(
            {stage_key: nn.Conv2d(self.embed_dim, out_dim, kernel_size=1) for stage_key in self.STAGE_NAMES}
        )

        self.load_pretrained(pretrained)

    @classmethod
    def _resolve_variant(cls, name: str) -> tuple[callable, int]:
        key = str(name).lower()
        if key not in cls.VARIANT_SPECS:
            raise ValueError(f"Unsupported DINOv3 variant: {name}")
        return cls.VARIANT_SPECS[key]

    @staticmethod
    def _resolve_one_based_block_indices(indices: Sequence[int], depth: int, name: str) -> list[int]:
        resolved: list[int] = []
        for idx in indices:
            block_index = int(idx)
            if block_index < 1 or block_index > depth:
                raise ValueError(f"{name} indices must be within [1, {depth}], got {block_index}.")
            resolved.append(block_index - 1)
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"{name} must not contain duplicate layer indices: {tuple(indices)}")
        return resolved

    @staticmethod
    def _resolve_zero_based_block_indices(indices: Sequence[int], depth: int) -> list[int]:
        resolved: list[int] = []
        for idx in indices:
            block_index = int(idx)
            if block_index < 0 or block_index >= depth:
                raise ValueError(f"sva_layers indices must be within [0, {depth - 1}], got {block_index}.")
            resolved.append(block_index)
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"sva_layers must not contain duplicate layer indices: {tuple(indices)}")
        return resolved

    @staticmethod
    def _strip_known_prefixes(key: str) -> str:
        prefixes = ("module.", "model.", "teacher.", "student.", "backbone.")
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
        return stripped

    def load_pretrained(self, pretrained: str | None) -> None:
        if not pretrained:
            return

        checkpoint_path = Path(pretrained)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "teacher", "student", "state_dict"):
                candidate = state.get(key)
                if isinstance(candidate, dict):
                    state = candidate
                    break

        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")

        encoder_state = self.encoder.state_dict()
        loadable_state: dict[str, Tensor] = {}
        mismatched_keys: list[str] = []
        for raw_key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = self._strip_known_prefixes(raw_key)
            if key not in encoder_state:
                continue
            if encoder_state[key].shape != value.shape:
                mismatched_keys.append(
                    f"{key}: checkpoint {tuple(value.shape)} != model {tuple(encoder_state[key].shape)}"
                )
                continue
            loadable_state[key] = value

        if not loadable_state:
            raise RuntimeError(
                f"No DINOv3 backbone weights matched the current model when loading {checkpoint_path}."
            )
        missing_keys = [key for key in encoder_state if key not in loadable_state]
        if missing_keys or mismatched_keys:
            problems: list[str] = []
            if missing_keys:
                missing_preview = ", ".join(missing_keys[:10])
                missing_suffix = " ..." if len(missing_keys) > 10 else ""
                problems.append(f"missing_keys={missing_preview}{missing_suffix}")
            if mismatched_keys:
                mismatch_preview = "; ".join(mismatched_keys[:5])
                mismatch_suffix = " ..." if len(mismatched_keys) > 5 else ""
                problems.append(f"mismatched_shapes={mismatch_preview}{mismatch_suffix}")
            raise RuntimeError(
                "Refusing to continue with a partial DINOv3 backbone load from "
                f"{checkpoint_path}: {' | '.join(problems)}"
            )

        self.encoder.load_state_dict(loadable_state, strict=True)

    def _run_block(self, block: nn.Module, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None) -> Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(lambda tokens: block(tokens, rope_sincos), x, use_reentrant=False)
        return block(x, rope_sincos)

    def _capture_patch_feature_map(self, tokens: Tensor, patch_h: int, patch_w: int) -> Tensor:
        patch_tokens = self.encoder._norm_patch_tokens(tokens)
        batch_size = patch_tokens.shape[0]
        return patch_tokens.transpose(1, 2).reshape(batch_size, self.embed_dim, patch_h, patch_w)

    def _apply_sva(self, tokens: Tensor, block_index: int, batch_size: int, num_views: int) -> Tensor:
        if num_views <= 1 or block_index not in self.sva_block_indices:
            return tokens

        total_tokens = tokens.shape[1]
        tokens_by_view = tokens.reshape(batch_size, num_views, total_tokens, self.embed_dim)
        ref_prefix = tokens_by_view[:, 0, : self.token_offset, :]
        ref_tokens = tokens_by_view[:, 0, self.token_offset :, :]
        src_tokens = tokens_by_view[:, 1:, self.token_offset :, :].reshape(batch_size, -1, self.embed_dim)
        if src_tokens.shape[1] == 0:
            return tokens

        updated_ref_tokens = self.sva[str(block_index)](ref_tokens, src_tokens)
        updated_ref_view = torch.cat([ref_prefix, updated_ref_tokens], dim=1)
        if num_views == 2:
            updated_tokens_by_view = torch.stack([updated_ref_view, tokens_by_view[:, 1]], dim=1)
        else:
            updated_tokens_by_view = torch.cat([updated_ref_view.unsqueeze(1), tokens_by_view[:, 1:]], dim=1)
        return updated_tokens_by_view.reshape(batch_size * num_views, total_tokens, self.embed_dim)

    def _project_stage(self, stage_key: str, fmap: Tensor, image_hw: tuple[int, int]) -> Tensor:
        image_h, image_w = image_hw
        target = (
            max(1, image_h // self.STAGE_SCALES[stage_key]),
            max(1, image_w // self.STAGE_SCALES[stage_key]),
        )
        projected = self.out_proj[stage_key](fmap)
        return F.interpolate(projected, size=target, mode="bilinear", align_corners=False)

    def _encode_flat_images(
        self,
        flat_images: Tensor,
        *,
        batch_size: int,
        num_views: int,
        apply_sva: bool,
    ) -> dict[str, Tensor]:
        tokens, (patch_h, patch_w) = self.encoder.prepare_tokens_with_masks(flat_images)
        rope_sincos = self.encoder.rope_embed(H=patch_h, W=patch_w) if self.encoder.rope_embed is not None else None

        stage_maps_by_block: dict[int, Tensor] = {}
        for block_index, block in enumerate(self.encoder.blocks):
            tokens = self._run_block(block, tokens, rope_sincos)
            if apply_sva:
                tokens = self._apply_sva(tokens, block_index=block_index, batch_size=batch_size, num_views=num_views)
            if block_index in self.output_block_indices:
                stage_maps_by_block[block_index] = self._capture_patch_feature_map(tokens, patch_h, patch_w)

        return {
            stage_key: stage_maps_by_block[block_index]
            for stage_key, block_index in zip(self.STAGE_NAMES, self.output_block_indices)
        }

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, 3, H, W], got {tuple(images.shape)}")

        stage_maps = self._encode_flat_images(
            images,
            batch_size=images.shape[0],
            num_views=1,
            apply_sva=False,
        )
        image_hw = (images.shape[-2], images.shape[-1])
        return {stage_key: self._project_stage(stage_key, fmap, image_hw) for stage_key, fmap in stage_maps.items()}

    def extract_feature(self, images: Tensor) -> dict[str, Tensor]:
        return self.forward(images)

    def forward_multiview(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected [B, V, 3, H, W], got {tuple(images.shape)}")

        batch_size, num_views, _, image_h, image_w = images.shape
        flat_images = images.reshape(batch_size * num_views, 3, image_h, image_w)
        stage_maps = self._encode_flat_images(
            flat_images,
            batch_size=batch_size,
            num_views=num_views,
            apply_sva=num_views > 1 and len(self.sva_block_indices) > 0,
        )

        output: dict[str, Tensor] = {}
        for stage_key, fmap in stage_maps.items():
            projected = self._project_stage(stage_key, fmap, (image_h, image_w))
            channels = projected.shape[1]
            output[stage_key] = projected.reshape(
                batch_size,
                num_views,
                channels,
                projected.shape[-2],
                projected.shape[-1],
            )
        return output
