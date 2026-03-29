from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .attention_backend import attention_forward, normalize_attention_backend
from .positional_encoding import AdaptiveAttentionScaling, FrustoconicalPositionalEncoding3D
from .utils import (
    group_wise_correlation,
    homo_warping,
    intrinsics_to_projection,
    sample_depth_planes,
    scale_intrinsics,
)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, nhead: int, dropout: float = 0.0, attention_backend: str = "auto") -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"Expected dim divisible by nhead, got dim={dim}, nhead={nhead}")
        self.num_heads = nhead
        self.head_dim = dim // nhead
        self.backend = normalize_attention_backend(attention_backend)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn_out = attention_forward(
            q,
            k,
            v,
            backend=self.backend,
            dropout_p=self.attn_drop.p,
            training=self.training,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        return self.out_proj(attn_out)


class CVTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        aas_enable: bool = True,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, nhead, dropout=dropout, attention_backend=attention_backend)
        self.aas = AdaptiveAttentionScaling(1.0) if aas_enable else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        a = self.attn(h)
        x = x + self.aas(a)
        x = x + self.mlp(self.norm2(x))
        return x


class CostVolumeTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        nhead: int,
        d_bins: int,
        scales: Sequence[int],
        num_groups: int = 8,
        aas_enable: bool = True,
        fpe_enable: bool = True,
        use_checkpoint: bool = False,
        share_across_scales: bool = False,
        process_all_scales: bool = False,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.d_bins = d_bins
        self.scales = tuple(scales)
        if not self.scales:
            raise ValueError("CostVolumeTransformer.scales must contain at least one scale.")
        self.active_scales = self.scales if process_all_scales else (self.scales[0],)
        self.num_groups = num_groups
        self.feature_dim = feature_dim
        self.use_checkpoint = use_checkpoint
        self.process_all_scales = process_all_scales
        self.fpe = FrustoconicalPositionalEncoding3D(feature_dim) if fpe_enable else None
        self.token_proj = nn.Linear(self.num_groups, feature_dim)

        def build_stack() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    CVTransformerBlock(
                        feature_dim,
                        nhead=nhead,
                        aas_enable=aas_enable,
                        attention_backend=attention_backend,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.blocks = build_stack() if share_across_scales else nn.ModuleList()
        self.scale_blocks = nn.ModuleDict()
        if not share_across_scales:
            for s in self.active_scales:
                self.scale_blocks[str(s)] = build_stack()
        self.share_across_scales = share_across_scales
        self.logit_head = nn.Conv3d(feature_dim, 1, kernel_size=1)

    def _run_blocks(self, x: Tensor, blocks: nn.ModuleList) -> Tensor:
        for block in blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint(lambda t: block(t), x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def _build_volume(self, features: Tensor, projections: Tensor, depth_values: Tensor) -> Tensor:
        b, v, c, h, w = features.shape
        ref = features[:, 0]
        ref_volume = ref.unsqueeze(2).expand(-1, -1, depth_values.shape[1], -1, -1)
        corr_acc = 0.0
        count = 0
        for i in range(1, v):
            src_vol = homo_warping(features[:, i], projections[:, i], projections[:, 0], depth_values)
            corr = group_wise_correlation(ref_volume, src_vol, self.num_groups)
            corr_acc = corr_acc + corr
            count += 1
        return corr_acc / float(max(count, 1))

    def forward(
        self,
        features: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        depth_range: Tensor,
        image_hw: tuple[int, int],
        num_depth_bins: int | None = None,
    ) -> dict[str, Tensor]:
        b, v, c, h, w = features.shape
        img_h, img_w = image_hw
        d_bins = num_depth_bins or self.d_bins
        depth_values = sample_depth_planes(depth_range, d_bins)

        scaled_intrinsics = scale_intrinsics(
            intrinsics.view(b * v, 3, 3),
            scale_x=w / float(img_w),
            scale_y=h / float(img_h),
        ).view(b, v, 3, 3)
        projections = torch.stack(
            [intrinsics_to_projection(scaled_intrinsics[:, i], extrinsics[:, i]) for i in range(v)], dim=1
        )

        base_volume = self._build_volume(features, projections, depth_values)  # [B,G,D,H,W]
        volume_tokens = []
        per_scale_logits: dict[str, Tensor] = {}

        for scale in self.active_scales:
            sd = max(1, d_bins // scale)
            sh = max(1, h // max(scale // 4, 1))
            sw = max(1, w // max(scale // 4, 1))
            scaled = F.interpolate(base_volume, size=(sd, sh, sw), mode="trilinear", align_corners=False)
            token = scaled.permute(0, 2, 3, 4, 1).reshape(b, sd * sh * sw, self.num_groups)
            volume_tokens.append((scale, token, (sd, sh, sw)))

        coarse_probs = None
        coarse_depth = None
        coarse_logits = None

        for scale, tokens, shape in volume_tokens:
            sd, sh, sw = shape
            x = self.token_proj(tokens)
            if self.fpe is not None:
                pos_feat = self.fpe(sample_depth_planes(depth_range, sd), sh, sw)
                x = x + pos_feat.reshape(b, sd * sh * sw, -1)
            blocks = self.blocks if self.share_across_scales else self.scale_blocks[str(scale)]
            x = self._run_blocks(x, blocks)
            vol = x.reshape(b, sd, sh, sw, self.feature_dim).permute(0, 4, 1, 2, 3)
            logits = self.logit_head(vol).squeeze(1)
            per_scale_logits[f"scale_{scale}"] = logits
            if scale == self.active_scales[0]:
                coarse_logits = logits
                probs = F.softmax(logits, dim=1)
                dv = sample_depth_planes(depth_range, sd)
                coarse_probs = probs
                coarse_depth = (probs * dv[:, :, None, None]).sum(dim=1, keepdim=True)

        assert coarse_depth is not None and coarse_probs is not None and coarse_logits is not None
        return {
            "coarse_depth": coarse_depth,
            "coarse_prob": coarse_probs,
            "coarse_confidence": coarse_probs.max(dim=1, keepdim=True).values,
            "coarse_logits": coarse_logits,
            "depth_values": sample_depth_planes(depth_range, coarse_probs.shape[1]),
            "cvt_scale_logits": per_scale_logits,
        }
