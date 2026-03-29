from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

AttentionBackend = Literal["auto", "sdpa", "eager"]

_warned_fallbacks: set[str] = set()


def normalize_attention_backend(backend: str | None) -> AttentionBackend:
    normalized = "auto" if backend is None else str(backend).lower()
    if normalized not in {"auto", "sdpa", "eager"}:
        raise ValueError(f"Unsupported attention backend: {backend}")
    return normalized  # type: ignore[return-value]


def _assert_sdpa_inputs(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise AssertionError(
            f"SDPA expects q/k/v with 4 dims [B, H, N, D], got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise AssertionError("SDPA path requires CUDA tensors.")
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise AssertionError(f"SDPA path expects fp16/bf16 tensors, got {q.dtype}.")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise AssertionError(f"SDPA path requires matching dtypes, got {q.dtype}, {k.dtype}, {v.dtype}.")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise AssertionError("Batch size mismatch between q, k, and v.")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise AssertionError("Head count mismatch between q, k, and v.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise AssertionError("head_dim mismatch between q, k, and v.")


def eager_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = False,
    scale: float | None = None,
    need_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    scale_factor = (q.shape[-1] ** -0.5) if scale is None else scale
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_scores = attn_scores + attn_mask.to(dtype=attn_scores.dtype)

    if is_causal:
        query_len = q.shape[-2]
        key_len = k.shape[-2]
        causal_mask = torch.ones((query_len, key_len), device=q.device, dtype=torch.bool).tril()
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

    attn_probs = torch.softmax(attn_scores, dim=-1)
    if dropout_p > 0.0:
        attn_probs = torch.dropout(attn_probs, dropout_p, train=training)
    out = torch.matmul(attn_probs, v)
    if need_weights:
        return out, attn_probs
    return out


def attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    backend: str = "auto",
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = False,
    scale: float | None = None,
    need_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    normalized_backend = normalize_attention_backend(backend)
    effective_dropout = float(dropout_p) if training else 0.0

    if need_weights and normalized_backend != "eager":
        if normalized_backend == "sdpa":
            raise ValueError("Attention weights are only available in eager mode.")
        normalized_backend = "eager"

    if normalized_backend == "eager":
        return eager_attention_forward(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=effective_dropout,
            is_causal=is_causal,
            training=training,
            scale=scale,
            need_weights=need_weights,
        )

    try:
        _assert_sdpa_inputs(q, k, v)
        sdpa_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": effective_dropout,
            "is_causal": is_causal,
        }
        if scale is not None:
            sdpa_kwargs["scale"] = scale
        out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        if need_weights:
            raise ValueError("Attention weights are not returned by the SDPA path.")
        return out
    except (AssertionError, RuntimeError) as exc:
        if normalized_backend == "sdpa":
            raise
        message = str(exc)
        if message not in _warned_fallbacks:
            warnings.warn(f"Falling back to eager attention because SDPA was unavailable: {message}", stacklevel=2)
            _warned_fallbacks.add(message)
        return eager_attention_forward(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=effective_dropout,
            is_causal=is_causal,
            training=training,
            scale=scale,
            need_weights=need_weights,
        )
