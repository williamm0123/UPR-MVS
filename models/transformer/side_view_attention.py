from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SideViewAttention(nn.Module):
    """Linear cross-view attention with query from reference and key/value from source views."""

    def __init__(self, dim: int, dropout: float = 0.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.norm_ref = nn.LayerNorm(dim)
        self.norm_src = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _phi(x: Tensor) -> Tensor:
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, ref_tokens: Tensor, src_tokens: Tensor) -> Tensor:
        """
        Args:
            ref_tokens: [B, N, C]
            src_tokens: [B, M, C]
        """
        q = self._phi(self.to_q(self.norm_ref(ref_tokens)))
        k = self._phi(self.to_k(self.norm_src(src_tokens)))
        v = self.to_v(src_tokens)

        kv = torch.matmul(k.transpose(1, 2), v)
        z = 1.0 / (torch.matmul(q, k.sum(dim=1, keepdim=True).transpose(1, 2)).clamp_min(self.eps))
        context = torch.matmul(q, kv)
        out = context * z
        out = self.dropout(self.out_proj(self.act(out)))
        return ref_tokens + out / math.sqrt(ref_tokens.shape[-1])
