"""Permutation-invariant Set-Transformer fusion over a variable agent-token set.

This is the reusable fusion *core* (the V2X-ViT idea, reduced to the set/vector
level rather than BEV grid warping): a masked self-attention stack over the
``{ego, neighbor_1, ..., neighbor_N}`` token set, followed by Pooling-by-Multihead-
Attention (PMA) to a single scene vector. It is deliberately I/O-agnostic so it can
be wrapped by the L3 model (semantic-embedding reconstruction) or anything else.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PMA(nn.Module):
    """Pooling by Multihead Attention (Set Transformer).

    ``num_seeds`` learned query vectors attend over the token set to produce
    ``num_seeds`` pooled vectors. We use a single seed to pool to one scene vector.
    """

    def __init__(self, d_model: int, num_heads: int, num_seeds: int = 1) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]; key_padding_mask: [B, T] (True = ignore)
        q = self.seed.expand(x.size(0), -1, -1)
        attn_out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln1(q + attn_out)
        h = self.ln2(h + self.ff(h))
        return h  # [B, num_seeds, D]


class SetTransformerFusion(nn.Module):
    """Masked self-attention stack + PMA pooling -> one fused vector ``[B, d_model]``."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pma = PMA(d_model, num_heads, num_seeds=1)

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tokens: [B, T, D]; key_padding_mask: [B, T] (True = pad/ignore). At least one
        # token per row must be valid (the ego token, which is never masked).
        h = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        pooled = self.pma(h, key_padding_mask=key_padding_mask)  # [B, 1, D]
        return pooled.squeeze(1)  # [B, D]
