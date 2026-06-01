"""Relative-pose positional encoding for cooperative-perception token fusion.

Each agent token (ego or a neighbor) carries an *ego-relative* pose
``(dx, dy, dyaw)``. The ego's own token uses ``(0, 0, 0)``, which maps to a
well-defined "reference" encoding the fusion head learns to anchor on. Because the
fusion head is trained, the encoding does not need to live in any pre-aligned
embedding space (this is precisely why training the head dissolves the
"PE breaks CLIP alignment / modality gap" problems): the head learns to consume it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RelativePoseEncoder(nn.Module):
    """Encode an ego-relative pose ``(dx, dy, dyaw)`` into a ``d_model`` vector.

    Translation is normalized by ``pos_scale`` and expanded with NeRF-style Fourier
    features; heading enters as ``(cos dyaw, sin dyaw)``. An MLP maps the result to
    ``d_model`` so it can be added directly to a projected agent token.
    """

    def __init__(self, d_model: int, num_freqs: int = 6, pos_scale: float = 50.0) -> None:
        super().__init__()
        self.num_freqs = int(num_freqs)
        self.pos_scale = float(pos_scale)
        freqs = (2.0 ** torch.arange(self.num_freqs).float()) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        # (dx, dy) -> 2 * (sin, cos) * num_freqs ; (dyaw) -> (cos, sin)
        in_dim = 2 * (2 * self.num_freqs) + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """``pose``: ``[..., 3]`` = ``(dx, dy, dyaw_rad)`` -> ``[..., d_model]``."""
        dx = pose[..., 0] / self.pos_scale
        dy = pose[..., 1] / self.pos_scale
        dyaw = pose[..., 2]
        feats = []
        for val in (dx, dy):
            ang = val.unsqueeze(-1) * self.freqs  # [..., num_freqs]
            feats.append(torch.sin(ang))
            feats.append(torch.cos(ang))
        feats.append(torch.cos(dyaw).unsqueeze(-1))
        feats.append(torch.sin(dyaw).unsqueeze(-1))
        x = torch.cat(feats, dim=-1)
        return self.mlp(x)
