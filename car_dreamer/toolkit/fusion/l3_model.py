"""Route-L3: self-supervised cooperative scene-embedding fusion.

Pipeline
--------
Inputs (all embeddings are *precomputed* by frozen encoders; the model is
backbone-agnostic so the encoder can be swapped without retraining the pipeline):

    ego_emb  : ego camera image embedding (e.g. CLIP image encoder)
    nbr_emb  : neighbor caption embeddings (e.g. a text encoder), one per neighbor
    nbr_pose : each neighbor's ego-relative pose (dx, dy, dyaw)
    nbr_mask : which neighbor slots are valid

The model projects ego + neighbor tokens to ``d_model``, adds an ego/neighbor type
embedding and a relative-pose encoding, fuses them with a masked Set-Transformer,
and projects to ``out_dim`` (the *output space* -- chosen to be a strong text
embedding space such as SigLIP-text / sentence-transformers, so action text-text
cosine is reliable). The fused vector ``z_fused`` is L2-normalized.

Training (self-supervised, L3): ``z_fused`` is pulled (InfoNCE) toward
``z_target`` -- the text embedding of a privileged, full-visibility *templated*
scene description built from CARLA ground truth (see ``templates.py``). Labels are
free. ``ego_dropout_p`` randomly hides the ego token (when at least one neighbor is
present) so the head cannot collapse to "ignore V2V".

Inference: actions are scored by cosine similarity between ``z_fused`` and the
embeddings of accelerate / decelerate / maintain language statements; argmax picks
the action. A thin calibration on top (a few oracle-labeled samples) can be added
later -- the representation itself is trained here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pose_encoding import RelativePoseEncoder
from .set_fusion import SetTransformerFusion


@dataclass
class L3FusionConfig:
    ego_emb_dim: int           # input dim of the ego image embedding
    nbr_emb_dim: int           # input dim of a neighbor caption embedding
    out_dim: int               # output/target space dim (e.g. SigLIP/sentence text dim)
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 2
    pose_num_freqs: int = 6
    pose_scale: float = 50.0   # meters; translation normalizer for pose encoding
    dropout: float = 0.0
    temperature: float = 0.07  # initial InfoNCE temperature (learned)
    ego_dropout_p: float = 0.0 # train-time prob of hiding ego (anti-collapse)


class L3FusionModel(nn.Module):
    """Cooperative scene-embedding fusion head (see module docstring)."""

    def __init__(self, cfg: L3FusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ego_proj = nn.Linear(cfg.ego_emb_dim, cfg.d_model)
        self.nbr_proj = nn.Linear(cfg.nbr_emb_dim, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)  # 0 = ego, 1 = neighbor
        self.missing_ego = nn.Parameter(torch.randn(cfg.d_model) * 0.02)  # placeholder when ego is dropped
        self.pose_enc = RelativePoseEncoder(cfg.d_model, cfg.pose_num_freqs, cfg.pose_scale)
        self.fusion = SetTransformerFusion(
            cfg.d_model, cfg.num_heads, cfg.num_layers, dropout=cfg.dropout
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.out_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.temperature)))

    def forward(
        self,
        ego_emb: torch.Tensor,
        nbr_emb: torch.Tensor,
        nbr_pose: torch.Tensor,
        nbr_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse ego + neighbor tokens into a scene embedding.

        ego_emb:  ``[B, ego_emb_dim]``
        nbr_emb:  ``[B, N, nbr_emb_dim]``
        nbr_pose: ``[B, N, 3]`` -- ``(dx, dy, dyaw_rad)`` relative to ego
        nbr_mask: ``[B, N]`` -- 1/True = valid neighbor, 0/False = padding

        returns ``z_fused``: ``[B, out_dim]``, L2-normalized.
        """
        B, N = nbr_emb.shape[0], nbr_emb.shape[1]
        device = ego_emb.device

        # --- ego token ---
        ego_tok = self.ego_proj(ego_emb)
        ego_type = self.type_emb(torch.zeros(B, dtype=torch.long, device=device))
        ego_pose = torch.zeros(B, 3, device=device, dtype=ego_tok.dtype)
        ego_tok = ego_tok + self.pose_enc(ego_pose) + ego_type  # [B, D]

        # --- neighbor tokens ---
        nbr_tok = self.nbr_proj(nbr_emb)
        nbr_type = self.type_emb(torch.ones(B, N, dtype=torch.long, device=device))
        nbr_tok = nbr_tok + self.pose_enc(nbr_pose) + nbr_type  # [B, N, D]

        # --- padding mask (True = ignore); ego is valid unless dropped this step ---
        if nbr_mask is None:
            nbr_pad = torch.zeros(B, N, dtype=torch.bool, device=device)
            has_nbr = torch.full((B,), N > 0, dtype=torch.bool, device=device)
        else:
            nbr_pad = ~nbr_mask.bool()
            has_nbr = nbr_mask.bool().any(dim=1)

        ego_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
        if self.training and self.cfg.ego_dropout_p > 0.0:
            drop = (torch.rand(B, device=device) < self.cfg.ego_dropout_p) & has_nbr
            ego_pad = drop.unsqueeze(1)
            # Replace dropped ego content with a learned placeholder so the (masked)
            # token still carries a sane value for attention numerics.
            ego_tok = torch.where(drop.unsqueeze(-1), self.missing_ego.expand(B, -1), ego_tok)

        tokens = torch.cat([ego_tok.unsqueeze(1), nbr_tok], dim=1)            # [B, 1+N, D]
        key_padding_mask = torch.cat([ego_pad, nbr_pad], dim=1)              # [B, 1+N]

        fused = self.fusion(tokens, key_padding_mask=key_padding_mask)       # [B, D]
        z = self.head(fused)
        return F.normalize(z, dim=-1)

    def info_nce_loss(self, z_fused: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE between fused scene embeddings and target embeddings.

        ``z_fused``, ``z_target``: ``[B, out_dim]``. The positive for row ``i`` is
        ``z_target[i]``; all other rows in the batch are negatives.
        """
        z_target = F.normalize(z_target, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * z_fused @ z_target.t()  # [B, B]
        labels = torch.arange(z_fused.size(0), device=z_fused.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    @torch.no_grad()
    def score_actions(self, z_fused: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine score of each fused embedding against each action statement.

        ``action_embeddings``: ``[A, out_dim]``. returns ``[B, A]`` cosine scores.
        """
        ae = F.normalize(action_embeddings, dim=-1)
        return z_fused @ ae.t()


def _smoke_test() -> None:
    """Random-tensor smoke test: forward + loss + backward + action scoring."""
    torch.manual_seed(0)
    B, N = 8, 4
    cfg = L3FusionConfig(ego_emb_dim=768, nbr_emb_dim=768, out_dim=768, d_model=256, ego_dropout_p=0.3)
    model = L3FusionModel(cfg)
    model.train()

    ego_emb = torch.randn(B, cfg.ego_emb_dim)
    nbr_emb = torch.randn(B, N, cfg.nbr_emb_dim)
    nbr_pose = torch.randn(B, N, 3) * torch.tensor([20.0, 20.0, 1.0])
    nbr_mask = (torch.rand(B, N) > 0.3)
    nbr_mask[:, 0] = True  # guarantee at least one neighbor per row

    z = model(ego_emb, nbr_emb, nbr_pose, nbr_mask)
    z_target = F.normalize(torch.randn(B, cfg.out_dim), dim=-1)
    loss = model.info_nce_loss(z, z_target)
    loss.backward()

    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)

    model.eval()
    with torch.no_grad():
        z_eval = model(ego_emb, nbr_emb, nbr_pose, nbr_mask)
        action_emb = F.normalize(torch.randn(3, cfg.out_dim), dim=-1)  # accel/decel/maintain
        scores = model.score_actions(z_eval, action_emb)
        action = scores.argmax(dim=-1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"z_fused shape        : {tuple(z.shape)}  (expected ({B}, {cfg.out_dim}))")
    print(f"||z||                : {z.norm(dim=-1).mean().item():.4f}  (expected ~1.0)")
    print(f"InfoNCE loss         : {loss.item():.4f}")
    print(f"all grads present    : {grad_ok}")
    print(f"action scores shape  : {tuple(scores.shape)}  (expected ({B}, 3))")
    print(f"argmax actions       : {action.tolist()}")
    print(f"trainable parameters : {n_params:,}")
    print("OK: L3 fusion head forward/backward/scoring all work.")


if __name__ == "__main__":
    _smoke_test()
