"""WAM Dreamer world model (Dreamer redesign, P2): encoder + RSSM + reward/cont/BEV heads.

Composes the frozen Stage-1 graph embedding (pre-pooled in the replay) + scalar queue/ego features into
an observation embedding, runs the :class:`~car_dreamer.toolkit.wam.wam_rssm.RSSM`, and predicts:

  * **reward** head  -- symlog regression to the P2 reward (``-`` cost-rate);
  * **cont** head    -- episode continuation (1 - terminal);
  * **BEV decoder**  -- reconstructs the observed BEV occupancy raster (``WAMBevDecoder``), the
    reconstruction signal that forces the latent to encode the scene state.

Alignment (DreamerV3 convention): at step ``t`` the RSSM consumes ``prev_action = a_{t-1}`` and ``obs_t``;
the reward head predicts the reward of the ``(t-1)->t`` transition (our ``reward[t-1]``). Both the action
and the reward target are shifted by one and ``t=0`` is masked out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev import BEV_NUM_CHANNELS, WAMBevDecoder, bev_iou, bev_reconstruction_loss
from .wam_rssm import RSSM, RSSMConfig

__all__ = ["WorldModelConfig", "WAMWorldModel", "symlog", "symexp"]


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(max(layers - 1, 0)):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


@dataclass
class WorldModelConfig:
    graph_embed_dim: int = 512          # obs_graph_embed dim (2 * Stage-1 hidden)
    scalar_dim: int = 5                 # [z, total_backlog, ego_v, num_candidates, num_notable]
    action_dims: Tuple[int, ...] = (9, 4, 2)   # factorized-discrete (S,B,D) sizes
    embed_dim: int = 256                # encoder output (RSSM obs embedding)
    scalar_hidden: int = 64
    head_hidden: int = 256
    head_layers: int = 3
    bev_channels: int = BEV_NUM_CHANNELS
    bev_size: int = 16
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    # loss weights
    w_reward: float = 1.0
    w_cont: float = 1.0
    w_bev: float = 1.0
    bev_pos_weight: float = 50.0        # up-weight the sparse occupied cells in the BEV BCE

    @property
    def action_dim(self) -> int:
        return int(sum(self.action_dims))


class WAMWorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        # keep rssm dims consistent with this world model
        cfg.rssm.embed_dim = cfg.embed_dim
        cfg.rssm.action_dim = cfg.action_dim
        self.cfg = cfg
        self.scalar_mlp = _mlp(cfg.scalar_dim, cfg.scalar_hidden, cfg.scalar_hidden, layers=2)
        self.enc_proj = nn.Sequential(
            nn.Linear(cfg.graph_embed_dim + cfg.scalar_hidden, cfg.embed_dim), nn.SiLU()
        )
        self.rssm = RSSM(cfg.rssm)
        feat = cfg.rssm.feat_dim
        self.reward_head = _mlp(feat, cfg.head_hidden, 1, cfg.head_layers)
        self.cont_head = _mlp(feat, cfg.head_hidden, 1, cfg.head_layers)
        self.bev_decoder = WAMBevDecoder(feat, channels=cfg.bev_channels, size=cfg.bev_size)

    @property
    def device(self) -> torch.device:
        return self.rssm.device

    # ---- encoding ----

    def encode(self, graph_embed: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """(``[...,Dg]``, ``[...,S]``) -> obs embedding ``[...,embed_dim]``."""
        s = self.scalar_mlp(symlog(scalars))
        return self.enc_proj(torch.cat([graph_embed, s], dim=-1))

    def action_onehot(self, actions: torch.Tensor) -> torch.Tensor:
        """Factorized-discrete indices ``[...,F]`` -> concatenated one-hot ``[...,Σdims]``."""
        parts = [F.one_hot(actions[..., i].long(), num_classes=n).float()
                 for i, n in enumerate(self.cfg.action_dims)]
        return torch.cat(parts, dim=-1)

    def get_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        stoch = state["stoch"].reshape(*state["stoch"].shape[:-2], -1)
        return torch.cat([state["deter"], stoch], dim=-1)

    # ---- training loss ----

    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        """One world-model training step. ``batch`` tensors are ``[B, T, ...]``.

        Keys: ``obs_graph_embed`` [B,T,Dg], ``obs_scalars`` [B,T,S], ``actions`` [B,T,F],
        ``rewards`` [B,T], ``is_first`` [B,T], ``is_terminal`` [B,T], ``bev_target`` [B,T,C,H,W].
        Returns ``(total_loss, scalar_metrics, post_state)``.
        """
        embed = self.encode(batch["obs_graph_embed"], batch["obs_scalars"])  # [B,T,E]
        onehot = self.action_onehot(batch["actions"])                        # [B,T,A]
        b, t = embed.shape[0], embed.shape[1]

        # DreamerV3 shift: prev_action at t is a_{t-1}
        prev_actions = torch.cat([torch.zeros_like(onehot[:, :1]), onehot[:, :-1]], dim=1)
        post, prior = self.rssm.observe(embed, prev_actions, batch["is_first"].float())
        feat = self.get_feat(post)  # [B,T,feat]

        # --- reward: predict reward-on-entering-t = reward[t-1] (shift + mask t=0) ---
        reward_pred = self.reward_head(feat).squeeze(-1)  # [B,T]
        reward_tgt = torch.cat([torch.zeros_like(batch["rewards"][:, :1]), batch["rewards"][:, :-1]], dim=1)
        step_mask = torch.ones(b, t, device=embed.device)
        step_mask[:, 0] = 0.0
        reward_loss = (((reward_pred - symlog(reward_tgt)) ** 2) * step_mask).sum() / step_mask.sum().clamp_min(1.0)

        # --- continuation: 1 - terminal ---
        cont_logit = self.cont_head(feat).squeeze(-1)  # [B,T]
        cont_tgt = 1.0 - batch["is_terminal"].float()
        cont_loss = F.binary_cross_entropy_with_logits(cont_logit, cont_tgt)

        # --- BEV occupancy reconstruction (pos-weighted BCE for the sparse target) ---
        bev_logits = self.bev_decoder(feat.reshape(b * t, -1))  # [B*T,C,H,W]
        bev_tgt = batch["bev_target"].reshape(b * t, *batch["bev_target"].shape[2:]).float()
        pos_w = torch.tensor(float(self.cfg.bev_pos_weight), device=embed.device)
        bev_loss = F.binary_cross_entropy_with_logits(bev_logits, bev_tgt, pos_weight=pos_w)

        # --- dynamics / representation KL ---
        kl = self.rssm.kl_loss(post, prior)

        total = (kl["kl"] + self.cfg.w_reward * reward_loss
                 + self.cfg.w_cont * cont_loss + self.cfg.w_bev * bev_loss)
        with torch.no_grad():
            iou = bev_iou(torch.sigmoid(bev_logits), bev_tgt)
            metrics = {
                "total": float(total), "kl": float(kl["kl"]), "dyn": float(kl["dyn"]), "rep": float(kl["rep"]),
                "reward": float(reward_loss), "cont": float(cont_loss), "bev": float(bev_loss), "bev_iou": float(iou),
            }
        return total, metrics, post

    # ---- imagination (used by P3 actor-critic) ----

    def imagine(self, start: Dict[str, torch.Tensor], actor, horizon: int) -> Dict[str, torch.Tensor]:
        """Roll the prior forward ``horizon`` steps, sampling actions from ``actor(feat)`` (a callable
        returning a one-hot action tensor). Returns stacked ``{feat, reward, cont, action}`` over the horizon."""
        state = start
        feats, rewards, conts, actions = [], [], [], []
        for _ in range(int(horizon)):
            feat = self.get_feat(state)
            action = actor(feat)  # [B, A] one-hot
            state = self.rssm.img_step(state, action)
            f2 = self.get_feat(state)
            feats.append(f2)
            rewards.append(symexp(self.reward_head(f2).squeeze(-1)))
            conts.append(torch.sigmoid(self.cont_head(f2).squeeze(-1)))
            actions.append(action)
        return {
            "feat": torch.stack(feats, dim=1),
            "reward": torch.stack(rewards, dim=1),
            "cont": torch.stack(conts, dim=1),
            "action": torch.stack(actions, dim=1),
        }
