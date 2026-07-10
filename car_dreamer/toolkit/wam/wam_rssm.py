"""Compact RSSM (Dreamer redesign, P2): a standalone torch port of the DreamerV3 recurrent state-space
model.

Structurally faithful to ``dreamerv3/nets.py:RSSM`` (deterministic GRU carry ``deter`` + stochastic
categorical ``stoch``, ``obs_step`` / ``img_step`` / ``observe`` / ``imagine``, KL dyn/rep losses with
free bits and a uniform mix), but **self-contained**: plain ``nn.Module`` + ``torch.distributions``, no
``nj`` / ``embodied`` framework, so it lives in the WAM toolkit and is offline unit-testable like the
rest of WAM.

state = ``{"deter": [B, D], "logit": [B, S, C], "stoch": [B, S, C]}`` (``stoch`` one-hot, straight-through).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough, kl_divergence

__all__ = ["RSSMConfig", "RSSM"]

State = Dict[str, torch.Tensor]


def _act(name: str) -> nn.Module:
    return {"silu": nn.SiLU, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}.get(name, nn.SiLU)()


@dataclass
class RSSMConfig:
    embed_dim: int = 256      # size of the observation embedding fed to obs_step
    action_dim: int = 14      # size of the (flattened one-hot) action vector
    deter: int = 512          # deterministic GRU carry dim
    stoch: int = 32           # number of categorical latent variables
    classes: int = 32         # categories per latent
    hidden: int = 512         # MLP width
    act: str = "silu"
    unimix: float = 0.01      # uniform mix into the categorical (stability/exploration)
    action_clip: float = 1.0
    free_bits: float = 1.0    # nats floor per KL term
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1

    @property
    def feat_dim(self) -> int:
        """``get_feat`` output size = deter + flattened stoch (input to heads / actor / critic)."""
        return int(self.deter + self.stoch * self.classes)


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        sc = cfg.stoch * cfg.classes
        self.img_in = nn.Sequential(nn.Linear(sc + cfg.action_dim, cfg.hidden), _act(cfg.act))
        self.gru = nn.GRUCell(cfg.hidden, cfg.deter)
        self.img_out = nn.Sequential(nn.Linear(cfg.deter, cfg.hidden), _act(cfg.act))
        self.img_stats = nn.Linear(cfg.hidden, sc)
        self.obs_out = nn.Sequential(nn.Linear(cfg.deter + cfg.embed_dim, cfg.hidden), _act(cfg.act))
        self.obs_stats = nn.Linear(cfg.hidden, sc)
        self.initial_deter = nn.Parameter(torch.zeros(cfg.deter))

    # ---- state helpers ----

    @property
    def device(self) -> torch.device:
        return self.initial_deter.device

    def initial(self, bs: int) -> State:
        deter = torch.tanh(self.initial_deter).unsqueeze(0).expand(bs, -1)
        zeros = torch.zeros(bs, self.cfg.stoch, self.cfg.classes, device=self.device)
        return {"deter": deter, "logit": zeros, "stoch": zeros}

    def get_feat(self, state: State) -> torch.Tensor:
        stoch = state["stoch"].reshape(state["stoch"].shape[0], -1)
        return torch.cat([state["deter"], stoch], dim=-1)

    def _dist(self, logit: torch.Tensor) -> Independent:
        logit = logit.reshape(*logit.shape[:-1], self.cfg.stoch, self.cfg.classes)
        if self.cfg.unimix > 0.0:
            probs = torch.softmax(logit, dim=-1)
            probs = (1.0 - self.cfg.unimix) * probs + self.cfg.unimix / self.cfg.classes
            logit = torch.log(probs)
        return Independent(OneHotCategoricalStraightThrough(logits=logit), 1)

    def get_dist(self, state: State) -> Independent:
        return self._dist(state["logit"].reshape(state["logit"].shape[0], -1))

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.cfg.action_clip <= 0.0:
            return action
        scale = self.cfg.action_clip / torch.clamp(action.abs(), min=self.cfg.action_clip)
        return action * scale.detach()

    # ---- one-step transitions ----

    def img_step(self, prev_state: State, prev_action: torch.Tensor) -> State:
        prev_stoch = prev_state["stoch"].reshape(prev_state["stoch"].shape[0], -1)
        x = torch.cat([prev_stoch, self._clip_action(prev_action)], dim=-1)
        x = self.img_in(x)
        deter = self.gru(x, prev_state["deter"])
        stats = self.img_stats(self.img_out(deter))
        dist = self._dist(stats)
        stoch = dist.rsample()
        return {"deter": deter, "logit": stats.reshape(stoch.shape), "stoch": stoch}

    def obs_step(
        self, prev_state: State, prev_action: torch.Tensor, embed: torch.Tensor, is_first: torch.Tensor
    ) -> Tuple[State, State]:
        # reset prev state/action on episode boundaries
        keep = (1.0 - is_first.to(embed.dtype)).unsqueeze(-1)  # [B,1]
        init = self.initial(is_first.shape[0])
        prev_action = self._clip_action(prev_action) * keep
        prev_state = {
            k: prev_state[k] * (keep if v.dim() == 2 else keep.unsqueeze(-1))
            + init[k] * (1.0 - (keep if v.dim() == 2 else keep.unsqueeze(-1)))
            for k, v in prev_state.items()
        }
        prior = self.img_step(prev_state, prev_action)
        x = self.obs_out(torch.cat([prior["deter"], embed], dim=-1))
        stats = self.obs_stats(x)
        dist = self._dist(stats)
        stoch = dist.rsample()
        post = {"deter": prior["deter"], "logit": stats.reshape(stoch.shape), "stoch": stoch}
        return post, prior

    # ---- sequence rollouts ----

    def observe(
        self, embed: torch.Tensor, action: torch.Tensor, is_first: torch.Tensor, state: Optional[State] = None
    ) -> Tuple[State, State]:
        """Filter a batch of sequences. ``embed [B,T,E]``, ``action [B,T,A]``, ``is_first [B,T]``.

        Returns ``(post, prior)`` each ``{k: [B,T,...]}``.
        """
        b, t = embed.shape[0], embed.shape[1]
        if state is None:
            state = self.initial(b)
        posts, priors = [], []
        for i in range(t):
            state, prior = self.obs_step(state, action[:, i], embed[:, i], is_first[:, i])
            posts.append(state)
            priors.append(prior)
        return self._stack(posts), self._stack(priors)

    def imagine(self, action: torch.Tensor, state: State) -> State:
        """Roll the prior forward under ``action [B,T,A]`` from ``state``. Returns ``{k: [B,T,...]}``."""
        t = action.shape[1]
        priors = []
        for i in range(t):
            state = self.img_step(state, action[:, i])
            priors.append(state)
        return self._stack(priors)

    @staticmethod
    def _stack(seq) -> State:
        return {k: torch.stack([s[k] for s in seq], dim=1) for k in seq[0]}

    # ---- losses (per element; caller reduces) ----

    def kl_loss(self, post: State, prior: State) -> Dict[str, torch.Tensor]:
        """DreamerV3 dyn/rep KL with free bits. ``post``/``prior`` are stacked ``{k:[B,T,...]}``.

        ``dyn`` trains the prior toward ``sg(post)``; ``rep`` trains ``post`` toward ``sg(prior)``.
        """
        def dist(state, detach):
            logit = state["logit"]
            logit = logit.detach() if detach else logit
            return self._dist(logit.reshape(*logit.shape[:-2], -1))

        free = self.cfg.free_bits
        dyn = kl_divergence(dist(post, True), dist(prior, False)).clamp_min(free)
        rep = kl_divergence(dist(post, False), dist(prior, True)).clamp_min(free)
        loss = self.cfg.kl_dyn_scale * dyn + self.cfg.kl_rep_scale * rep
        return {"kl": loss.mean(), "dyn": dyn.mean(), "rep": rep.mean()}
