"""WAM Dreamer actor-critic (Dreamer redesign, P3): learn the cooperation policy in imagination.

Given a trained :class:`~car_dreamer.toolkit.wam.wam_world_model.WAMWorldModel`, the actor + critic are
trained purely on **imagined** latent rollouts (no env / CARLA): from replay-derived start states, the
actor proposes factorized-discrete ``(S,B,D)`` actions, the world model imagines the next latent + reward
+ continue, and the actor is updated by REINFORCE on λ-returns while the critic regresses the λ-returns
(with a slow target network). The world model is frozen here (P4 does joint online fine-tuning).

At inference the actor's ``mode`` gives a deterministic per-step ``(S,B,D)``; the ``(S,B,D,n)`` trunk is
recovered by grouping consecutive same-action steps (framing A).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategoricalStraightThrough

from .wam_world_model import symexp

__all__ = ["ActorCriticConfig", "FactorizedActor", "Critic", "WAMActorCritic", "lambda_return"]


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(max(layers - 1, 0)):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class _FactorizedDist:
    """A product of independent one-hot categoricals over factors ``(S, B, D)`` (unimix-smoothed)."""

    def __init__(self, logits: Sequence[torch.Tensor], unimix: float = 0.01):
        self.dists = []
        for lg in logits:
            if unimix > 0.0:
                p = torch.softmax(lg, dim=-1)
                p = (1.0 - unimix) * p + unimix / lg.shape[-1]
                lg = torch.log(p)
            self.dists.append(OneHotCategoricalStraightThrough(logits=lg))

    def sample(self) -> torch.Tensor:
        return torch.cat([d.sample() for d in self.dists], dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.cat([F.one_hot(d.logits.argmax(-1), d.logits.shape[-1]).float() for d in self.dists], dim=-1)

    def _split(self, action: torch.Tensor) -> List[torch.Tensor]:
        out, off = [], 0
        for d in self.dists:
            n = d.logits.shape[-1]
            out.append(action[..., off : off + n])
            off += n
        return out

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return sum(d.log_prob(a) for d, a in zip(self.dists, self._split(action)))

    def entropy(self) -> torch.Tensor:
        return sum(d.entropy() for d in self.dists)


@dataclass
class ActorCriticConfig:
    feat_dim: int = 1536
    action_dims: Tuple[int, ...] = (9, 4, 2)
    hidden: int = 512
    layers: int = 3
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4
    imag_horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_coef: float = 3e-4
    unimix: float = 0.01
    slow_tau: float = 0.02
    grad_clip: float = 100.0

    @property
    def action_dim(self) -> int:
        return int(sum(self.action_dims))


class FactorizedActor(nn.Module):
    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(cfg.feat_dim, cfg.hidden, cfg.action_dim, cfg.layers)

    def forward(self, feat: torch.Tensor) -> _FactorizedDist:
        logits = self.net(feat)
        parts, off = [], 0
        for n in self.cfg.action_dims:
            parts.append(logits[..., off : off + n])
            off += n
        return _FactorizedDist(parts, unimix=self.cfg.unimix)


class Critic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.net = _mlp(cfg.feat_dim, cfg.hidden, 1, cfg.layers)
        self.slow = _mlp(cfg.feat_dim, cfg.hidden, 1, cfg.layers)
        self.slow.load_state_dict(self.net.state_dict())
        for p in self.slow.parameters():
            p.requires_grad_(False)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)

    def slow_value(self, feat: torch.Tensor) -> torch.Tensor:
        return self.slow(feat).squeeze(-1)

    @torch.no_grad()
    def update_slow(self, tau: float) -> None:
        for s, n in zip(self.slow.parameters(), self.net.parameters()):
            s.mul_(1.0 - tau).add_(tau * n)


def lambda_return(
    reward: torch.Tensor, value: torch.Tensor, disc: torch.Tensor, lam: float
) -> torch.Tensor:
    """GAE-style λ-return. ``reward``/``value``/``disc`` are ``[N, H]`` (``disc = gamma * cont``).

    ``G_t = r_t + disc_t·[(1-λ)·V_{t+1} + λ·G_{t+1}]``, bootstrapped with ``V`` at the horizon.
    """
    h = reward.shape[1]
    out = torch.zeros_like(reward)
    g_next = value[:, -1]
    for t in reversed(range(h)):
        boot = value[:, t + 1] if (t + 1) < h else value[:, -1]
        g = reward[:, t] + disc[:, t] * ((1.0 - lam) * boot + lam * g_next)
        out[:, t] = g
        g_next = g
    return out


class WAMActorCritic:
    """Trains a :class:`FactorizedActor` + :class:`Critic` on imagined rollouts of a frozen world model."""

    def __init__(self, world_model, cfg: ActorCriticConfig, device=None):
        self.wm = world_model
        self.cfg = cfg
        self.device = device or world_model.device
        self.actor = FactorizedActor(cfg).to(self.device)
        self.critic = Critic(cfg).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.ret_scale = 1.0  # EMA of the return 5-95 percentile range (DreamerV3 retnorm)

    @torch.no_grad()
    def rollout(self, start: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Imagine ``imag_horizon`` steps from ``start`` under the current actor (no grad).

        Returns stacked ``feat [N,H,F]``, ``action [N,H,A]``, ``reward [N,H]``, ``cont [N,H]``.
        """
        state = {k: v.detach() for k, v in start.items()}
        feats, actions, rewards, conts = [], [], [], []
        for _ in range(self.cfg.imag_horizon):
            feat = self.wm.get_feat(state)          # state where the action is chosen
            action = self.actor(feat).sample()
            nxt = self.wm.rssm.img_step(state, action)
            f2 = self.wm.get_feat(nxt)              # resulting state
            feats.append(feat)                       # store the PRE-transition feat (aligns with action)
            actions.append(action)
            rewards.append(symexp(self.wm.reward_head(f2).squeeze(-1)))   # reward of the transition
            conts.append(torch.sigmoid(self.wm.cont_head(f2).squeeze(-1)))
            state = nxt
        return {
            "feat": torch.stack(feats, dim=1),
            "action": torch.stack(actions, dim=1),
            "reward": torch.stack(rewards, dim=1),
            "cont": torch.stack(conts, dim=1),
        }

    def train_step(self, start: Dict[str, torch.Tensor]) -> Dict[str, float]:
        traj = self.rollout(start)
        feat, action, reward, cont = traj["feat"], traj["action"], traj["reward"], traj["cont"]
        disc = self.cfg.gamma * cont

        with torch.no_grad():
            target_v = self.critic.slow_value(feat)
            returns = lambda_return(reward, target_v, disc, self.cfg.lam)
            # DreamerV3 return normalization: scale advantages by the running 5-95 percentile range so
            # REINFORCE is well-conditioned regardless of the (large, negative) P2 reward magnitude.
            lo, hi = torch.quantile(returns, 0.05), torch.quantile(returns, 0.95)
            self.ret_scale = 0.98 * self.ret_scale + 0.02 * float((hi - lo).clamp_min(1.0))
            scale = max(self.ret_scale, 1.0)

        # --- actor: REINFORCE on normalized advantage (returns - baseline), + entropy bonus ---
        dist = self.actor(feat.detach())
        logp = dist.log_prob(action.detach())
        ent = dist.entropy()
        adv = ((returns - target_v) / scale).detach()
        actor_loss = -(logp * adv).mean() - self.cfg.entropy_coef * ent.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        # --- critic: regress the λ-returns ---
        value = self.critic.value(feat.detach())
        critic_loss = 0.5 * ((value - returns.detach()) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()
        self.critic.update_slow(self.cfg.slow_tau)

        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "return_mean": float(returns.mean().detach()),
            "reward_mean": float(reward.mean().detach()),
            "entropy": float(ent.mean().detach()),
            "value_mean": float(value.mean().detach()),
            "ret_scale": float(scale),
        }

    def state_dict(self) -> Dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
