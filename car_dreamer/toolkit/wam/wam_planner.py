"""Online Dreamer policy for cooperative-perception action selection (Dreamer redesign, P4).

Wraps a trained world model + actor into a stateful online controller: at each decision epoch it steps
the RSSM posterior with the current observation, reads the actor's deterministic ``(S,B,D)`` action, and
decodes it to a :class:`SubAction`. The ``(S,B,D,n)`` trunk is recovered upstream by grouping consecutive
same-action epochs (framing A). A ``local-only`` safety fallback (compared under the true P2 cost) is left
to the caller to preserve the no-degradation guarantee.

Self-contained + offline-testable: :meth:`WAMDreamerPolicy.act` takes a pre-pooled graph embedding +
scalars, so it runs without CARLA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from .action_chunk import SubAction
from .wam_actor_critic import ActorCriticConfig, FactorizedActor
from .wam_mdp import ActionSpec, decode_subaction
from .wam_world_model import WAMWorldModel, WorldModelConfig

__all__ = ["WAMDreamerPolicy", "load_dreamer_policy"]


class WAMDreamerPolicy:
    """Stateful online actor: RSSM posterior carried across epochs, deterministic ``(S,B,D)`` per epoch."""

    def __init__(
        self,
        world_model: WAMWorldModel,
        actor: FactorizedActor,
        action_spec: ActionSpec,
        device: Optional[torch.device] = None,
    ):
        self.wm = world_model
        self.actor = actor
        self.spec = action_spec
        self.device = device or world_model.device
        self.wm.eval()
        self.actor.eval()
        self._state: Optional[Dict[str, torch.Tensor]] = None
        self._prev_action: Optional[torch.Tensor] = None  # [1, A] one-hot

    def reset(self) -> None:
        self._state = None
        self._prev_action = None

    def _onehot_to_indices(self, onehot: torch.Tensor) -> Tuple[int, ...]:
        idx, off = [], 0
        for n in self.spec.dims:
            idx.append(int(onehot[0, off : off + n].argmax()))
            off += n
        return tuple(idx)

    @torch.no_grad()
    def act(
        self,
        graph_embed: torch.Tensor,
        scalars: torch.Tensor,
        candidate_ids: Sequence[int],
        *,
        is_first: bool = False,
    ) -> SubAction:
        """Step the RSSM with the current obs and return the decoded :class:`SubAction`.

        ``graph_embed`` [Dg], ``scalars`` [S] are this epoch's observation (frozen graph pool + queue/ego
        scalars, matching the replay). ``candidate_ids`` maps member slots to collaborator ids.
        """
        ge = graph_embed.to(self.device).reshape(1, -1)
        sc = scalars.to(self.device).reshape(1, -1)
        embed = self.wm.encode(ge, sc)  # [1, E]
        first = torch.tensor([1.0 if (is_first or self._state is None) else 0.0], device=self.device)
        if self._state is None:
            self._state = self.wm.rssm.initial(1)
            self._prev_action = torch.zeros(1, self.spec_action_dim, device=self.device)
        post, _ = self.wm.rssm.obs_step(self._state, self._prev_action, embed, first)
        self._state = post
        feat = self.wm.get_feat(post)
        onehot = self.actor(feat).mode()  # [1, A]
        self._prev_action = onehot
        return decode_subaction(self._onehot_to_indices(onehot), candidate_ids, self.spec)

    @property
    def spec_action_dim(self) -> int:
        return int(sum(self.spec.dims))


def load_dreamer_policy(
    world_model_ckpt: Union[str, Path],
    actor_critic_ckpt: Union[str, Path],
    *,
    action_spec: Optional[ActionSpec] = None,
    device: Union[str, torch.device] = "cpu",
) -> WAMDreamerPolicy:
    """Rebuild a :class:`WAMDreamerPolicy` from a P2 world-model checkpoint + a P3 actor-critic checkpoint."""
    device = torch.device(device)
    wm_ck = torch.load(world_model_ckpt, map_location=device, weights_only=False)
    wm = WAMWorldModel(wm_ck["cfg"])
    wm.load_state_dict(wm_ck["model"])
    wm.to(device).eval()

    ac_ck = torch.load(actor_critic_ckpt, map_location=device, weights_only=False)
    ac_cfg: ActorCriticConfig = ac_ck["cfg"]
    actor = FactorizedActor(ac_cfg)
    actor.load_state_dict(ac_ck["actor_critic"]["actor"])
    actor.to(device).eval()

    if action_spec is None:
        action_spec = ActionSpec(
            max_members=int(ac_cfg.action_dims[0]) - 1,
            modalities=("objlist", "bev")[: int(ac_cfg.action_dims[2])],
            use_duration=False,
        )
    return WAMDreamerPolicy(wm, actor, action_spec, device=device)
