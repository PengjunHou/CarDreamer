"""WAM cooperative-perception semi-MDP (Dreamer redesign, P1): reward + action codec + eq-12 U.

The decision process the Dreamer world model learns:

* **state**  = observed object-graph context (:class:`RolloutContext`) + Lyapunov queue (Z, {Q_m});
* **action** = one sub-action ``a=(S,B,D,n)`` (:class:`SubAction`) -- a temporally-extended action of
  ``n`` slots (semi-MDP / options);
* **reward** = ``-`` P2 cost-rate over the sub-action (:func:`action_cost_rate`, paper eq 42), computed
  from the :class:`WorldActionScorer` rollout (realized per-slot ``Ũ`` + comm) under the epoch-frozen
  queue state.

This module is pure/CARLA-free -- it drives the existing scorer + Lyapunov machinery over a
``RolloutContext``, so it is fully offline-buildable and unit-testable. It is the reward backbone the
Dreamer world model (``wam_rssm`` / ``wam_world_model``) is trained to imagine, and the action codec
the factorized-discrete actor uses.

The uncertainty used in the reward is the **paper eq(11)-(12)** aggregate (:func:`eq12_motion_uncertainty`):
equal-weight mean of ``TrΣ_o`` over the GT-notable objects, then saturated ``1 - exp(-Ũ/σ0)`` -- NOT the
soft-gated notable-probability-weighted mean, which enlarges with the notable union and misleadingly
*rises* under cooperation (see docs/WAM/bandwidth_sweep_phase3.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .action_chunk import ActionChunk, SubAction
from .graph import MODALITIES
from .heads import per_object_trace
from .lyapunov import CostRateBreakdown, action_cost_rate
from .rollout_scorer import RolloutContext, RolloutResult, WorldActionScorer

__all__ = [
    "ActionSpec",
    "MDPConfig",
    "encode_subaction",
    "decode_subaction",
    "subaction_from_metadata",
    "subaction_cost",
    "eq12_motion_uncertainty",
]


# =====================================================================
# Action codec: SubAction <-> factorized-discrete indices (S, B, D, n)
# =====================================================================


@dataclass(frozen=True)
class ActionSpec:
    """Discretization of the paper action for the factorized-discrete actor.

    ``S`` is a member slot index in ``[0, max_members]`` where **0 == local-only (∅)** and ``k>=1``
    selects ``candidate_ids[k-1]`` (|S|<=1 in v1). ``B`` / ``D`` index the respective grids.

    Two framings (:attr:`use_duration`):

    * **framing A** (``use_duration=False``, default): the per-step action is ``(S, B, D)`` at a fixed
      per-step duration :attr:`step_slots`; the ``(S,B,D,n)`` trunk is recovered at inference by grouping
      consecutive same-action steps. Dreamer-standard fixed small time steps.
    * **framing B** (``use_duration=True``): the action is ``(S, B, D, n)`` with ``n`` a 4th categorical
      over :attr:`duration_grid` (a temporally-extended semi-MDP action).
    """

    max_members: int = 8
    bandwidth_grid: Tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)
    modalities: Tuple[str, ...] = ("objlist", "bev")
    duration_grid: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    use_duration: bool = False
    step_slots: int = 2

    def __post_init__(self) -> None:
        for m in self.modalities:
            if m not in MODALITIES:
                raise ValueError(f"modality {m!r} not in {MODALITIES}")

    @property
    def dims(self) -> Tuple[int, ...]:
        """Per-factor categorical sizes: ``(|S|, |B|, |D|)`` (framing A) or ``(|S|, |B|, |D|, |n|)`` (B)."""
        base = (self.max_members + 1, len(self.bandwidth_grid), len(self.modalities))
        return base + (len(self.duration_grid),) if self.use_duration else base

    def _nearest(self, grid: Sequence[float], value: float) -> int:
        return min(range(len(grid)), key=lambda i: abs(float(grid[i]) - float(value)))


def encode_subaction(sub: SubAction, candidate_ids: Sequence[int], spec: ActionSpec) -> Tuple[int, ...]:
    """Encode a :class:`SubAction` to factorized indices ``(s_idx, b_idx, d_idx[, n_idx])``.

    Local-only -> ``s_idx=0`` (B/D indices default to 0, ignored downstream). A cooperative sub-action's
    member must be in ``candidate_ids`` (its slot k -> ``s_idx = k+1``); B/D snap to the nearest grid entry.
    ``n_idx`` is appended only when ``spec.use_duration`` (framing B).
    """
    cand = [int(c) for c in candidate_ids]
    tail: Tuple[int, ...] = (spec._nearest(spec.duration_grid, sub.duration_slots),) if spec.use_duration else ()
    if sub.is_local_only:
        return (0, 0, 0) + tail
    m = int(sub.selected[0])
    s_idx = cand.index(m) + 1 if m in cand else 0
    if s_idx == 0:  # selected member not among candidates -> treat as local-only
        return (0, 0, 0) + tail
    b_idx = spec._nearest(spec.bandwidth_grid, float(sub.bandwidth_by_vehicle.get(m, 1.0)))
    mod = str(sub.modality_by_vehicle.get(m, spec.modalities[0]))
    d_idx = spec.modalities.index(mod) if mod in spec.modalities else 0
    return (s_idx, b_idx, d_idx) + tail


def decode_subaction(
    indices: Sequence[int], candidate_ids: Sequence[int], spec: ActionSpec
) -> SubAction:
    """Decode factorized indices back into a :class:`SubAction` (inverse of :func:`encode_subaction`).

    Duration comes from ``indices[3]`` over ``duration_grid`` (framing B) or the fixed ``spec.step_slots``
    (framing A).
    """
    idx = [int(i) for i in indices]
    s_idx, b_idx, d_idx = idx[0], idx[1], idx[2]
    cand = [int(c) for c in candidate_ids]
    if spec.use_duration and len(idx) >= 4:
        n = int(spec.duration_grid[idx[3] % len(spec.duration_grid)])
    else:
        n = int(spec.step_slots)
    if s_idx <= 0 or s_idx - 1 >= len(cand):  # local-only (or a slot no candidate fills)
        return SubAction(selected=(), duration_slots=n)
    m = cand[s_idx - 1]
    b = float(spec.bandwidth_grid[b_idx % len(spec.bandwidth_grid)])
    d = str(spec.modalities[d_idx % len(spec.modalities)])
    return SubAction(
        selected=(m,), bandwidth_by_vehicle={m: b}, modality_by_vehicle={m: d}, duration_slots=n
    )


def subaction_from_metadata(
    meta_action: Optional[Dict], candidate_ids: Sequence[int], *, duration_slots: int
) -> SubAction:
    """Parse a recorded ``active_subaction`` dict into a per-step :class:`SubAction`.

    The recorder stores ``{'S': [member], 'B': {member: ratio}, 'D': {member: [modalities]}, 'n': ...}``.
    Returns local-only when ``S`` is empty or its member is not among ``candidate_ids``. A multi-modality
    ``D`` (e.g. a member sent both) collapses to ``"bev"`` if present, else its first entry. The returned
    sub-action's duration is ``duration_slots`` (framing-A per-step), not the recorded ``n``.
    """
    cand = [int(c) for c in candidate_ids]
    if not meta_action or not meta_action.get("S"):
        return SubAction(selected=(), duration_slots=int(duration_slots))
    m = int(meta_action["S"][0])  # |S| <= 1
    if m not in cand:
        return SubAction(selected=(), duration_slots=int(duration_slots))
    b_map = meta_action.get("B", {})
    b = float(b_map.get(m, b_map.get(str(m), 1.0)))
    d_map = meta_action.get("D", {})
    mods = d_map.get(m, d_map.get(str(m), ["bev"]))
    mod = "bev" if "bev" in mods else (str(mods[0]) if mods else "bev")
    return SubAction(
        selected=(m,), bandwidth_by_vehicle={m: b}, modality_by_vehicle={m: mod},
        duration_slots=int(duration_slots),
    )


# =====================================================================
# Reward: -P2 cost-rate over one sub-action (epoch-frozen queue)
# =====================================================================


@dataclass
class MDPConfig:
    """Scalars for the P2 cost-rate reward (mirror :class:`LyapunovConfig` / scheduler)."""

    lam: float = 1.0
    c0: float = 0.5
    budget_bandwidth: float = 0.4


def subaction_cost(
    ctx: RolloutContext,
    sub: SubAction,
    scorer: WorldActionScorer,
    cfg: MDPConfig,
    *,
    z: float = 0.0,
    link_backlogs: Optional[Dict[int, float]] = None,
) -> Tuple[float, RolloutResult, CostRateBreakdown]:
    """Roll ``sub`` (as a 1-sub-action chunk) and return ``(reward, rollout, cost_breakdown)``.

    ``reward = -cost_breakdown.total`` (negated P2 cost-rate, paper eq 42), scored under the
    epoch-frozen queue state ``(z, link_backlogs)`` -- identical semantics to the online Lyapunov
    scheduler's per-frame program, so a Dreamer agent trained on this reward is directly comparable.
    """
    chunk = ActionChunk((sub,))
    roll = scorer.score_chunk(ctx, chunk)
    br = action_cost_rate(
        horizon_slots=chunk.horizon_slots,
        lam=cfg.lam,
        c0=cfg.c0,
        per_slot_uncertainty=roll.per_slot_uncertainty,
        per_slot_bandwidth=roll.per_slot_bandwidth,
        z=float(z),
        link_backlogs=link_backlogs or {},
        per_member_arrival_bits=roll.per_member_predicted_load_bits,
        per_member_service_bits=roll.per_member_predicted_service_bits,
    )
    return -float(br.total), roll, br


# =====================================================================
# Paper eq(11)-(12) motion uncertainty (equal-weight over GT-notable, saturated)
# =====================================================================


def eq12_motion_uncertainty(
    traj_log_var: torch.Tensor,
    object_node_ids: torch.Tensor,
    notable_ids: Sequence[int],
    *,
    sigma0: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Paper eq(11) aggregate + eq(12) normalization of the notable-object motion uncertainty.

    ``Ũ^mot = mean_{o ∈ notable} TrΣ_o`` (equal weights, GT notable set), then
    ``U^mot = 1 - exp(-Ũ^mot / σ0)``. Contrast with :func:`heads.policy_uncertainty` (soft-gated,
    notable-probability-weighted, divided by ``Σw + mass_floor``), which enlarges with the notable
    union and misleadingly rises under cooperation. Returns 0 when no notable object is present.
    """
    if traj_log_var.numel() == 0:
        return 0.0
    per_obj = per_object_trace(traj_log_var, valid_mask=valid_mask)  # [Q]
    ids = [int(v) for v in object_node_ids.tolist()]
    notable = {int(i) for i in notable_ids}
    sel = [i for i, oid in enumerate(ids) if oid in notable and i < int(per_obj.shape[0])]
    if not sel:
        return 0.0
    agg = float(sum(float(per_obj[i]) for i in sel) / len(sel))  # equal-weight mean
    return 1.0 - math.exp(-agg / max(float(sigma0), 1e-6))
