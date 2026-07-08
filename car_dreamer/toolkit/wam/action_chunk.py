"""Action-chunk data model (V2X paper Sec I.B) + segmentation into installable ``CommPolicy`` s.

The paper's cooperative-perception **action** is ``a = (S, B, D, n)``: a selected member set ``S``
(|S|<=1 in the v1 simplification), a per-member bandwidth allocation ``B``, per-member modalities ``D``
(``"bev"`` in v1), and an integer **execution duration** ``n`` slots (``n >= n_min``, eq 2). The
``local-only`` action ``a^loc = (∅, ∅, ∅, n)`` has no members.

At each decision epoch the controller issues an **action chunk** ``a_tk = (a^0, ..., a^{J-1})`` (ACT-style):
an ordered sequence of ``J`` sub-actions with sub-epoch start offsets ``t_k^(j) = Σ_{i<j} n^(i)`` (eq 4) and
**planned horizon** ``F(a) = Σ_j n^(j) <= F_max`` (eq 7). Setting ``J=1`` recovers the single
temporally-extended-action-per-epoch scheme.

This module is pure/CARLA-free: the data model, a heuristic candidate generator (the v1 stand-in for the
Sec III generative proposer ``W_θ``), and expansion of a chunk into a list of consecutive
:class:`~car_dreamer.toolkit.communication.process.CommPolicy` lifetimes (each sub-action installs as one
``CommPolicy`` segment, reusing the existing "a policy runs for its full duration" machinery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from ..communication.process import CommPolicy, make_local_policy
from .graph import MODALITIES

__all__ = [
    "SubAction",
    "ActionChunk",
    "local_only_chunk",
    "enumerate_candidate_chunks",
    "sub_action_to_comm_policy",
    "action_chunk_to_comm_segments",
]


# =====================================================================
# Data model
# =====================================================================


@dataclass(frozen=True)
class SubAction:
    """One sub-action ``a^(j) = (S, B, D, n)`` (paper Sec I.B).

    ``selected`` is the member set ``S`` (|S|<=1 in v1; empty == local-only). ``bandwidth_by_vehicle``
    is ``B`` (per-member ratio in ``[0, 1]``); ``modality_by_vehicle`` is ``D`` (member -> modality in
    :data:`MODALITIES`, ``"bev"`` in v1). ``duration_slots`` is ``n`` (slots).
    """

    selected: Tuple[int, ...] = ()
    bandwidth_by_vehicle: Dict[int, float] = field(default_factory=dict)
    modality_by_vehicle: Dict[int, str] = field(default_factory=dict)
    duration_slots: int = 1

    @property
    def is_local_only(self) -> bool:
        return len(self.selected) == 0

    def total_bandwidth(self) -> float:
        """``Σ_{m∈S} B_m`` — the allocated bandwidth ratio this sub-action consumes (0 for local-only)."""
        return float(sum(float(self.bandwidth_by_vehicle.get(int(m), 0.0)) for m in self.selected))


@dataclass(frozen=True)
class ActionChunk:
    """An action chunk ``a_tk = (a^0, ..., a^{J-1})`` (paper Sec I.B, eqs 3-7)."""

    sub_actions: Tuple[SubAction, ...]

    @property
    def num_subepochs(self) -> int:
        """``J`` — the number of sub-actions."""
        return len(self.sub_actions)

    @property
    def horizon_slots(self) -> int:
        """``F(a) = Σ_j n^(j)`` — the planned open-loop horizon (eq 7)."""
        return int(sum(int(a.duration_slots) for a in self.sub_actions))

    def subepoch_start_offsets(self) -> Tuple[int, ...]:
        """``t_k^(j) - t_k = Σ_{i<j} n^(i)`` — per-sub-epoch start offsets from ``t_k`` (eq 4)."""
        offsets: List[int] = []
        acc = 0
        for a in self.sub_actions:
            offsets.append(acc)
            acc += int(a.duration_slots)
        return tuple(offsets)

    def sub_action_at(self, tau: int) -> Tuple[int, SubAction]:
        """Return ``(j, a^(j))`` for local slot ``tau in [0, F)`` (eq 6, piecewise-constant action).

        ``tau`` at/after the horizon clamps to the last sub-action.
        """
        if not self.sub_actions:
            raise ValueError("empty ActionChunk has no sub-action")
        tau = int(tau)
        acc = 0
        for j, a in enumerate(self.sub_actions):
            acc += int(a.duration_slots)
            if tau < acc:
                return j, a
        return len(self.sub_actions) - 1, self.sub_actions[-1]

    def per_subepoch_bandwidth(self) -> Tuple[float, ...]:
        """``Σ B^(j)`` per sub-epoch (used by the cost-rate bandwidth term, eq 42)."""
        return tuple(a.total_bandwidth() for a in self.sub_actions)


# =====================================================================
# Candidate generation (heuristic — v1 stand-in for the Sec III proposer W_θ)
# =====================================================================


def local_only_chunk(*, n_slots: int, n_min: int = 1) -> ActionChunk:
    """The always-feasible ``a^loc`` chunk: one local-only sub-action of duration ``max(n_slots, n_min)``.

    This is the maximal-duration local-only baseline that keeps the per-frame program (P2) always
    feasible and gives the no-degradation guarantee (paper Sec IV.C).
    """
    n = max(int(n_slots), int(n_min))
    return ActionChunk((SubAction(selected=(), duration_slots=n),))


def _cooperative_sub_actions(
    candidates: Sequence[int],
    bandwidth_grid: Sequence[float],
    duration_grid: Sequence[int],
    modality: str,
) -> List[SubAction]:
    subs: List[SubAction] = []
    for m in candidates:
        mi = int(m)
        for b in bandwidth_grid:
            for d in duration_grid:
                subs.append(
                    SubAction(
                        selected=(mi,),
                        bandwidth_by_vehicle={mi: float(b)},
                        modality_by_vehicle={mi: str(modality)},
                        duration_slots=int(d),
                    )
                )
    return subs


def enumerate_candidate_chunks(
    candidates: Sequence[int],
    *,
    bandwidth_grid: Sequence[float] = (0.2, 0.5, 0.8, 1.0),
    duration_grid: Sequence[int] = (5, 10, 20),
    j_max: int = 1,
    modality: str = "bev",
    f_max: Optional[int] = None,
    n_min: int = 1,
    include_local_only: bool = True,
    local_only_slots: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> List[ActionChunk]:
    """Heuristic candidate action chunks (the v1 stand-in for the paper's generative proposer ``W_θ``).

    Single-member cooperative sub-actions are formed as ``member × bandwidth_grid × duration_grid`` (|S|<=1);
    chunks of length ``1..j_max`` are their products (plus a local sub-action per duration so ``J>1`` chunks
    can interleave cooperation and local sensing). Chunks whose horizon ``F(a) > f_max`` are pruned. The
    always-feasible :func:`local_only_chunk` is appended (paper Sec IV: candidate set = proposals + local-only).
    """
    if modality not in MODALITIES:
        raise ValueError(f"modality {modality!r} not in {MODALITIES}")
    base = _cooperative_sub_actions(candidates, bandwidth_grid, duration_grid, modality)
    # local sub-action options let J>1 chunks alternate cooperation with local sensing
    base_with_local = list(base) + [SubAction(selected=(), duration_slots=int(d)) for d in duration_grid]

    chunks: List[ActionChunk] = []
    seen: set = set()

    def _add(chunk: ActionChunk) -> bool:
        if f_max is not None and chunk.horizon_slots > int(f_max):
            return True  # infeasible-by-horizon: skip but keep enumerating
        key = tuple(
            (a.selected, tuple(sorted(a.bandwidth_by_vehicle.items())), a.duration_slots)
            for a in chunk.sub_actions
        )
        if key in seen:
            return True
        seen.add(key)
        chunks.append(chunk)
        return not (max_candidates is not None and len(chunks) >= int(max_candidates))

    for length in range(1, max(int(j_max), 1) + 1):
        pool = base if length == 1 else base_with_local
        for combo in product(pool, repeat=length):
            if not _add(ActionChunk(tuple(combo))):
                break
        else:
            continue
        break

    if include_local_only:
        slots = int(local_only_slots) if local_only_slots is not None else int(max(duration_grid))
        chunks.append(local_only_chunk(n_slots=slots, n_min=n_min))
    return chunks


# =====================================================================
# Segmentation: ActionChunk -> installable CommPolicy segments
# =====================================================================


def sub_action_to_comm_policy(
    sub: SubAction,
    *,
    policy_id: int,
    request_vehicle_id: int,
    start_step: int,
    default_modality: str = "bev",
) -> CommPolicy:
    """Wrap one sub-action as an installable :class:`CommPolicy` of lifetime ``duration_slots``.

    Local-only sub-action -> :func:`make_local_policy`. Cooperative -> a ``CommPolicy`` whose
    ``modalities_by_vehicle`` values are singleton tuples (the comm layer bundles a member's modalities
    into one message).
    """
    if sub.is_local_only:
        return make_local_policy(
            policy_id=int(policy_id),
            request_vehicle_id=int(request_vehicle_id),
            start_step=int(start_step),
            duration_steps=int(sub.duration_slots),
        )
    selected = tuple(int(m) for m in sub.selected)
    modalities = {
        int(m): (str(sub.modality_by_vehicle.get(int(m), default_modality)),) for m in selected
    }
    bandwidth = {int(m): float(sub.bandwidth_by_vehicle.get(int(m), 1.0)) for m in selected}
    return CommPolicy(
        policy_id=int(policy_id),
        request_vehicle_id=int(request_vehicle_id),
        start_step=int(start_step),
        duration_steps=int(sub.duration_slots),
        selected_collaborators=selected,
        modalities_by_vehicle=modalities,
        bandwidth_by_vehicle=bandwidth,
        reason="lyapunov_chunk",
    )


def action_chunk_to_comm_segments(
    chunk: ActionChunk,
    *,
    first_policy_id: int,
    request_vehicle_id: int,
    start_step: int,
) -> List[CommPolicy]:
    """Expand a chunk into consecutive :class:`CommPolicy` segments (one per sub-action).

    Segment ``j`` starts at ``start_step + subepoch_start_offsets()[j]`` and runs for ``n^(j)`` slots, so
    the segments tile ``[start_step, start_step + F(a))`` with no gap or overlap.
    """
    offsets = chunk.subepoch_start_offsets()
    segments: List[CommPolicy] = []
    for j, sub in enumerate(chunk.sub_actions):
        segments.append(
            sub_action_to_comm_policy(
                sub,
                policy_id=int(first_policy_id) + j,
                request_vehicle_id=int(request_vehicle_id),
                start_step=int(start_step) + int(offsets[j]),
            )
        )
    return segments
