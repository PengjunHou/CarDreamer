"""Fixed collaboration policy family for policy-conditioned data generation.

Implements the 8 fixed policies defined in Section IV.B-C of the paper:
  P1  Ego-Only            — no collaboration
  P2  Full-Low-Equal      — all vehicles, low freq, equal bandwidth
  P3  Full-High-Equal     — all vehicles, high freq, equal bandwidth
  P4  Top2-Mid-Equal      — top-2 by s_collab, mid freq, equal bandwidth
  P5  Top2-Adaptive-Value — top-2 by s_collab, adaptive freq, value-weighted BW
  P6  Nearest2-Mid-Dist   — 2 closest vehicles, mid freq, distance-weighted BW
  P7  Random2-Mid-Equal   — random 2 vehicles, mid freq, equal bandwidth
  P8  Random3-Adaptive    — random 3 vehicles, adaptive freq, value-weighted BW

Each policy produces a CollaborationAction that specifies per-vehicle
(alpha_i, nu_i, bandwidth_i) for a given timestep.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# Sharing frequency levels (Section IV.B)
NU_LOW: float = 0.2
NU_MID: float = 0.5
NU_HIGH: float = 1.0

# Total normalized bandwidth budget B_rx (Section II)
BANDWIDTH_BUDGET: float = 1.0


@dataclass
class VehicleInfo:
    """Minimal per-vehicle info needed by policies to make decisions."""
    vehicle_id: int
    sender_collab: float   # aggregate collaboration value s_collab (scalar summary)
    distance_m: float      # spatial distance to ego


@dataclass
class CollaborationAction:
    """Policy action u_t = ({alpha_i}, {nu_i}, {W_i}) for all candidate vehicles.

    Attributes:
        alpha:     {vehicle_id -> 0 or 1} — selection indicator
        nu:        {vehicle_id -> float}  — sharing frequency
        bandwidth: {vehicle_id -> float}  — allocated bandwidth (normalized)
    """
    alpha: Dict[int, float] = field(default_factory=dict)
    nu: Dict[int, float] = field(default_factory=dict)
    bandwidth: Dict[int, float] = field(default_factory=dict)

    def is_selected(self, vehicle_id: int) -> bool:
        return bool(self.alpha.get(vehicle_id, 0.0) > 0.5)

    def active_ids(self) -> List[int]:
        return [vid for vid, a in self.alpha.items() if a > 0.5]


class FixedCollaborationPolicy(ABC):
    """Base class for all fixed collaboration policies."""

    policy_id: str = ""

    @abstractmethod
    def __call__(
        self,
        vehicles: Sequence[VehicleInfo],
        rng: random.Random | None = None,
    ) -> CollaborationAction:
        """Compute the collaboration action for the given candidate vehicles.

        Args:
            vehicles: candidate collaborator vehicles at time t
            rng:      optional random source for stochastic policies

        Returns:
            CollaborationAction with per-vehicle alpha, nu, bandwidth
        """

    def _equal_bandwidth(self, selected_ids: List[int], all_ids: List[int]) -> Dict[int, float]:
        """Equal bandwidth split among selected vehicles; 0 for unselected."""
        n = len(selected_ids)
        bw_each = BANDWIDTH_BUDGET / n if n > 0 else 0.0
        return {vid: (bw_each if vid in selected_ids else 0.0) for vid in all_ids}

    def _value_weighted_bandwidth(
        self,
        selected_ids: List[int],
        collab_map: Dict[int, float],
        all_ids: List[int],
    ) -> Dict[int, float]:
        """Bandwidth proportional to s_collab; 0 for unselected."""
        total = sum(max(collab_map.get(vid, 0.0), 0.0) for vid in selected_ids)
        bw: Dict[int, float] = {}
        for vid in all_ids:
            if vid in selected_ids and total > 0:
                bw[vid] = BANDWIDTH_BUDGET * max(collab_map.get(vid, 0.0), 0.0) / total
            elif vid in selected_ids:
                bw[vid] = BANDWIDTH_BUDGET / len(selected_ids)
            else:
                bw[vid] = 0.0
        return bw

    def _distance_weighted_bandwidth(
        self,
        selected_ids: List[int],
        dist_map: Dict[int, float],
        all_ids: List[int],
    ) -> Dict[int, float]:
        """Bandwidth proportional to 1/distance; 0 for unselected."""
        inv_dists = {vid: 1.0 / max(dist_map.get(vid, 1.0), 0.1) for vid in selected_ids}
        total = sum(inv_dists.values())
        bw: Dict[int, float] = {}
        for vid in all_ids:
            if vid in selected_ids and total > 0:
                bw[vid] = BANDWIDTH_BUDGET * inv_dists[vid] / total
            elif vid in selected_ids:
                bw[vid] = BANDWIDTH_BUDGET / len(selected_ids)
            else:
                bw[vid] = 0.0
        return bw

    def _make_action(
        self,
        selected_ids: List[int],
        nu_map: Dict[int, float],
        bw_map: Dict[int, float],
        all_ids: List[int],
    ) -> CollaborationAction:
        alpha = {vid: (1.0 if vid in selected_ids else 0.0) for vid in all_ids}
        nu = {vid: (nu_map.get(vid, 0.0) if vid in selected_ids else 0.0) for vid in all_ids}
        return CollaborationAction(alpha=alpha, nu=nu, bandwidth=bw_map)


# ---------------------------------------------------------------------------
# P1 — Ego-Only: no collaboration
# ---------------------------------------------------------------------------
class EgoOnlyPolicy(FixedCollaborationPolicy):
    """P1: No vehicle is selected; ego relies only on its own sensors."""

    policy_id = "P1"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        all_ids = [v.vehicle_id for v in vehicles]
        return CollaborationAction(
            alpha={vid: 0.0 for vid in all_ids},
            nu={vid: 0.0 for vid in all_ids},
            bandwidth={vid: 0.0 for vid in all_ids},
        )


# ---------------------------------------------------------------------------
# P2 — Full-Low-Equal: all vehicles, low freq, equal bandwidth
# ---------------------------------------------------------------------------
class FullLowEqualPolicy(FixedCollaborationPolicy):
    """P2: All candidate vehicles selected, uniform low frequency, equal bandwidth."""

    policy_id = "P2"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        all_ids = [v.vehicle_id for v in vehicles]
        nu_map = {vid: NU_LOW for vid in all_ids}
        bw_map = self._equal_bandwidth(all_ids, all_ids)
        return self._make_action(all_ids, nu_map, bw_map, all_ids)


# ---------------------------------------------------------------------------
# P3 — Full-High-Equal: all vehicles, high freq, equal bandwidth
# ---------------------------------------------------------------------------
class FullHighEqualPolicy(FixedCollaborationPolicy):
    """P3: All candidate vehicles selected, uniform high frequency, equal bandwidth."""

    policy_id = "P3"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        all_ids = [v.vehicle_id for v in vehicles]
        nu_map = {vid: NU_HIGH for vid in all_ids}
        bw_map = self._equal_bandwidth(all_ids, all_ids)
        return self._make_action(all_ids, nu_map, bw_map, all_ids)


# ---------------------------------------------------------------------------
# P4 — Top2-Mid-Equal: top-2 by s_collab, mid freq, equal bandwidth
# ---------------------------------------------------------------------------
class Top2MidEqualPolicy(FixedCollaborationPolicy):
    """P4: Select 2 vehicles with highest s_collab, mid frequency, equal bandwidth."""

    policy_id = "P4"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        all_ids = [v.vehicle_id for v in vehicles]
        sorted_veh = sorted(vehicles, key=lambda v: v.sender_collab, reverse=True)
        selected = [v.vehicle_id for v in sorted_veh[:2]]
        nu_map = {vid: NU_MID for vid in selected}
        bw_map = self._equal_bandwidth(selected, all_ids)
        return self._make_action(selected, nu_map, bw_map, all_ids)


# ---------------------------------------------------------------------------
# P5 — Top2-Adaptive-Value: top-2 by s_collab, adaptive freq, value-weighted BW
# ---------------------------------------------------------------------------
class Top2AdaptiveValuePolicy(FixedCollaborationPolicy):
    """P5: Top-2 by s_collab; rank-1 gets high freq, rank-2 gets mid freq;
    bandwidth allocated proportional to s_collab."""

    policy_id = "P5"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        all_ids = [v.vehicle_id for v in vehicles]
        sorted_veh = sorted(vehicles, key=lambda v: v.sender_collab, reverse=True)
        top2 = sorted_veh[:2]
        selected = [v.vehicle_id for v in top2]
        nu_map: Dict[int, float] = {}
        for rank, v in enumerate(top2):
            nu_map[v.vehicle_id] = NU_HIGH if rank == 0 else NU_MID
        collab_map = {v.vehicle_id: v.sender_collab for v in vehicles}
        bw_map = self._value_weighted_bandwidth(selected, collab_map, all_ids)
        return self._make_action(selected, nu_map, bw_map, all_ids)


# ---------------------------------------------------------------------------
# P6 — Nearest2-Mid-Dist: 2 closest vehicles, mid freq, distance-weighted BW
# ---------------------------------------------------------------------------
class Nearest2MidDistPolicy(FixedCollaborationPolicy):
    """P6: Select 2 closest vehicles; mid frequency; bandwidth proportional to 1/distance."""

    policy_id = "P6"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        all_ids = [v.vehicle_id for v in vehicles]
        sorted_veh = sorted(vehicles, key=lambda v: v.distance_m)
        selected = [v.vehicle_id for v in sorted_veh[:2]]
        nu_map = {vid: NU_MID for vid in selected}
        dist_map = {v.vehicle_id: v.distance_m for v in vehicles}
        bw_map = self._distance_weighted_bandwidth(selected, dist_map, all_ids)
        return self._make_action(selected, nu_map, bw_map, all_ids)


# ---------------------------------------------------------------------------
# P7 — Random2-Mid-Equal: random 2 vehicles, mid freq, equal bandwidth
# ---------------------------------------------------------------------------
class Random2MidEqualPolicy(FixedCollaborationPolicy):
    """P7: Randomly select 2 vehicles; mid frequency; equal bandwidth."""

    policy_id = "P7"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        _rng = rng or random.Random()
        all_ids = [v.vehicle_id for v in vehicles]
        k = min(2, len(all_ids))
        selected = _rng.sample(all_ids, k)
        nu_map = {vid: NU_MID for vid in selected}
        bw_map = self._equal_bandwidth(selected, all_ids)
        return self._make_action(selected, nu_map, bw_map, all_ids)


# ---------------------------------------------------------------------------
# P8 — Random3-Adaptive-Value: random 3 vehicles, adaptive freq, value-weighted BW
# ---------------------------------------------------------------------------
class Random3AdaptiveValuePolicy(FixedCollaborationPolicy):
    """P8: Randomly select 3 vehicles; rank-1 gets high freq, others mid freq;
    bandwidth proportional to s_collab."""

    policy_id = "P8"

    def __call__(self, vehicles: Sequence[VehicleInfo], rng=None) -> CollaborationAction:
        _rng = rng or random.Random()
        all_ids = [v.vehicle_id for v in vehicles]
        k = min(3, len(all_ids))
        selected_ids = _rng.sample(all_ids, k)
        # sort selected by s_collab descending to assign adaptive frequencies
        collab_map = {v.vehicle_id: v.sender_collab for v in vehicles}
        selected_sorted = sorted(selected_ids, key=lambda vid: collab_map.get(vid, 0.0), reverse=True)
        nu_map: Dict[int, float] = {}
        for rank, vid in enumerate(selected_sorted):
            nu_map[vid] = NU_HIGH if rank == 0 else NU_MID
        bw_map = self._value_weighted_bandwidth(selected_ids, collab_map, all_ids)
        return self._make_action(selected_ids, nu_map, bw_map, all_ids)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_POLICIES: List[FixedCollaborationPolicy] = [
    EgoOnlyPolicy(),
    FullLowEqualPolicy(),
    FullHighEqualPolicy(),
    Top2MidEqualPolicy(),
    Top2AdaptiveValuePolicy(),
    Nearest2MidDistPolicy(),
    Random2MidEqualPolicy(),
    Random3AdaptiveValuePolicy(),
]

POLICY_REGISTRY: Dict[str, FixedCollaborationPolicy] = {p.policy_id: p for p in ALL_POLICIES}

# Canonical seen/unseen split for Section IV.H evaluation protocols
SEEN_POLICY_IDS: List[str] = ["P1", "P2", "P3", "P4", "P6"]
UNSEEN_POLICY_IDS: List[str] = ["P5", "P7", "P8"]


def get_policy(policy_id: str) -> FixedCollaborationPolicy:
    """Look up a policy by ID string (e.g. 'P1', 'P4')."""
    if policy_id not in POLICY_REGISTRY:
        raise KeyError(f"Unknown policy_id '{policy_id}'. Valid IDs: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[policy_id]


def list_policy_ids() -> List[str]:
    return sorted(POLICY_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helpers to bridge CandidateVehicleState → VehicleInfo
# ---------------------------------------------------------------------------

def vehicle_info_from_state(vehicle) -> VehicleInfo:
    """Convert a CandidateVehicleState to the VehicleInfo needed by policies."""
    import math
    collab = vehicle.sender_collab
    scalar_collab = float(sum(collab.values()) / len(collab)) if collab else float(vehicle.complementarity * vehicle.accessibility)
    dist = vehicle.communication_stats.get("distance_m")
    if dist is None:
        dist = math.sqrt(float(vehicle.delta_pos[0]) ** 2 + float(vehicle.delta_pos[1]) ** 2)
    return VehicleInfo(vehicle_id=int(vehicle.vehicle_id), sender_collab=scalar_collab, distance_m=float(dist))


# ---------------------------------------------------------------------------
# Episode-level helpers (Section IV.E: trajectory generation)
# ---------------------------------------------------------------------------

def apply_action_to_episode(episode, policy: FixedCollaborationPolicy, *, seed_base: int = 0):
    """Return a new episode with policy actions written into each step's vehicle fields.

    For each step, the policy is called with a per-step seed to produce
    (alpha_i, nu_i, bandwidth_i) for every candidate vehicle. These values are
    stored back into CandidateVehicleState so the dataset can include them in
    the node feature vector for policy-conditioned learning.
    """
    import dataclasses as _dc
    new_steps = []
    for step_idx, step in enumerate(episode.steps):
        infos = [vehicle_info_from_state(v) for v in step.candidate_vehicles]
        rng = random.Random(seed_base + step_idx)
        action = policy(infos, rng=rng)
        new_vehicles = []
        for v in step.candidate_vehicles:
            vid = v.vehicle_id
            new_v = _dc.replace(
                v,
                alpha=float(action.alpha.get(vid, 0.0)),
                nu=float(action.nu.get(vid, 0.0)),
                bandwidth=float(action.bandwidth.get(vid, 0.0)),
            )
            new_vehicles.append(new_v)
        new_step = _dc.replace(step, candidate_vehicles=new_vehicles, policy_id=policy.policy_id)
        new_steps.append(new_step)
    return _dc.replace(
        episode,
        steps=new_steps,
        policy_id=policy.policy_id,
        metadata={**episode.metadata, "policy_id": policy.policy_id},
    )


def generate_policy_conditioned_episodes(base_episode, policies=None, *, seed_base: int = 0):
    """Generate one episode variant per policy from a single base episode.

    Implements Section IV.E trajectory generation: the same underlying scene
    is replayed under each of the 8 fixed policies to produce the diverse
    training dataset for policy-conditioned dynamics learning.
    """
    import dataclasses as _dc
    if policies is None:
        policies = ALL_POLICIES
    result = []
    for pol in policies:
        new_ep = apply_action_to_episode(base_episode, pol, seed_base=seed_base)
        new_ep = _dc.replace(new_ep, episode_id=f"{base_episode.episode_id}__{pol.policy_id}")
        result.append(new_ep)
    return result
