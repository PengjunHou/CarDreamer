"""Lyapunov-guided world-action-model policy search (V2X paper Sec IV, Algorithm 1).

At each decision epoch ``t_k`` the base station solves the per-frame program (P2) in ratio form — frame cost
divided by planned horizon ``F(a)`` — over a small candidate set (heuristic chunks + the always-feasible
maximal-duration local-only chunk), scoring each candidate by rolling out the coupled perception/motion
dynamics (:class:`WorldActionScorer`) and pricing bandwidth with the virtual queue ``Z`` and each link's
backlog ``Q_m``. The chosen chunk is installed as ``CommPolicy`` segments; its predicted per-slot
uncertainty is stored as the **reference trajectory** ``Ũ_{v,t|tk}``. Between epochs the queues evolve per
slot (eq 27/35) and the realized uncertainty is compared against the reference — a deviation beyond
``ε_gap`` after ``T_min`` triggers an early re-plan (event-driven epoch, eq 8):

    ``t_{k+1} = min{ t_k + F(a),  inf{ t > t_k + T_min : |U_{v,t} − Ũ_{v,t|tk}| > ε_gap } }``.

Keeping local-only as an always-feasible candidate gives the **no-degradation** property: the installed
chunk's cost rate never exceeds the local-only baseline (paper Sec IV.C). No Stage-2 UWM is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .action_chunk import ActionChunk, action_chunk_to_comm_segments, enumerate_candidate_chunks, local_only_chunk
from .lyapunov import CostRateBreakdown, LyapunovConfig, LyapunovState, action_cost_rate
from .rollout_scorer import RolloutContext, RolloutResult, WorldActionScorer
from .stage1_policy import bev_payload_bytes

__all__ = [
    "SchedulerConfig",
    "ReferenceTrajectory",
    "candidate_chunks",
    "select_action_chunk",
    "next_epoch_step",
    "LyapunovScheduler",
]

RealizedFn = Callable[[int], Optional[float]]  # step -> realized U (or None if unknown)


@dataclass
class SchedulerConfig:
    """Sec IV scheduler hyperparameters."""

    lam: float = 1.0                 # Λ
    c0: float = 0.5                  # amortized replan penalty
    budget_bandwidth: float = 0.5    # B̄_bgt (per-slot allocated-bandwidth budget, ratio)
    F_max_slots: int = 20            # F_max (eq 7)
    n_min_slots: int = 5             # n_min (eq 2)
    B_max_ratio: float = 1.0         # Σ B ≤ B_max (per sub-action)
    bandwidth_grid: Tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)
    duration_grid: Tuple[int, ...] = (5, 10, 20)
    j_max: int = 2
    eps_gap: float = 0.15            # ε_gap (eq 8)
    t_min_slots: int = 3             # T_min (eq 8)
    T_a_slots: int = 5               # Ta (eq 8 monitoring cadence)
    ts_seconds: float = 0.5          # Ts (slot duration for the queue bits<->rate bridge)


# =====================================================================
# Candidate generation + (P2) selection
# =====================================================================


def candidate_chunks(ctx: RolloutContext, cfg: SchedulerConfig) -> List[ActionChunk]:
    """Feasible candidates: heuristic chunks + always-feasible local-only, filtered by (P2) constraints."""
    members = sorted({int(c.actor_id) for c in ctx.collaborators})
    chunks = enumerate_candidate_chunks(
        members,
        bandwidth_grid=cfg.bandwidth_grid, duration_grid=cfg.duration_grid, j_max=cfg.j_max,
        modality="bev", f_max=cfg.F_max_slots, n_min=cfg.n_min_slots,
        local_only_slots=cfg.F_max_slots,
    )
    feasible: List[ActionChunk] = []
    for c in chunks:
        if c.horizon_slots > cfg.F_max_slots:
            continue
        if any(sa.duration_slots < cfg.n_min_slots for sa in c.sub_actions):
            continue
        if any(len(sa.selected) > 1 for sa in c.sub_actions):  # |S_j| ≤ 1 (v1)
            continue
        if any(sa.total_bandwidth() > cfg.B_max_ratio + 1e-9 for sa in c.sub_actions):
            continue
        feasible.append(c)
    if not feasible:  # degenerate config -> at least the local-only baseline
        feasible = [local_only_chunk(n_slots=max(cfg.n_min_slots, 1), n_min=cfg.n_min_slots)]
    return feasible


def _score_candidate(
    ctx: RolloutContext, chunk: ActionChunk, scorer: WorldActionScorer, lyap: LyapunovState, cfg: SchedulerConfig
) -> Tuple[CostRateBreakdown, RolloutResult]:
    roll = scorer.score_chunk(ctx, chunk)
    br = action_cost_rate(
        horizon_slots=chunk.horizon_slots, lam=cfg.lam, c0=cfg.c0,
        per_slot_uncertainty=roll.per_slot_uncertainty,
        per_slot_bandwidth=roll.per_slot_bandwidth,
        z=lyap.z.value,
        link_backlogs=lyap.backlogs(),
        per_member_arrival_bits=roll.per_member_predicted_load_bits,
        per_member_service_bits=roll.per_member_predicted_service_bits,
    )
    return br, roll


def select_action_chunk(
    ctx: RolloutContext, *, scorer: WorldActionScorer, lyap: LyapunovState, cfg: SchedulerConfig
) -> Tuple[ActionChunk, CostRateBreakdown, RolloutResult]:
    """(P2) argmin over feasible candidates. Local-only is always present -> no-degradation guarantee."""
    best: Optional[Tuple[float, ActionChunk, CostRateBreakdown, RolloutResult]] = None
    for chunk in candidate_chunks(ctx, cfg):
        br, roll = _score_candidate(ctx, chunk, scorer, lyap, cfg)
        if best is None or br.total < best[0]:
            best = (br.total, chunk, br, roll)
    assert best is not None
    return best[1], best[2], best[3]


# =====================================================================
# Reference trajectory + event-driven epoch (eq 8)
# =====================================================================


@dataclass(frozen=True)
class ReferenceTrajectory:
    """The installed chunk's predicted per-slot uncertainty ``Ũ_{v,t|tk}`` (eq 8 trigger reference)."""

    install_step: int
    horizon_slots: int
    per_slot_uncertainty: Tuple[float, ...]

    def expected_at(self, step: int) -> Optional[float]:
        i = int(step) - int(self.install_step)
        if 0 <= i < len(self.per_slot_uncertainty):
            return float(self.per_slot_uncertainty[i])
        return None


def next_epoch_step(ref: ReferenceTrajectory, *, realized_uncertainty_fn: RealizedFn, cfg: SchedulerConfig) -> int:
    """``t_{k+1} = min{t_k+F, inf{t> t_k+T_min: |U−Ũ|>ε_gap}}`` sampled every ``T_a`` (eq 8)."""
    end = int(ref.install_step) + int(ref.horizon_slots)
    t = int(ref.install_step) + int(cfg.t_min_slots) + 1
    step = max(int(cfg.T_a_slots), 1)
    while t < end:
        exp = ref.expected_at(t)
        real = realized_uncertainty_fn(t)
        if exp is not None and real is not None and abs(float(real) - float(exp)) > cfg.eps_gap:
            return t
        t += step
    return end


# =====================================================================
# Algorithm 1 driver (offline + online-shareable)
# =====================================================================


class LyapunovScheduler:
    """Stateful Sec IV / Algorithm-1 driver. Drives an offline episode or backs the online sampler."""

    def __init__(self, cfg: SchedulerConfig, scorer: WorldActionScorer, *, request_vehicle_id: int = 0):
        self.cfg = cfg
        self.scorer = scorer
        self.request_vehicle_id = int(request_vehicle_id)
        self.lyap = LyapunovState(
            LyapunovConfig(lam=cfg.lam, c0=cfg.c0, budget_bandwidth=cfg.budget_bandwidth, ts_seconds=cfg.ts_seconds)
        )
        self.ref: Optional[ReferenceTrajectory] = None
        self.chunk: Optional[ActionChunk] = None
        self.next_epoch: Optional[int] = None
        self._policy_id = 0
        self.epochs = 0
        self.replans = 0
        self._u_c_sum = 0.0
        self._slots = 0

    # ---- decision logic ----

    def is_decision_epoch(self, step: int, realized_uncertainty_fn: Optional[RealizedFn] = None) -> bool:
        if self.next_epoch is None:
            return True
        if int(step) >= int(self.next_epoch):
            return True
        if self.ref is not None and realized_uncertainty_fn is not None:
            exp = self.ref.expected_at(step)
            real = realized_uncertainty_fn(step)
            past_min = int(step) > int(self.ref.install_step) + int(self.cfg.t_min_slots)
            if past_min and exp is not None and real is not None and abs(float(real) - float(exp)) > self.cfg.eps_gap:
                self.replans += 1
                return True
        return False

    def plan(self, ctx: RolloutContext, step: int):
        """Solve (P2), install the chunk, store the reference trajectory. Returns ``(segments, breakdown, chunk, rollout)``."""
        chunk, br, roll = select_action_chunk(ctx, scorer=self.scorer, lyap=self.lyap, cfg=self.cfg)
        segments = action_chunk_to_comm_segments(
            chunk, first_policy_id=self._policy_id, request_vehicle_id=self.request_vehicle_id, start_step=int(step)
        )
        self._policy_id += len(segments)
        self.ref = ReferenceTrajectory(int(step), chunk.horizon_slots, tuple(roll.per_slot_uncertainty))
        self.chunk = chunk
        self.next_epoch = int(step) + chunk.horizon_slots
        self.epochs += 1
        return segments, br, chunk, roll

    # ---- per-slot queue evolution ----

    def _slot_comm(self, ctx: RolloutContext, step: int) -> Tuple[Dict[int, float], Dict[int, float], float]:
        """Realized per-slot ``(service_bits, arrival_bits, allocated_bandwidth)`` for the active sub-action."""
        if self.chunk is None or self.ref is None:
            return {}, {}, 0.0
        _, sub = self.chunk.sub_action_at(int(step) - int(self.ref.install_step))
        service: Dict[int, float] = {}
        arrival: Dict[int, float] = {}
        collab_by_id = {int(c.actor_id): c for c in ctx.collaborators}
        ts = max(int(ctx.sensor_period_steps), 1)
        bit_scale = max(float(getattr(self.scorer, "bit_scale", 1.0)), 1.0)
        for m in sub.selected:
            collab = collab_by_id.get(int(m))
            if collab is None:
                continue
            dist = math.hypot(float(collab.x) - float(ctx.ego.x), float(collab.y) - float(ctx.ego.y))
            ratio = float(sub.bandwidth_by_vehicle.get(int(m), 1.0))
            rate_bps = max(float(ctx.link_rate_fn(int(m), dist, ratio)), 1.0)
            # per-slot service R·dt; per-tick arrival L; in payload units (÷ bit_scale) to match the scorer
            service[int(m)] = rate_bps * float(ctx.dt_seconds) / bit_scale
            if int(step) % ts == 0:
                arrival[int(m)] = 8.0 * float(bev_payload_bytes(ctx.bev_spec)) / bit_scale
        return service, arrival, float(sub.total_bandwidth())

    def observe_slot(self, ctx: RolloutContext, step: int) -> None:
        service, arrival, allocated = self._slot_comm(ctx, step)
        self.lyap.advance_slot(
            per_member_service_bits=service, per_member_arrival_bits=arrival, allocated_bandwidth=allocated
        )

    # ---- offline episode driver ----

    def run_offline(
        self,
        contexts: Sequence[RolloutContext],
        *,
        realized_uncertainty_fn: Optional[RealizedFn] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """Drive Algorithm 1 over a per-step context sequence. Returns ``(rows, summary)``.

        ``realized_uncertainty_fn(step)`` supplies the realized ``U`` for the eq-8 trigger and the ``U_c``
        accumulation; when ``None`` the reference (no-mismatch) value is used, so epochs occur every ``F``.
        """
        n = len(contexts) if max_steps is None else min(len(contexts), int(max_steps))
        realized_fn = realized_uncertainty_fn or (lambda s: (self.ref.expected_at(s) if self.ref else None))
        rows: List[Dict[str, float]] = []
        last_br: Optional[CostRateBreakdown] = None
        for step in range(n):
            ctx = contexts[step]
            replan_here = 0
            if self.is_decision_epoch(step, realized_fn):
                was_planned = self.ref is not None
                _, last_br, chunk, _ = self.plan(ctx, step)
                if was_planned:
                    replan_here = 1  # a re-plan before the scheduled horizon (or a fresh epoch)
            # active sub-action bandwidth + realized uncertainty
            local_slot = int(step) - int(self.ref.install_step) if self.ref else 0
            _, sub = self.chunk.sub_action_at(local_slot) if self.chunk else (0, None)
            realized_u = realized_fn(step)
            if realized_u is None:
                realized_u = self.ref.expected_at(step) if self.ref else 0.0
            self.observe_slot(ctx, step)
            snap = self.lyap.snapshot()
            self._u_c_sum += float(realized_u)
            self._slots += 1
            br = last_br or CostRateBreakdown(0.0, 0.0, 0.0, 0.0)
            rows.append({
                "step": float(step),
                "epoch": 1.0 if (self.ref is not None and self.ref.install_step == step) else 0.0,
                "replan": float(replan_here),
                "active_members": float(len(sub.selected)) if sub is not None else 0.0,
                "allocated_bandwidth": float(sub.total_bandwidth()) if sub is not None else 0.0,
                "expected_u": float(self.ref.expected_at(step)) if (self.ref and self.ref.expected_at(step) is not None) else float("nan"),
                "realized_u": float(realized_u),
                "z": snap["z"],
                "total_backlog": snap["total_backlog"],
                "planning_rate": br.planning_rate,
                "uncertainty_rate": br.uncertainty_rate,
                "bandwidth_rate": br.bandwidth_rate,
                "net_load_rate": br.net_load_rate,
                "cost_rate_total": br.total,
            })
        summary = self.summary()
        return rows, summary

    def summary(self) -> Dict[str, float]:
        snap = self.lyap.snapshot()
        t = max(self._slots, 1)
        u_bar = self._u_c_sum / t
        return {
            "steps": float(self._slots),
            "epochs": float(self.epochs),
            "replans": float(self.replans),
            "time_avg_uncertainty": u_bar,
            "u_c": u_bar + self.cfg.c0 * self.epochs / t,        # overhead-inclusive U_c (eq 33)
            "time_avg_bandwidth": snap["time_avg_bandwidth"],    # B̄
            "mean_backlog": snap["mean_backlog"],
            "z_final": snap["z"],
            "z_over_t": snap["z_over_t"],                        # budget-compliance witness (Prop 1)
        }
