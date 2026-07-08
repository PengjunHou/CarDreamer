"""Lyapunov queues + per-frame cost-rate for the V2X policy search (paper Sec I.D, Sec IV).

Two queues make the paper's long-term constraints tractable per frame:

* **Per-link backlog** ``Q_m(t)`` (bits) — the transmit-queue proxy for queuing delay (eq 27):
  ``Q_m(t+1) = max{Q_m(t) − R_m·Ts, 0} + 1[m∈S]·L_m``.
* **Virtual budget queue** ``Z(t)`` (bandwidth-ratio units) — the online shadow price of the long-term
  bandwidth budget ``B̄_bgt`` (eq 35): ``Z(t+1) = max{Z(t) + (Σ_m B_m − B̄_bgt), 0}``. Mean-rate stability of
  ``Z`` is equivalent to ``B̄ ≤ B̄_bgt``; ``Z(T)/T → 0`` is the budget-compliance witness (Prop 1).

The per-frame program (P2, eq 41) is scored in **ratio form** (frame cost / frame length ``F(a)``); its
interpretable decomposition (eq 42) is :class:`CostRateBreakdown`. The candidate-independent constant
``−Z·B̄_bgt`` is dropped from the per-frame objective (the paper's "may be dropped in implementation"), so a
local-only chunk has zero bandwidth/net-load rate and its cost rate is the baseline the no-degradation
property compares against.

Pure Python floats (no torch) — fast and trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence

__all__ = [
    "LyapunovConfig",
    "LinkQueue",
    "BudgetVirtualQueue",
    "LyapunovState",
    "CostRateBreakdown",
    "action_cost_rate",
]


@dataclass
class LyapunovConfig:
    """Scalars shared by the queues + cost-rate (paper Sec IV)."""

    lam: float = 1.0                 # Λ  (drift-plus-penalty weight; Prop-1 utility/backlog knob)
    c0: float = 0.5                  # amortized re-planning penalty (uncertainty units)
    budget_bandwidth: float = 0.5    # B̄_bgt  (per-slot allocated-bandwidth budget, ratio units)
    ts_seconds: float = 0.5          # Ts  (slot duration; bits <-> bandwidth bridge for the caller)


# =====================================================================
# Queues
# =====================================================================


@dataclass
class LinkQueue:
    """Per-link (member -> target) backlog ``Q_m`` in bits (paper eq 27)."""

    backlog_bits: float = 0.0

    def update(self, *, service_bits: float, arrival_bits: float) -> float:
        """``Q_m(t+1) = max{Q_m − R_m·Ts, 0} + 1[m∈S]·L_m``. ``service_bits = R_m·Ts``, ``arrival_bits = 1[m∈S]·L_m``."""
        self.backlog_bits = max(self.backlog_bits - float(service_bits), 0.0) + float(arrival_bits)
        return self.backlog_bits


@dataclass
class BudgetVirtualQueue:
    """Virtual queue ``Z`` enforcing the long-term bandwidth budget (paper eq 35)."""

    value: float = 0.0

    def update(self, *, allocated: float, budget: float) -> float:
        """``Z(t+1) = max{Z + (Σ_m B_m − B̄_bgt), 0}``. ``allocated = Σ_m B_m`` (this slot), ``budget = B̄_bgt``."""
        self.value = max(self.value + (float(allocated) - float(budget)), 0.0)
        return self.value


class LyapunovState:
    """Bundles ``{Q_m}`` + ``Z`` and evolves them per installed slot, tracking Prop-1 witnesses.

    Running means: ``mean_backlog`` (time-avg ``Σ_m Q_m``), ``time_avg_bandwidth`` (``B̄``), and
    ``z_over_t`` (``Z(T)/T`` budget-compliance witness).
    """

    def __init__(self, config: LyapunovConfig):
        self.config = config
        self.z = BudgetVirtualQueue()
        self.q: Dict[int, LinkQueue] = {}
        self._slots = 0
        self._sum_total_backlog = 0.0
        self._sum_allocated = 0.0

    def ensure_links(self, member_ids: Iterable[int]) -> None:
        for m in member_ids:
            self.q.setdefault(int(m), LinkQueue())

    def link_backlog(self, m: int) -> float:
        q = self.q.get(int(m))
        return q.backlog_bits if q is not None else 0.0

    def backlogs(self) -> Dict[int, float]:
        return {m: q.backlog_bits for m, q in self.q.items()}

    def total_backlog(self) -> float:
        return float(sum(q.backlog_bits for q in self.q.values()))

    def advance_slot(
        self,
        *,
        per_member_service_bits: Mapping[int, float],
        per_member_arrival_bits: Mapping[int, float],
        allocated_bandwidth: float,
        budget_bandwidth: float | None = None,
    ) -> None:
        """One installed physical slot: update every ``Q_m`` (eq 27) then ``Z`` (eq 35); accrue running means."""
        budget = self.config.budget_bandwidth if budget_bandwidth is None else float(budget_bandwidth)
        members = set(self.q) | set(per_member_service_bits) | set(per_member_arrival_bits)
        self.ensure_links(members)
        for m in members:
            self.q[int(m)].update(
                service_bits=float(per_member_service_bits.get(int(m), 0.0)),
                arrival_bits=float(per_member_arrival_bits.get(int(m), 0.0)),
            )
        self.z.update(allocated=float(allocated_bandwidth), budget=budget)
        self._slots += 1
        self._sum_total_backlog += self.total_backlog()
        self._sum_allocated += float(allocated_bandwidth)

    def snapshot(self) -> Dict[str, float]:
        n = max(self._slots, 1)
        return {
            "z": self.z.value,
            "total_backlog": self.total_backlog(),
            "mean_backlog": self._sum_total_backlog / n,
            "time_avg_bandwidth": self._sum_allocated / n,
            "z_over_t": self.z.value / n,
            "slots": float(self._slots),
        }


# =====================================================================
# Per-frame cost rate (P2 / eq 42)
# =====================================================================


@dataclass(frozen=True)
class CostRateBreakdown:
    """Interpretable per-slot cost-rate decomposition of the (P2) objective (paper eq 42)."""

    planning_rate: float      # Λ c0 / F                              (amortized planning cost)
    uncertainty_rate: float   # (Λ/F) Σ_τ Ũ_τ                          (uncertainty rate)
    bandwidth_rate: float     # (Z/F) Σ_τ Σ_m B_m                      (bandwidth rate, price Z)
    net_load_rate: float      # (1/F) Σ_m Q_m (L̂_m − R̂_m Ts)          (net-load rate, price Q_m)

    @property
    def total(self) -> float:
        return self.planning_rate + self.uncertainty_rate + self.bandwidth_rate + self.net_load_rate

    def as_dict(self) -> Dict[str, float]:
        return {
            "planning_rate": self.planning_rate,
            "uncertainty_rate": self.uncertainty_rate,
            "bandwidth_rate": self.bandwidth_rate,
            "net_load_rate": self.net_load_rate,
            "total": self.total,
        }


def action_cost_rate(
    *,
    horizon_slots: int,
    lam: float,
    c0: float,
    per_slot_uncertainty: Sequence[float],
    per_slot_bandwidth: Sequence[float],
    z: float,
    link_backlogs: Mapping[int, float],
    per_member_arrival_bits: Mapping[int, float],
    per_member_service_bits: Mapping[int, float],
) -> CostRateBreakdown:
    """The (P2) ratio objective for one candidate chunk (paper eq 41-42), decomposed into cost rates.

    ``per_slot_uncertainty`` / ``per_slot_bandwidth`` are per-slot sequences over the chunk horizon
    (``Ũ_{tk+τ}`` and the active ``Σ_m B_m`` at slot ``τ``). ``per_member_*_bits`` are per-member frame
    totals (``L̂_m`` arrivals, ``R̂_m·Ts`` service). The candidate-independent ``−Z·B̄_bgt`` is dropped.
    """
    f = max(int(horizon_slots), 1)
    planning = float(lam) * float(c0) / f
    uncertainty = float(lam) * float(sum(per_slot_uncertainty)) / f
    bandwidth = float(z) * float(sum(per_slot_bandwidth)) / f
    members = set(link_backlogs) | set(per_member_arrival_bits) | set(per_member_service_bits)
    net_load = (
        sum(
            float(link_backlogs.get(m, 0.0))
            * (float(per_member_arrival_bits.get(m, 0.0)) - float(per_member_service_bits.get(m, 0.0)))
            for m in members
        )
        / f
    )
    return CostRateBreakdown(planning, uncertainty, bandwidth, net_load)
