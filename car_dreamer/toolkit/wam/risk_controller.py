"""Risk-aware longitudinal speed controller (V2X paper Sec I.E, eqs 29-32).

The target vehicle follows a fixed reference route and regulates only its longitudinal speed, adapting to
the task-oriented semantic uncertainty: an action that lowers uncertainty lets the vehicle proceed faster,
while residual uncertainty forces it to slow down. This closes the loop between perception and motion and
makes the ego state *endogenous* to the action sequence — the chosen speed decides both what the vehicle
observes and where it goes.

For each acceleration candidate ``u ∈ U^ctl``:
  * speed profile ``ν_{t+τ}(u) = clip(ν_t + u·τ·Δt, 0, v_max)``                                    (eq 29)
  * ego pose advances along the route by arc length ``Σ_τ ν·Δt``                                    (eq 29)
  * predictive uncertainty inflates each object's avoidance radius ``r^eff_o = r_o + κ·√TrΣ_o``     (eq 30)
  * ``Risk(u) = Σ_τ Σ_o w_o · exp(−‖p_ego − μ_o‖² / (2(r_v + r^eff_o + ε_sf)²))``                   (eq 31)
The controller trades progress against risk and control effort:
  ``u* = argmax_u ω_ν·ν̄(u) − ω_r·Risk(u) − ω_u·|u|``                                                (eq 32)

Pure numpy; no torch, no CARLA. ``TrΣ_o`` and ``w_o`` come from the Stage-1 ``U_φ`` trajectory head
(``per_object_trace`` and ``notable_prob``); ``μ_o`` is constant-velocity object propagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

__all__ = [
    "RiskControlConfig",
    "TrackedObject",
    "ControlDecision",
    "cumulative_lengths",
    "point_at_arclength",
    "advance_pose_along_route",
    "object_future_positions",
    "risk_of_accel",
    "select_accel",
]

Point2D = Tuple[float, float]


@dataclass
class RiskControlConfig:
    """Risk-aware controller hyperparameters (paper Sec I.E)."""

    accel_set: Tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)  # U^ctl
    v_max: float = 12.0
    dt: float = 0.1
    horizon_steps: int = 20          # τ = 1..H
    kappa: float = 1.0               # κ  (variance -> radius inflation)
    r_vehicle: float = 2.0           # r_v
    eps_safety: float = 1.0          # ε_sf
    w_speed: float = 1.0             # ω_ν
    w_risk: float = 4.0              # ω_r
    w_accel: float = 0.1             # ω_u


@dataclass(frozen=True)
class TrackedObject:
    """One object ``o`` for risk evaluation (from ``U_φ``)."""

    xy0: Point2D                     # current position
    vxy: Point2D = (0.0, 0.0)        # velocity (for μ propagation)
    radius: float = 1.5              # r_o
    trace_sigma: float = 0.0         # TrΣ_o (horizon-averaged predicted variance trace)
    weight: float = 1.0              # w_o (notable_prob)


@dataclass(frozen=True)
class ControlDecision:
    u_star: float
    mean_speed: float
    risk: float
    speed_series: Tuple[float, ...] = field(default_factory=tuple)


# =====================================================================
# Route arc-length geometry
# =====================================================================


def cumulative_lengths(route_xy: Sequence[Point2D]) -> List[float]:
    cum = [0.0]
    for i in range(1, len(route_xy)):
        d = math.hypot(route_xy[i][0] - route_xy[i - 1][0], route_xy[i][1] - route_xy[i - 1][1])
        cum.append(cum[-1] + d)
    return cum


def point_at_arclength(route_xy: Sequence[Point2D], cum: Sequence[float], s: float) -> Point2D:
    """Interpolate the point at arc length ``s`` along the polyline (clamped to the endpoints)."""
    n = len(route_xy)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(route_xy[0][0]), float(route_xy[0][1]))
    total = cum[-1]
    if s <= 0.0:
        return (float(route_xy[0][0]), float(route_xy[0][1]))
    if s >= total:
        return (float(route_xy[-1][0]), float(route_xy[-1][1]))
    for i in range(1, n):
        if s <= cum[i]:
            seg = cum[i] - cum[i - 1]
            t = 0.0 if seg <= 1e-9 else (s - cum[i - 1]) / seg
            return (
                float(route_xy[i - 1][0] + t * (route_xy[i][0] - route_xy[i - 1][0])),
                float(route_xy[i - 1][1] + t * (route_xy[i][1] - route_xy[i - 1][1])),
            )
    return (float(route_xy[-1][0]), float(route_xy[-1][1]))


def advance_pose_along_route(
    route_xy: Sequence[Point2D], arc_start: float, speed_series: Sequence[float], dt: float
) -> List[Point2D]:
    """Ego positions at steps τ=1..H: advance by cumulative arc length ``arc_start + Σ_{i≤τ} ν_i·Δt`` (eq 29)."""
    cum = cumulative_lengths(route_xy)
    out: List[Point2D] = []
    s = float(arc_start)
    for v in speed_series:
        s += float(v) * float(dt)
        out.append(point_at_arclength(route_xy, cum, s))
    return out


def object_future_positions(obj: TrackedObject, horizon_steps: int, dt: float) -> List[Point2D]:
    """Constant-velocity μ propagation ``μ_o(τ) = xy0 + v·τ·Δt`` for τ=1..H."""
    x0, y0 = float(obj.xy0[0]), float(obj.xy0[1])
    vx, vy = float(obj.vxy[0]), float(obj.vxy[1])
    return [(x0 + vx * (tau * dt), y0 + vy * (tau * dt)) for tau in range(1, int(horizon_steps) + 1)]


# =====================================================================
# Risk + control
# =====================================================================


def _speed_series(v0: float, u: float, cfg: RiskControlConfig) -> List[float]:
    """``ν_{t+τ}(u) = clip(ν_t + u·τ·Δt, 0, v_max)`` for τ=1..H (eq 29)."""
    return [
        min(max(float(v0) + float(u) * (tau * cfg.dt), 0.0), cfg.v_max)
        for tau in range(1, int(cfg.horizon_steps) + 1)
    ]


def risk_of_accel(
    u: float, *, ego_v0: float, route_xy: Sequence[Point2D], ego_arc0: float,
    objects: Sequence[TrackedObject], cfg: RiskControlConfig,
) -> Tuple[float, float]:
    """Return ``(risk, mean_speed)`` for acceleration ``u`` (eqs 29-31)."""
    speeds = _speed_series(ego_v0, u, cfg)
    mean_speed = float(sum(speeds) / max(len(speeds), 1))
    if not objects:
        return 0.0, mean_speed
    ego_positions = advance_pose_along_route(route_xy, ego_arc0, speeds, cfg.dt)
    obj_tracks = [(o, object_future_positions(o, cfg.horizon_steps, cfg.dt)) for o in objects]
    risk = 0.0
    for tau in range(len(ego_positions)):
        ex, ey = ego_positions[tau]
        for o, mu in obj_tracks:
            mx, my = mu[tau]
            d2 = (ex - mx) ** 2 + (ey - my) ** 2
            r_eff = float(o.radius) + cfg.kappa * math.sqrt(max(float(o.trace_sigma), 0.0))
            denom = 2.0 * (cfg.r_vehicle + r_eff + cfg.eps_safety) ** 2
            risk += float(o.weight) * math.exp(-d2 / max(denom, 1e-9))
    return float(risk), mean_speed


def select_accel(
    *, ego_v0: float, route_xy: Sequence[Point2D], ego_arc0: float,
    objects: Sequence[TrackedObject], cfg: RiskControlConfig,
) -> ControlDecision:
    """``u* = argmax_u ω_ν·ν̄ − ω_r·Risk − ω_u·|u|`` (eq 32)."""
    best: ControlDecision | None = None
    best_obj = -math.inf
    for u in cfg.accel_set:
        risk, mean_speed = risk_of_accel(
            u, ego_v0=ego_v0, route_xy=route_xy, ego_arc0=ego_arc0, objects=objects, cfg=cfg
        )
        objective = cfg.w_speed * mean_speed - cfg.w_risk * risk - cfg.w_accel * abs(float(u))
        if objective > best_obj:
            best_obj = objective
            best = ControlDecision(
                u_star=float(u), mean_speed=mean_speed, risk=risk,
                speed_series=tuple(_speed_series(ego_v0, u, cfg)),
            )
    assert best is not None  # accel_set is non-empty
    return best
