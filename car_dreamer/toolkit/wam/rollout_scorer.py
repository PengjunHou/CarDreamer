"""World-action rollout scorer (V2X paper Sec III): per-slot ``Ũ`` trajectory + predicted comm load.

To score a candidate action chunk before it is executed, the controller rolls out the coupled
(ego, environment, observation) dynamics over the chunk horizon ``F(a)``. For each slot ``τ``:

  (i)   the **uncertainty evaluator** ``U_φ`` (the reused Stage-1 :class:`WAMPerceptionModel`) evaluates the
        current rolled observation to obtain the per-slot semantic uncertainty ``Ũ_{tk+τ}(a) = α·U^mot + (1−α)·U^cov``;
  (ii)  the **risk-aware controller** (Sec I.E) selects ``u*`` and advances the ego along its fixed route;
  (iii) the next observation is predicted under the active sub-action ``a^(j(τ))`` — objects propagate by
        constant velocity (matching :func:`runtime.predict_notable_motion`), members deliver with
        latency/freshness (fed into ``build_stage1_policy_graph``'s ``exp(-γ·latency)`` channel), and the
        predicted payload arrivals ``L̂`` / service ``R̂·Ts`` accumulate for the queue terms of (P2).

Reconciliation of the "model wants a window, rollout is single-step" mismatch: a rolling deque of the last
``history_window`` per-slot graphs is maintained (cold-start = repeat the first graph), so
``WAMPerceptionModel.forward`` sees a normal window and never knows it is a rollout. ``policy_uncertainty`` /
``per_object_trace`` need no ground truth, so ``Ũ`` is computable during the rollout. There is **no Stage-2
UWM**: candidates come from the heuristic enumerator and ``Ũ`` comes entirely from ``U_φ``. If
``perception_model is None`` a visibility-based rule fallback keeps everything runnable with zero checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .action_chunk import ActionChunk
from .bev import BevSpec
from .coverage import CoverageConfig, build_coverage_raster
from .graph import GraphBuildSpec, VehicleNodeInput
from .heads import per_object_trace, policy_uncertainty
from .risk_controller import (
    ControlDecision,
    RiskControlConfig,
    TrackedObject,
    cumulative_lengths,
    point_at_arclength,
    select_accel,
)
from .runtime import ObjectState, WAMPolicy
from .stage1_policy import bev_payload_bytes, build_stage1_policy_graph

__all__ = ["RolloutContext", "RolloutResult", "WorldActionScorer"]

Point2D = Tuple[float, float]
LinkRateFn = Callable[[int, float, float], float]  # (member_id, distance_m, bandwidth_ratio) -> bits/s


@dataclass
class RolloutContext:
    """The ``C^BS`` snapshot at the decision epoch ``t_k`` (offline-buildable, no CARLA)."""

    ego: VehicleNodeInput
    collaborators: Tuple[VehicleNodeInput, ...]
    objects: Tuple[ObjectState, ...]
    route_xy: Tuple[Point2D, ...]
    notable_ids: Tuple[int, ...]
    link_rate_fn: LinkRateFn
    graph_spec: GraphBuildSpec = field(default_factory=GraphBuildSpec)
    bev_spec: BevSpec = field(default_factory=BevSpec)
    ego_v0: float = 0.0
    ego_arc0_m: float = 0.0
    past_route_xy: Tuple[Point2D, ...] = ()
    dt_seconds: float = 0.1
    sensor_period_steps: int = 1
    proc_delay_s: float = 0.05

    @property
    def ego_pose(self) -> Tuple[float, float, float]:
        return (float(self.ego.x), float(self.ego.y), float(self.ego.yaw))


@dataclass
class RolloutResult:
    per_slot_uncertainty: List[float]                      # Ũ_{tk+τ}, τ = 0..F-1  (reference trajectory)
    per_member_predicted_load_bits: Dict[int, float]       # L̂_m over the chunk
    per_member_predicted_service_bits: Dict[int, float]    # R̂_m·Ts over the chunk
    per_slot_bandwidth: List[float]                        # Σ_m B_m active at each slot
    ego_speed_series: List[float]


class WorldActionScorer:
    """Sec III world-action rollout using ``U_φ`` (+ rule fallback), the risk controller, and coverage."""

    def __init__(
        self,
        *,
        perception_model: Optional[torch.nn.Module] = None,
        risk_cfg: Optional[RiskControlConfig] = None,
        coverage_config: Optional[CoverageConfig] = None,
        sigma_scale: float = 4.0,
        alpha: float = 0.5,
        history_window: int = 4,
        freshness_gamma: float = 5.0,
        mass_floor: float = 0.0,
        gate_k: Optional[float] = None,
        ego_fov: float = 150.0,
        ego_sight_range: float = 40.0,
        collaborator_fov: float = 150.0,
        collaborator_sight_range: float = 40.0,
        bev_payload_mode: str = "feature",
        bev_feature_dim: int = 256,
        bev_feature_dtype_bytes: int = 4,
        rule_visible_uncertainty: float = 0.2,
        rule_invisible_uncertainty: float = 2.0,
        bit_scale: Optional[float] = None,
        device: str = "cpu",
    ):
        self.model = perception_model
        self.risk_cfg = risk_cfg or RiskControlConfig()
        # Default corridor reaches beyond a single vehicle's sensor range, so a downstream collaborator can
        # fill the far coverage gap that the ego alone cannot (this is where cooperation earns its cost).
        self.coverage_config = coverage_config or CoverageConfig(
            freshness_gamma=freshness_gamma, future_route_distance_m=60.0
        )
        self.sigma_scale = float(sigma_scale)
        self.alpha = float(alpha)
        self.history_window = max(int(history_window), 1)
        self.freshness_gamma = float(freshness_gamma)
        self.mass_floor = float(mass_floor)
        self.gate_k = gate_k
        self.ego_fov = float(ego_fov)
        self.ego_sight_range = float(ego_sight_range)
        self.collaborator_fov = float(collaborator_fov)
        self.collaborator_sight_range = float(collaborator_sight_range)
        self.bev_payload_mode = str(bev_payload_mode)
        self.bev_feature_dim = int(bev_feature_dim)
        self.bev_feature_dtype_bytes = int(bev_feature_dtype_bytes)
        self.rule_visible_uncertainty = float(rule_visible_uncertainty)
        self.rule_invisible_uncertainty = float(rule_invisible_uncertainty)
        # δ_max-style normalizer (paper Sec I.D): express backlog / arrivals / service in *payload units*
        # (one bev payload ≈ 1.0) so the net-load term is commensurate with the ratio-scale bandwidth term
        # and the [0,1] uncertainty term — otherwise raw bits (~10^4) dominate the (P2) objective.
        self.bit_scale = float(bit_scale) if bit_scale is not None else max(
            8.0 * bev_payload_bytes(BevSpec(), mode=self.bev_payload_mode,
                                    feature_dim=self.bev_feature_dim, feature_dtype_bytes=self.bev_feature_dtype_bytes),
            1.0,
        )
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device).eval()

    # ---- geometry / propagation ----

    @staticmethod
    def _route_yaw_deg(route_xy: Sequence[Point2D], cum: Sequence[float], s: float) -> float:
        n = len(route_xy)
        if n < 2:
            return 0.0
        for i in range(1, n):
            if s <= cum[i]:
                dx = route_xy[i][0] - route_xy[i - 1][0]
                dy = route_xy[i][1] - route_xy[i - 1][1]
                return math.degrees(math.atan2(dy, dx))
        dx = route_xy[-1][0] - route_xy[-2][0]
        dy = route_xy[-1][1] - route_xy[-2][1]
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _propagate_objects(objects: Sequence[ObjectState], elapsed_s: float) -> List[ObjectState]:
        out: List[ObjectState] = []
        for s in objects:
            out.append(replace(s, x=float(s.x) + float(s.vx) * elapsed_s, y=float(s.y) + float(s.vy) * elapsed_s))
        return out

    @staticmethod
    def _propagate_vehicle(v: VehicleNodeInput, elapsed_s: float) -> VehicleNodeInput:
        return replace(v, x=float(v.x) + float(v.vx) * elapsed_s, y=float(v.y) + float(v.vy) * elapsed_s)

    # ---- comm load ----

    def _member_link(
        self, ctx: RolloutContext, ego: VehicleNodeInput, member: VehicleNodeInput, ratio: float
    ) -> Tuple[float, float, float]:
        """Return ``(latency_s, load_bits, rate_bps)`` for one member's bev message under bandwidth ``ratio``."""
        dist = math.hypot(float(member.x) - float(ego.x), float(member.y) - float(ego.y))
        rate_bps = max(float(ctx.link_rate_fn(int(member.actor_id), dist, float(ratio))), 1.0)
        load_bytes = bev_payload_bytes(
            ctx.bev_spec, mode=self.bev_payload_mode,
            feature_dim=self.bev_feature_dim, feature_dtype_bytes=self.bev_feature_dtype_bytes,
        )
        load_bits = 8.0 * float(load_bytes)
        latency_s = float(ctx.proc_delay_s) + load_bits / rate_bps
        return latency_s, load_bits, rate_bps

    # ---- U_phi (or rule) per-slot uncertainty + tracked objects ----

    def _uphi_uncertainty(
        self, window: Sequence[object], objects: Sequence[ObjectState], notable_ids: Sequence[int]
    ) -> Tuple[float, List[TrackedObject]]:
        obj_by_id = {int(o.actor_id): o for o in objects}
        with torch.no_grad():
            out = self.model(list(window))
        u_mot_raw = policy_uncertainty(
            out["notable_prob"], out["traj_log_var"], mass_floor=self.mass_floor, gate_k=self.gate_k
        )
        u_mot = 1.0 - math.exp(-float(u_mot_raw) / max(self.sigma_scale, 1e-6))
        traces = per_object_trace(out["traj_log_var"])  # [Q]
        ids = [int(v) for v in out["object_node_ids"].tolist()]
        notable_prob = out["notable_prob"].tolist()
        tracked: List[TrackedObject] = []
        for i, oid in enumerate(ids):
            o = obj_by_id.get(oid)
            if o is None:
                continue
            tracked.append(
                TrackedObject(
                    xy0=(float(o.x), float(o.y)), vxy=(float(o.vx), float(o.vy)),
                    radius=0.5 * max(float(o.length), float(o.width)),
                    trace_sigma=float(traces[i]) if i < len(traces) else 0.0,
                    weight=float(notable_prob[i]) if i < len(notable_prob) else 0.0,
                )
            )
        return float(u_mot), tracked

    def _rule_uncertainty(
        self, objects: Sequence[ObjectState], notable_ids: Sequence[int]
    ) -> Tuple[float, List[TrackedObject]]:
        notable = set(int(i) for i in notable_ids)
        tracked: List[TrackedObject] = []
        traces: List[float] = []
        for o in objects:
            if int(o.actor_id) not in notable:
                continue
            trace = self.rule_visible_uncertainty if bool(o.visible_to_ego) else self.rule_invisible_uncertainty
            traces.append(trace)
            tracked.append(
                TrackedObject(
                    xy0=(float(o.x), float(o.y)), vxy=(float(o.vx), float(o.vy)),
                    radius=0.5 * max(float(o.length), float(o.width)),
                    trace_sigma=float(trace), weight=1.0,
                )
            )
        u_mot_raw = float(sum(traces) / len(traces)) if traces else 0.0
        u_mot = 1.0 - math.exp(-u_mot_raw / max(self.sigma_scale, 1e-6))
        return float(u_mot), tracked

    # ---- coverage ----

    def _coverage_uncertainty(
        self, ctx: RolloutContext, ego_pose: Tuple[float, float, float],
        members: Sequence[VehicleNodeInput], freshness: Sequence[float],
    ) -> float:
        if not ctx.route_xy:
            return 0.0
        observers = tuple((int(m.actor_id), float(m.x), float(m.y), float(m.yaw)) for m in members)
        _, metrics = build_coverage_raster(
            ego_pose=ego_pose,
            route_xy=ctx.route_xy,
            past_route_xy=ctx.past_route_xy,
            ego_observer=(int(ctx.ego.actor_id), ego_pose[0], ego_pose[1], ego_pose[2]),
            collaborator_observers=observers,
            actor_polygons=None,
            ego_fov=self.ego_fov, ego_sight_range=self.ego_sight_range,
            collaborator_fov=self.collaborator_fov, collaborator_sight_range=self.collaborator_sight_range,
            config=self.coverage_config, spec=ctx.bev_spec,
            collaborator_freshness=tuple(freshness),
        )
        return float(metrics.get("coverage_uncertainty", 0.0))

    # ---- main ----

    def score_chunk(self, ctx: RolloutContext, chunk: ActionChunk) -> RolloutResult:
        dt = float(ctx.dt_seconds)
        ts = max(int(ctx.sensor_period_steps), 1)
        route = list(ctx.route_xy)
        cum = cumulative_lengths(route) if len(route) >= 2 else [0.0]

        v = float(ctx.ego_v0)
        arc = float(ctx.ego_arc0_m)
        collab_by_id = {int(c.actor_id): c for c in ctx.collaborators}

        window: List[object] = []
        per_slot_u: List[float] = []
        per_slot_bw: List[float] = []
        speed_series: List[float] = []
        load_bits: Dict[int, float] = {}
        service_bits: Dict[int, float] = {}

        F = chunk.horizon_slots
        for i in range(F):
            elapsed = i * dt
            _, sub = chunk.sub_action_at(i)

            # ego pose this slot (advanced along the fixed route)
            if len(route) >= 2:
                ex, ey = point_at_arclength(route, cum, arc)
                eyaw = self._route_yaw_deg(route, cum, arc)
            else:
                ex, ey, eyaw = float(ctx.ego.x), float(ctx.ego.y), float(ctx.ego.yaw)
            ego_slot = replace(ctx.ego, x=ex, y=ey, yaw=eyaw)
            ego_pose = (ex, ey, eyaw)

            # propagate objects + members
            objects = self._propagate_objects(ctx.objects, elapsed)
            members = [self._propagate_vehicle(collab_by_id[m], elapsed) for m in sub.selected if m in collab_by_id]

            # per-member latency / freshness / comm load under this sub-action
            latency_by_vehicle: Dict[int, float] = {}
            freshness: List[float] = []
            for m in members:
                ratio = float(sub.bandwidth_by_vehicle.get(int(m.actor_id), 1.0))
                latency_s, l_bits, rate_bps = self._member_link(ctx, ego_slot, m, ratio)
                latency_by_vehicle[int(m.actor_id)] = latency_s
                freshness.append(math.exp(-self.freshness_gamma * latency_s))
                # per-slot service R·dt; per-tick arrival L; both in payload units (÷ bit_scale)
                service_bits[int(m.actor_id)] = service_bits.get(int(m.actor_id), 0.0) + rate_bps * dt / self.bit_scale
                if i % ts == 0:  # a sensor tick: one payload arrives at the link
                    load_bits[int(m.actor_id)] = load_bits.get(int(m.actor_id), 0.0) + l_bits / self.bit_scale

            # per-slot semantic uncertainty Ũ_τ = α U^mot + (1-α) U^cov
            if self.model is not None:
                wam_policy = WAMPolicy(
                    selected_vehicle_ids=tuple(int(m.actor_id) for m in members),
                    modality_by_vehicle={int(m.actor_id): "bev" for m in members},
                    bandwidth_by_vehicle={int(m.actor_id): float(sub.bandwidth_by_vehicle.get(int(m.actor_id), 1.0)) for m in members},
                    frequency_steps=ts, reason="rollout",
                )
                graph = build_stage1_policy_graph(
                    ego=ego_slot, collaborators=members, objects=objects, policy=wam_policy,
                    spec=ctx.graph_spec, notable_ids=ctx.notable_ids, latency_by_vehicle=latency_by_vehicle,
                    bev_spec=ctx.bev_spec, bev_payload_mode=self.bev_payload_mode,
                    bev_feature_dim=self.bev_feature_dim, bev_feature_dtype_bytes=self.bev_feature_dtype_bytes,
                    gamma_freshness=self.freshness_gamma,
                ).to(self.device)
                if not window:  # cold-start: repeat the first graph to fill the window
                    window = [graph] * self.history_window
                else:
                    window.append(graph)
                    window = window[-self.history_window:]
                u_mot, tracked = self._uphi_uncertainty(window, objects, ctx.notable_ids)
            else:
                u_mot, tracked = self._rule_uncertainty(objects, ctx.notable_ids)

            u_cov = self._coverage_uncertainty(ctx, ego_pose, members, freshness)
            u_tau = self.alpha * u_mot + (1.0 - self.alpha) * u_cov
            per_slot_u.append(float(u_tau))
            per_slot_bw.append(float(sub.total_bandwidth()))

            # risk-aware controller advances the ego (endogenous ego motion)
            decision: ControlDecision = select_accel(
                ego_v0=v, route_xy=route, ego_arc0=arc, objects=tracked, cfg=self.risk_cfg
            )
            v = min(max(v + decision.u_star * dt, 0.0), self.risk_cfg.v_max)
            arc += v * dt
            speed_series.append(float(v))

        return RolloutResult(
            per_slot_uncertainty=per_slot_u,
            per_member_predicted_load_bits=load_bits,
            per_member_predicted_service_bits=service_bits,
            per_slot_bandwidth=per_slot_bw,
            ego_speed_series=speed_series,
        )
