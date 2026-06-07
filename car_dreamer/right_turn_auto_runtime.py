from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import carla
import numpy as np
import torch
from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

from .toolkit import (
    NetResource,
    Observer,
    PayloadEncoder,
    PayloadSelectorDecision,
    V2VMessage,
    _dist_m,
    _tx_bytes_for_latency,
    canonicalize_payload_type,
    get_vehicle_pos,
)
from .toolkit.policy import (
    BANDWIDTH_BUDGET,
    CollaborationAction,
    PolicySelectorDecision,
    SceneSummary,
    VehicleInfo,
    get_policy,
)


GROUP_ID = 0
RECEIVED_BUFFER_SIZE = 256
RUNTIME_LOGGER = get_runtime_logger("car_dreamer.runtime")


class RightTurnAutoRuntimeMixin:
    def _reset_policy_runtime_state(self) -> None:
        self._policy_current_action = CollaborationAction()
        self._policy_last_action_step = -1
        self._policy_send_credit = {}
        self._policy_send_decision = {}
        self._policy_last_comm_step = -1
        self._policy_current_decision = PolicySelectorDecision(policy_id=str(getattr(self, "_collaboration_policy_id", "")))
        self._policy_prev_comm_summary = {
            "attempted_message_count": 0.0,
            "dropped_message_count": 0.0,
            "drop_ratio_prev_round": 0.0,
        }
        self._payload_decisions_by_sender = {}
        self._payload_last_decision_step = -1

    def register_collaboration_policy(self, policy) -> None:
        registry = getattr(self, "_policy_registry", None)
        if registry is None:
            raise RuntimeError("Policy registry is not initialized.")
        registry.register(policy)

    def set_policy_override(self, policy_id: str | None) -> None:
        policy_id = str(policy_id or "").strip()
        if policy_id:
            getattr(self, "_policy_registry").get(policy_id)
        self._policy_override_id = policy_id
        self._policy_last_action_step = -1
        self._policy_last_comm_step = -1

    def clear_policy_override(self) -> None:
        self.set_policy_override(None)

    def set_policy_selector(self, selector) -> None:
        self._policy_selector = selector
        self._policy_last_action_step = -1

    def register_payload_encoder(self, payload_type: str, encoder: PayloadEncoder) -> None:
        registry = getattr(self, "_payload_registry", None)
        if registry is None:
            raise RuntimeError("Payload registry is not initialized.")
        registry.register(payload_type, encoder, default=True)

    def set_payload_override(self, payload_type: str | None) -> None:
        payload_type = str(payload_type or "").strip()
        if payload_type:
            payload_type = canonicalize_payload_type(payload_type)
            getattr(self, "_payload_registry").get(payload_type)
        self._payload_override = payload_type
        self._payload_last_decision_step = -1

    def clear_payload_override(self) -> None:
        self.set_payload_override(None)

    def set_payload_selector(self, selector) -> None:
        self._payload_selector = selector
        self._payload_last_decision_step = -1

    def _get_runtime_debug_interval(self) -> int:
        runtime_cfg = get_runtime_logging_config()
        base_interval = int(runtime_cfg["step_debug_interval"])
        override = int(getattr(self, "_runtime_step_debug_interval_override", 0) or 0)
        if override > 0:
            return max(base_interval, override)
        return base_interval

    def _cache_actor(self, actor: Optional[carla.Actor]) -> None:
        if actor is not None:
            self._actor_cache[int(actor.id)] = actor

    def _refresh_actor_cache(self) -> None:
        self._actor_cache = {}
        self._cache_actor(getattr(self, "ego", None))
        for actor in self.group_vehs:
            self._cache_actor(actor)

    def _get_group_member_actor(self, actor_id: int) -> Optional[carla.Actor]:
        actor = self._actor_cache.get(int(actor_id))
        if actor is not None:
            return actor
        if getattr(self, "ego", None) is not None and int(self.ego.id) == int(actor_id):
            self._cache_actor(self.ego)
            return self.ego
        for actor in self.group_vehs:
            if int(actor.id) == int(actor_id):
                self._cache_actor(actor)
                return actor
        actor = self._world._world.get_actor(int(actor_id))
        if actor is not None:
            self._cache_actor(actor)
        return actor

    def _reset_group_runtime_state(self) -> None:
        self.group_vehs = []
        self.background_vehs = []
        self.pedestrians = []
        self.groups = {}
        self._prev_action = None
        self._actor_cache = {}
        self._in_flight = []
        self._received = defaultdict(lambda: deque(maxlen=RECEIVED_BUFFER_SIZE))
        self._veh_net_res = {}
        self._comm_link_analysis_by_sender = {}
        self._comm_step_summary = {}
        self._comm_step_summary_step = -1
        self._reset_policy_runtime_state()
        RUNTIME_LOGGER.debug("Group runtime state reset.")

    def _get_active_policy_id(self) -> str:
        decision = getattr(self, "_policy_current_decision", None)
        if decision is not None and str(getattr(decision, "policy_id", "")).strip():
            return str(decision.policy_id)
        return str(getattr(self, "_collaboration_policy_id", "")).strip()

    def _get_current_policy_decision(self) -> Dict[str, Any]:
        decision = getattr(self, "_policy_current_decision", None)
        if decision is None:
            return {
                "policy_id": self._get_active_policy_id(),
                "reason": "",
                "overridden": False,
            }
        return {
            "policy_id": str(getattr(decision, "policy_id", "")),
            "reason": str(getattr(decision, "reason", "")),
            "overridden": bool(getattr(decision, "overridden", False)),
        }

    def _ensure_comm_step_summary(self) -> Dict[str, float]:
        current_step = int(getattr(self, "_time_step", 0))
        if int(getattr(self, "_comm_step_summary_step", -1)) != current_step:
            self._comm_step_summary = {
                "attempted_message_count": 0.0,
                "dropped_message_count": 0.0,
                "enqueued_message_count": 0.0,
                "dropped_capacity_exceeded_count": 0.0,
            }
            self._comm_step_summary_step = current_step
        return self._comm_step_summary

    def _record_comm_link_analysis(
        self,
        sender_id: int,
        receiver_id: int,
        payload_bytes: int,
        analysis: Any,
        *,
        dropped: bool,
        drop_reason: str = "",
    ) -> None:
        if dropped and not bool(getattr(self, "_log_dropped_messages", True)):
            return
        if analysis is None:
            return
        self._comm_link_analysis_by_sender[int(sender_id)] = {
            "sender_id": int(sender_id),
            "receiver_id": int(receiver_id),
            "payload_bytes": float(payload_bytes),
            "required_load_bps": float(getattr(analysis, "required_load_bps", 0.0)),
            "link_rate_bps": float(getattr(analysis, "link_rate_bps", 0.0)),
            "shannon_bps": float(getattr(analysis, "shannon_bps", 0.0)),
            "comm_bandwidth_hz": float(getattr(analysis, "bandwidth_hz", 0.0)),
            "comm_snr_db": float(getattr(analysis, "snr_db", 0.0)),
            "comm_feasible": 1.0 if bool(getattr(analysis, "feasible", False)) else 0.0,
            "dropped_capacity_exceeded": 1.0 if bool(dropped) else 0.0,
            "drop_reason_capacity_exceeded": 1.0 if str(drop_reason) == "capacity_exceeded" else 0.0,
            "analysis_latency_s": float(getattr(analysis, "latency_s", 0.0)),
            "analysis_distance_m": float(getattr(analysis, "distance_m", 0.0)),
        }

    def _get_latest_comm_link_analysis(self, vehicle_id: int) -> Dict[str, float]:
        stats = self._comm_link_analysis_by_sender.get(int(vehicle_id), {})
        return {str(key): float(value) for key, value in stats.items()}

    def _get_current_comm_step_summary(self) -> Dict[str, float]:
        summary = self._ensure_comm_step_summary()
        return {str(key): float(value) for key, value in summary.items()}

    def _build_scene_summary(self, infos: List[VehicleInfo]) -> SceneSummary:
        if not infos:
            return SceneSummary(
                num_candidates=0,
                avg_link_latency_s=float(getattr(self, "_policy_prev_comm_summary", {}).get("avg_link_latency_s", 0.0)),
                drop_ratio_prev_round=float(getattr(self, "_policy_prev_comm_summary", {}).get("drop_ratio_prev_round", 0.0)),
            )
        collabs = [float(info.collaboration_score) for info in infos]
        dists = [float(info.distance_m) for info in infos]
        comm_stats = self._comm_link_analysis_by_sender or {}
        latencies = [
            float(stats.get("analysis_latency_s", stats.get("latest_latency_s", 0.0)))
            for stats in comm_stats.values()
            if isinstance(stats, dict)
        ]
        prev_summary = getattr(self, "_policy_prev_comm_summary", {})
        return SceneSummary(
            num_candidates=len(infos),
            max_collaboration_score=max(collabs) if collabs else 0.0,
            mean_collaboration_score=sum(collabs) / len(collabs) if collabs else 0.0,
            min_distance_m=min(dists) if dists else 0.0,
            mean_distance_m=sum(dists) / len(dists) if dists else 0.0,
            avg_link_latency_s=(sum(latencies) / len(latencies)) if latencies else float(prev_summary.get("avg_link_latency_s", 0.0)),
            drop_ratio_prev_round=float(prev_summary.get("drop_ratio_prev_round", 0.0)),
        )

    def _select_active_policy(self, infos: List[VehicleInfo]) -> Tuple[Any, PolicySelectorDecision]:
        registry = getattr(self, "_policy_registry", None)
        if registry is None:
            policy = get_policy(str(getattr(self, "_collaboration_policy_id", "P3")))
            return policy, PolicySelectorDecision(policy_id=policy.policy_id)
        scene_summary = self._build_scene_summary(infos)
        override_id = str(getattr(self, "_policy_override_id", "")).strip()
        if override_id:
            policy = registry.get(override_id)
            return policy, PolicySelectorDecision(
                policy_id=override_id,
                reason="policy_override",
                overridden=True,
                scene_summary=scene_summary,
            )
        if str(getattr(self, "_policy_mode", "fixed")).lower() != "adaptive":
            policy_id = str(getattr(self, "_collaboration_policy_id", "P3"))
            policy = registry.get(policy_id)
            return policy, PolicySelectorDecision(
                policy_id=policy_id,
                reason="fixed_policy_mode",
                overridden=False,
                scene_summary=scene_summary,
            )
        selector = getattr(self, "_policy_selector", None)
        if selector is None:
            policy_id = str(getattr(self, "_collaboration_policy_id", "P3"))
            policy = registry.get(policy_id)
            return policy, PolicySelectorDecision(
                policy_id=policy_id,
                reason="missing_policy_selector",
                overridden=False,
                scene_summary=scene_summary,
            )
        decision = selector(scene_summary, registry=registry)
        policy = registry.get(str(decision.policy_id))
        return policy, decision

    def _destroy_group_observers(self) -> None:
        for observer in self._other_observers.values():
            observer.destroy()
        self._other_observers = {}
        self.group_obs = {}

    def _configure_traffic_lights(self) -> None:
        for tl in self._world.carla_actors(actor_type="traffic_light"):
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(9999)
            tl.set_red_time(0)
            tl.set_yellow_time(0)

    def _create_group_observer(self, vehicle: carla.Actor) -> None:
        group_observation = self._config.group_observation
        observer = Observer(self._world, group_observation)
        self._other_observers[int(vehicle.id)] = observer
        observer.reset(vehicle)
        self.group_obs[int(vehicle.id)], _ = observer.get_observation(self.get_state())

    def generate_group_vehicles(self):
        self.groups.setdefault(GROUP_ID, set())
        self.groups[GROUP_ID].add(int(self.ego.id))
        spawn_points = self._config.group_spawn_points
        assert spawn_points is not None and len(spawn_points) >= self.num_group_vehs, (
            "Not enough spawn points for the number of group vehicles"
        )
        for spawn_point in spawn_points[: self.num_group_vehs]:
            transform = carla.Transform(
                carla.Location(*spawn_point[:3]),
                carla.Rotation(yaw=spawn_point[3]),
            )
            vehicle = self._world.spawn_actor(transform=transform)
            self._create_group_observer(vehicle)
            self.group_vehs.append(vehicle)
            self.groups[GROUP_ID].add(int(vehicle.id))
            self._cache_actor(vehicle)
        RUNTIME_LOGGER.info(
            "Generated group vehicles count=%d ids=%s group_members=%s",
            len(self.group_vehs),
            [int(vehicle.id) for vehicle in self.group_vehs],
            sorted(self.groups.get(GROUP_ID, set())),
        )

    def generate_background_actors(self) -> None:
        """Spawn non-V2V background traffic and pedestrians to enrich the scene.

        These actors are *not* part of the cooperative group (``group_vehs``) and
        never participate in V2V messaging. Their purpose is to occlude the ego's
        right-turn conflict region and add dynamic hazards, so the cooperative
        perception signal has something to recover (this is what makes V2V matter
        for the downstream L3 fusion). They are managed by the WorldManager and are
        auto-destroyed on the next reset.
        """
        self.background_vehs = []
        self.pedestrians = []

        num_bg = int(getattr(self._config, "num_background_vehs", 0))
        if num_bg > 0:
            transforms = self._select_background_vehicle_transforms(num_bg)
            try:
                actors = self._world.spawn_auto_actors(num_bg, transforms=transforms)
            except Exception:
                RUNTIME_LOGGER.exception("Failed to spawn background vehicles.")
                actors = []
            self.background_vehs = list(actors)
            for actor in self.background_vehs:
                self._cache_actor(actor)

        num_ped = int(getattr(self._config, "num_pedestrians", 0))
        if num_ped > 0:
            try:
                walkers = self._world.spawn_walkers(
                    num_ped,
                    spawn_transforms=self._select_pedestrian_transforms(num_ped),
                    center=self._scene_center_location(),
                    radius=float(getattr(self._config, "background_radius_m", 60.0)),
                )
            except Exception:
                RUNTIME_LOGGER.exception("Failed to spawn pedestrians.")
                walkers = []
            self.pedestrians = [walker for walker, _ in walkers]

        RUNTIME_LOGGER.info(
            "Generated background actors vehicles=%d pedestrians=%d",
            len(self.background_vehs),
            len(self.pedestrians),
        )

    def _scene_center_location(self) -> carla.Location:
        center = getattr(self._config, "scene_center", None)
        if center is not None and len(center) >= 2:
            return carla.Location(x=float(center[0]), y=float(center[1]), z=0.1)
        points = self._config.group_spawn_points or []
        if points:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            return carla.Location(x=sum(xs) / len(xs), y=sum(ys) / len(ys), z=0.1)
        return carla.Location(0.0, 0.0, 0.1)

    def _reserved_scene_locations(self) -> List[carla.Location]:
        """Spawn points of ego and group vehicles, to keep background clear of them."""
        locs: List[carla.Location] = []
        for point in (self._config.group_spawn_points or []):
            locs.append(carla.Location(x=float(point[0]), y=float(point[1]), z=0.1))
        ego_end = getattr(self._config, "lane_end_point", None)
        if ego_end is not None and len(ego_end) >= 2:
            locs.append(carla.Location(x=float(ego_end[0]), y=float(ego_end[1]), z=0.1))
        return locs

    def _select_background_vehicle_transforms(self, n: int) -> List[carla.Transform]:
        explicit = getattr(self._config, "background_veh_spawn_points", None) or []
        transforms = [
            carla.Transform(
                carla.Location(*point[:3]),
                carla.Rotation(yaw=float(point[3]) if len(point) > 3 else 0.0),
            )
            for point in explicit
        ]
        if len(transforms) >= n:
            return transforms[:n]

        center = self._scene_center_location()
        radius = float(getattr(self._config, "background_radius_m", 60.0))
        reserved = self._reserved_scene_locations()
        min_clearance_m = 4.0
        candidates = [
            transform
            for transform in self._world.get_spawn_points()
            if transform.location.distance(center) <= radius
            and all(transform.location.distance(loc) >= min_clearance_m for loc in reserved)
        ]
        np.random.shuffle(candidates)
        transforms.extend(candidates)
        return transforms[:n]

    def _select_pedestrian_transforms(self, n: int) -> List[carla.Transform]:
        explicit = getattr(self._config, "pedestrian_spawn_points", None) or []
        transforms = [carla.Transform(carla.Location(*point[:3])) for point in explicit]
        # Any shortfall is filled by navigation-mesh sampling inside spawn_walkers.
        return transforms[:n]

    def _update_group_observations(self) -> None:
        for actor in self.group_vehs:
            observer = self._other_observers.get(int(actor.id))
            if observer is not None:
                self.group_obs[int(actor.id)], _ = observer.get_observation(self.get_state())

    def _setup_basic_agent(self) -> None:
        from .toolkit.planner.agents.navigation.basic_agent import BasicAgent

        self.ego_end = self._config.lane_end_point
        ego_transform = carla.Transform(
            carla.Location(*self.ego_end[:3]),
            carla.Rotation(yaw=self.ego_end[3]),
        )
        self.agent = BasicAgent(self.ego)
        self.agent.set_destination(ego_transform.location)
        self._cache_actor(self.ego)

    def _build_policy_vehicle_infos(self) -> List[VehicleInfo]:
        if getattr(self, "ego", None) is None:
            return []
        ego_tf = self.ego.get_transform()
        infos: List[VehicleInfo] = []
        for actor in self.group_vehs:
            tf = actor.get_transform()
            dx = float(tf.location.x) - float(ego_tf.location.x)
            dy = float(tf.location.y) - float(ego_tf.location.y)
            distance_m = math.sqrt(dx * dx + dy * dy)
            latest_stats = self._get_latest_comm_link_analysis(int(actor.id))
            feasible = float(latest_stats.get("comm_feasible", 1.0))
            latency_s = max(float(latest_stats.get("analysis_latency_s", 0.0)), 0.0)
            distance_score = math.exp(-0.03 * max(distance_m, 0.0))
            latency_score = math.exp(-2.0 * latency_s)
            collaboration_score = float(distance_score * latency_score * max(feasible, 0.0))
            infos.append(
                VehicleInfo(
                    vehicle_id=int(actor.id),
                    collaboration_score=collaboration_score,
                    distance_m=float(distance_m),
                )
            )
        return infos

    def _compute_current_policy_action(self) -> CollaborationAction:
        if getattr(self, "_policy_last_action_step", -1) == int(self._time_step):
            return self._policy_current_action
        infos = self._build_policy_vehicle_infos()
        policy, decision = self._select_active_policy(infos)
        previous_policy_id = self._get_active_policy_id()
        self._policy_current_decision = decision
        self._collaboration_policy_id = str(decision.policy_id)
        self._collaboration_policy = policy
        if previous_policy_id and previous_policy_id != self._collaboration_policy_id:
            self._policy_send_credit = {}
            self._policy_send_decision = {}
        if policy is None:
            self._policy_current_action = CollaborationAction()
            self._policy_last_action_step = int(self._time_step)
            return self._policy_current_action
        rng = random.Random(int(getattr(self, "_collaboration_policy_seed", 0)) + int(self._time_step))
        self._policy_current_action = self._normalize_runtime_policy_action(policy(infos, rng=rng))
        self._policy_last_action_step = int(self._time_step)
        return self._policy_current_action

    def _get_policy_action_value(self, vehicle_id: int) -> Dict[str, float]:
        action = self._compute_current_policy_action()
        vid = int(vehicle_id)
        decision = self._get_current_policy_decision()
        return {
            "policy_id": str(decision.get("policy_id", "")),
            "policy_selector_reason": str(decision.get("reason", "")),
            "policy_overridden": 1.0 if bool(decision.get("overridden", False)) else 0.0,
            "alpha": float(action.alpha.get(vid, 0.0)),
            "nu": float(action.nu.get(vid, 0.0)),
            "bandwidth": float(action.bandwidth.get(vid, 0.0)),
        }

    def _normalize_runtime_policy_action(self, action: CollaborationAction) -> CollaborationAction:
        normalized = action.normalized_bandwidth(budget=BANDWIDTH_BUDGET)
        active_ids = normalized.active_ids()
        original_total = sum(max(float(action.bandwidth.get(int(vid), 0.0)), 0.0) for vid in active_ids)
        normalized_total = sum(max(float(normalized.bandwidth.get(int(vid), 0.0)), 0.0) for vid in active_ids)
        if (
            original_total > BANDWIDTH_BUDGET + 1e-9
            and should_log_periodic(
                int(self._time_step),
                int(self._get_runtime_debug_interval()),
                logger=RUNTIME_LOGGER,
            )
        ):
            RUNTIME_LOGGER.debug(
                "Normalized runtime bandwidth allocation step=%d total_before=%.6f total_after=%.6f active_ids=%s",
                int(self._time_step),
                float(original_total),
                float(normalized_total),
                sorted(int(vid) for vid in active_ids),
            )
        return normalized

    def _advance_policy_send_schedule(self) -> Dict[int, bool]:
        if getattr(self, "_policy_last_comm_step", -1) == int(self._time_step):
            return dict(self._policy_send_decision)
        action = self._compute_current_policy_action()
        decisions: Dict[int, bool] = {}
        for vid, alpha in action.alpha.items():
            vehicle_id = int(vid)
            if float(alpha) <= 0.5:
                self._policy_send_credit[vehicle_id] = 0.0
                decisions[vehicle_id] = False
                continue
            nu = max(float(action.nu.get(vehicle_id, 0.0)), 0.0)
            credit = float(self._policy_send_credit.get(vehicle_id, 0.0)) + nu
            should_send = credit >= 1.0 - 1e-6
            if should_send:
                credit = max(credit - 1.0, 0.0)
            self._policy_send_credit[vehicle_id] = credit
            decisions[vehicle_id] = should_send
        self._policy_send_decision = decisions
        self._policy_last_comm_step = int(self._time_step)
        return dict(decisions)

    def _select_payload_decision(
        self,
        *,
        sender_id: int,
        bandwidth: float,
        distance_m: float,
    ) -> PayloadSelectorDecision:
        registry = getattr(self, "_payload_registry", None)
        if registry is None:
            return PayloadSelectorDecision(
                payload_type="object_list",
                payload_encoder_id="object_list_v1",
                reason="missing_registry",
            )
        override = str(getattr(self, "_payload_override", "")).strip()
        if override:
            encoder_id = registry.default_encoder_id(override)
            return PayloadSelectorDecision(
                payload_type=override,
                payload_encoder_id=encoder_id,
                reason="payload_override",
                overridden=True,
            )
        selector = getattr(self, "_payload_selector", None)
        if selector is None:
            return PayloadSelectorDecision(
                payload_type="object_list",
                payload_encoder_id=registry.default_encoder_id("object_list"),
                reason="missing_payload_selector",
                overridden=False,
            )
        latest_comm_stats = self._get_latest_comm_link_analysis(int(sender_id))
        return selector(
            sender_id=int(sender_id),
            bandwidth=float(bandwidth),
            distance_m=float(distance_m),
            latest_comm_stats=latest_comm_stats,
            registry=registry,
            enabled_types=list(getattr(self, "_payload_enabled_types", ["object_list"])),
        )

    def _compute_current_payload_decisions(self) -> Dict[int, PayloadSelectorDecision]:
        if getattr(self, "_payload_last_decision_step", -1) == int(self._time_step):
            return dict(self._payload_decisions_by_sender)
        action = self._compute_current_policy_action()
        decisions: Dict[int, PayloadSelectorDecision] = {}
        actor_map = self._build_group_actor_map()
        ego_id = int(getattr(self.ego, "id", -1)) if getattr(self, "ego", None) is not None else -1
        ego_actor = actor_map.get(ego_id)
        for actor in self.group_vehs:
            sender_id = int(actor.id)
            if sender_id == ego_id:
                continue
            bandwidth = float(action.bandwidth.get(sender_id, 0.0))
            distance_m = _dist_m(actor, ego_actor) if ego_actor is not None else 0.0
            decisions[sender_id] = self._select_payload_decision(
                sender_id=sender_id,
                bandwidth=bandwidth,
                distance_m=distance_m,
            )
        self._payload_decisions_by_sender = decisions
        self._payload_last_decision_step = int(self._time_step)
        return dict(decisions)

    def _get_payload_action_value(self, vehicle_id: int) -> Dict[str, Any]:
        decisions = self._compute_current_payload_decisions()
        decision = decisions.get(int(vehicle_id))
        if decision is None:
            registry = getattr(self, "_payload_registry", None)
            encoder_id = "object_list_v1"
            if registry is not None:
                encoder_id = registry.default_encoder_id("object_list")
            decision = PayloadSelectorDecision(
                payload_type="object_list",
                payload_encoder_id=encoder_id,
                reason="default_object_list",
                overridden=False,
            )
        return {
            "payload_type": str(decision.payload_type),
            "payload_encoder_id": str(decision.payload_encoder_id),
            "payload_selector_reason": str(decision.reason),
            "payload_overridden": 1.0 if bool(decision.overridden) else 0.0,
        }

    def _scale_net_resource_for_bandwidth(self, base: NetResource, bandwidth: float) -> NetResource:
        scale = max(float(bandwidth), float(getattr(self, "_collaboration_bandwidth_floor", 0.1)))
        return NetResource(
            bandwidth_hz=float(base.bandwidth_hz) * scale,
            tx_power_dbm=float(base.tx_power_dbm),
            noise_figure_db=float(base.noise_figure_db),
            carrier_freq_hz=float(base.carrier_freq_hz),
        )

    # =========================================================
    # Communication helpers
    # =========================================================

    def _make_payload(
        self,
        sender: carla.Actor,
        payload_decision: PayloadSelectorDecision,
    ) -> Dict[str, Any]:
        obs = self.obs if int(sender.id) == int(self.ego.id) else self.group_obs.get(int(sender.id), {})
        payload: Dict[str, Any] = {}
        registry = getattr(self, "_payload_registry", None)
        if registry is None:
            raise RuntimeError("Payload registry is not initialized.")
        encoder = registry.get(payload_decision.payload_type, payload_decision.payload_encoder_id)
        encoding = encoder.encode(
            sender,
            obs,
            self.feature_size,
            jpeg_quality=int(getattr(self, "_payload_image_jpeg_quality", 80)),
        )
        payload = encoding.to_payload_dict()
        payload["selector_metadata"] = {
            "payload_type": str(payload_decision.payload_type),
            "payload_encoder_id": str(payload_decision.payload_encoder_id),
            "reason": str(payload_decision.reason),
            "overridden": bool(payload_decision.overridden),
        }

        tf = sender.get_transform()
        vel = sender.get_velocity()
        payload.update(
            {
                "pose": {
                    "x": float(tf.location.x),
                    "y": float(tf.location.y),
                    "yaw": float(tf.rotation.yaw),
                },
                "vel": {"vx": float(vel.x), "vy": float(vel.y)},
                "sender_id": int(sender.id),
                "sensor_name": "cam0",
            }
        )
        return payload

    def _build_group_actor_map(self) -> Dict[int, carla.Actor]:
        actor_map: Dict[int, carla.Actor] = {}
        if getattr(self, "ego", None) is not None:
            actor_map[int(self.ego.id)] = self.ego
        for actor in self.group_vehs:
            actor_map[int(actor.id)] = actor
        self._actor_cache.update(actor_map)
        return actor_map

    def _compute_group_degrees(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        out_deg: Dict[int, int] = {}
        in_deg: Dict[int, int] = {}
        for members in self.groups.values():
            member_ids = list(members)
            degree = max(len(member_ids) - 1, 0)
            for sender_id in member_ids:
                out_deg[int(sender_id)] = degree
            for receiver_id in member_ids:
                in_deg[int(receiver_id)] = degree
        return out_deg, in_deg

    def _enqueue_message(
        self,
        group_id: int,
        sender_id: int,
        receiver_id: int,
        payload: Dict[str, Any],
        payload_bytes: int,
        latency_s: float,
        distance_m: float,
        fixed_dt: float,
    ) -> None:
        delay_steps = max(int(math.ceil(latency_s / max(fixed_dt, 1e-6))), 0)
        deliver_step = int(self._time_step + delay_steps)
        self._in_flight.append(
            V2VMessage(
                sender_id=int(sender_id),
                receiver_id=int(receiver_id),
                group_id=int(group_id),
                payload=payload,
                payload_bytes=int(payload_bytes),
                created_step=int(self._time_step),
                deliver_step=deliver_step,
                latency_s=float(latency_s),
                distance_m=float(distance_m),
            )
        )

    def _run_group_communication(self) -> None:
        if not self.groups:
            return
        step_summary = self._ensure_comm_step_summary()
        actor_map = self._build_group_actor_map()
        fixed_dt = float(self._world._settings.fixed_delta_seconds)
        ego_id = int(self.ego.id)
        current_action = self._compute_current_policy_action()
        policy_decision = self._get_current_policy_decision()
        send_decisions = self._advance_policy_send_schedule()
        payload_decisions = self._compute_current_payload_decisions()
        enqueued_count = 0
        sender_ids = set()
        total_payload_bytes = 0

        for group_id, members in self.groups.items():
            member_ids = list(members)
            for sender_id in member_ids:
                if int(sender_id) == ego_id:
                    continue
                sender = actor_map.get(int(sender_id))
                if sender is None:
                    continue
                alpha = float(current_action.alpha.get(int(sender_id), 0.0))
                if alpha <= 0.5 or not bool(send_decisions.get(int(sender_id), False)):
                    continue
                sender_ids.add(int(sender_id))
                receiver = actor_map.get(ego_id)
                distance_to_ego = _dist_m(sender, receiver) if receiver is not None else 0.0
                payload_decision = payload_decisions.get(
                    int(sender_id),
                    self._select_payload_decision(
                        sender_id=int(sender_id),
                        bandwidth=float(current_action.bandwidth.get(int(sender_id), 0.0)),
                        distance_m=distance_to_ego,
                    ),
                )
                payload = self._make_payload(sender, payload_decision)
                payload["policy_action"] = {
                    "policy_id": str(policy_decision.get("policy_id", getattr(self, "_collaboration_policy_id", ""))),
                    "policy_selector_reason": str(policy_decision.get("reason", "")),
                    "policy_overridden": bool(policy_decision.get("overridden", False)),
                    "alpha": alpha,
                    "nu": float(current_action.nu.get(int(sender_id), 0.0)),
                    "bandwidth": float(current_action.bandwidth.get(int(sender_id), 0.0)),
                    "beta": str(payload_decision.payload_type),
                    "payload_type": str(payload_decision.payload_type),
                    "payload_encoder_id": str(payload_decision.payload_encoder_id),
                }
                payload_bytes = _tx_bytes_for_latency(
                    payload,
                    overhead_bytes=getattr(self.latency_model, "overhead_bytes", 64),
                )
                for receiver_id in member_ids:
                    if int(receiver_id) != ego_id or int(receiver_id) == int(sender_id):
                        continue
                    receiver = actor_map.get(int(receiver_id))
                    if receiver is None:
                        continue
                    bandwidth = float(current_action.bandwidth.get(int(sender_id), 0.0))
                    sender_res = self._scale_net_resource_for_bandwidth(
                        self._veh_net_res.get(int(sender_id), self._default_net_res),
                        bandwidth,
                    )
                    receiver_res = self._scale_net_resource_for_bandwidth(
                        self._veh_net_res.get(int(receiver_id), self._default_net_res),
                        bandwidth,
                    )
                    analysis_fn = getattr(self.latency_model, "analyze_transmission", None)
                    if callable(analysis_fn):
                        analysis = analysis_fn(
                            sender=sender,
                            receiver=receiver,
                            payload_size_bytes=payload_bytes,
                            sender_res=sender_res,
                            receiver_res=receiver_res,
                            out_degree=1,
                            in_degree=1,
                            alpha=alpha,
                            nu=float(current_action.nu.get(int(sender_id), 0.0)),
                            fixed_dt=fixed_dt,
                        )
                        latency_s = float(getattr(analysis, "latency_s", 0.0))
                    else:
                        analysis = None
                        latency_s = self.latency_model.compute_latency_s(
                            sender=sender,
                            receiver=receiver,
                            payload_size_bytes=payload_bytes,
                            sender_res=sender_res,
                            receiver_res=receiver_res,
                            out_degree=1,
                            in_degree=1,
                            alpha=alpha,
                            nu=float(current_action.nu.get(int(sender_id), 0.0)),
                            fixed_dt=fixed_dt,
                        )
                    step_summary["attempted_message_count"] += 1.0
                    if (
                        bool(getattr(self, "_drop_on_capacity_exceeded", False))
                        and analysis is not None
                        and not bool(getattr(analysis, "feasible", True))
                    ):
                        step_summary["dropped_message_count"] += 1.0
                        step_summary["dropped_capacity_exceeded_count"] += 1.0
                        self._record_comm_link_analysis(
                            int(sender_id),
                            int(receiver_id),
                            payload_bytes,
                            analysis,
                            dropped=True,
                            drop_reason="capacity_exceeded",
                        )
                        if should_log_periodic(
                            int(self._time_step),
                            int(self._get_runtime_debug_interval()),
                            logger=RUNTIME_LOGGER,
                        ):
                            RUNTIME_LOGGER.debug(
                                "Dropped message step=%d sender=%d receiver=%d payload_bytes=%d r_req=%.3f R_link=%.3f",
                                self._time_step,
                                int(sender_id),
                                int(receiver_id),
                                int(payload_bytes),
                                float(getattr(analysis, "required_load_bps", 0.0)),
                                float(getattr(analysis, "link_rate_bps", 0.0)),
                            )
                        continue
                    self._record_comm_link_analysis(
                        int(sender_id),
                        int(receiver_id),
                        payload_bytes,
                        analysis,
                        dropped=False,
                    )
                    self._enqueue_message(
                        group_id=group_id,
                        sender_id=int(sender_id),
                        receiver_id=int(receiver_id),
                        payload=payload,
                        payload_bytes=payload_bytes,
                        latency_s=latency_s,
                        distance_m=_dist_m(sender, receiver),
                        fixed_dt=fixed_dt,
                    )
                    step_summary["enqueued_message_count"] += 1.0
                    enqueued_count += 1
                    total_payload_bytes += int(payload_bytes)
        if should_log_periodic(
            int(self._time_step),
            int(self._get_runtime_debug_interval()),
            logger=RUNTIME_LOGGER,
        ):
            RUNTIME_LOGGER.debug(
                "Communication round step=%d policy=%s senders=%s attempted=%d dropped=%d enqueued=%d in_flight=%d payload_bytes=%d",
                self._time_step,
                getattr(self, "_collaboration_policy_id", ""),
                sorted(sender_ids),
                int(step_summary.get("attempted_message_count", 0.0)),
                int(step_summary.get("dropped_message_count", 0.0)),
                enqueued_count,
                len(self._in_flight),
                total_payload_bytes,
            )
        attempted = float(step_summary.get("attempted_message_count", 0.0))
        dropped = float(step_summary.get("dropped_message_count", 0.0))
        avg_latency_s = 0.0
        if self._comm_link_analysis_by_sender:
            latencies = [
                float(stats.get("analysis_latency_s", 0.0))
                for stats in self._comm_link_analysis_by_sender.values()
                if isinstance(stats, dict)
            ]
            if latencies:
                avg_latency_s = float(sum(latencies) / len(latencies))
        self._policy_prev_comm_summary = {
            "attempted_message_count": attempted,
            "dropped_message_count": dropped,
            "drop_ratio_prev_round": (dropped / attempted) if attempted > 0.0 else 0.0,
            "avg_link_latency_s": avg_latency_s,
        }
    def _deliver_messages(self) -> None:
        if not self._in_flight:
            return
        current_step = int(self._time_step)
        remaining: List[V2VMessage] = []
        delivered_count = 0
        for msg in self._in_flight:
            if int(msg.deliver_step) <= current_step:
                self._received[int(msg.receiver_id)].append(msg)
                delivered_count += 1
            else:
                remaining.append(msg)
        self._in_flight = remaining
        if should_log_periodic(
            current_step,
            int(self._get_runtime_debug_interval()),
            logger=RUNTIME_LOGGER,
        ):
            RUNTIME_LOGGER.debug(
                "Delivered messages step=%d delivered=%d remaining_in_flight=%d ego_received=%d",
                current_step,
                delivered_count,
                len(self._in_flight),
                len(self._received.get(int(self.ego.id), deque())),
            )

    def _build_graph_info(self) -> Dict[str, Any]:
        ego_encoding = self._payload_registry.get("object_list").encode(
            self.ego,
            self.obs,
            self.feature_size,
        )
        msgs = self._received.get(int(self.ego.id), deque())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._graph_builder.build(
            ego_actor=self.ego,
            carla_world=self._world._world,
            ego_feat=ego_encoding.feat,
            ego_feat_dim=ego_encoding.feat_dim,
            msgs=msgs,
            t_step=self._time_step,
            dt=float(self._config.world.fixed_delta_seconds),
            device=device,
        )

    def _merge_step_info(self, info: Dict[str, Any], requested_action: Any) -> Dict[str, Any]:
        del requested_action
        shared_data = self._build_graph_info()
        reward_info = {
            k: v
            for k, v in info.items()
            if k.startswith("r_")
            or k in [
                "wpt_dis",
                "speed_parallel",
                "speed_perpendicular",
                "speed_norm",
                "ttc",
                "time_penalty",
            ]
        }
        reward_info["ego_x"] = self.ego.get_transform().location.x
        reward_info["ego_y"] = self.ego.get_transform().location.y
        shared_data.update(reward_info)
        if should_log_periodic(
            int(self._time_step),
            int(self._get_runtime_debug_interval()),
            logger=RUNTIME_LOGGER,
        ):
            valid_nodes = int(np.asarray(shared_data.get("node_mask", np.zeros(0))).sum())
            edge_index = np.asarray(shared_data.get("edge_index", np.zeros((2, 0))))
            msg_count = len(self._received.get(int(self.ego.id), deque()))
            RUNTIME_LOGGER.debug(
                "Step info merged step=%d valid_nodes=%d num_edges=%d ego_received_msgs=%d reward_keys=%s",
                self._time_step,
                valid_nodes,
                edge_index.shape[1] if edge_index.ndim == 2 else 0,
                msg_count,
                sorted(reward_info.keys()),
            )
        return shared_data

    def _build_reset_info(self) -> Dict[str, Any]:
        shared_data = self._build_graph_info()
        ego_location = np.array([*get_vehicle_pos(self.ego)])
        reward_info = {
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": 0,
            "speed_perpendicular": 0,
            "speed_norm": 0,
            "wpt_dis": self.get_wpt_dist(ego_location),
            "r_waypoints": 0,
            "r_speed": 0,
            "r_collision": 0,
            "r_out_of_lane": 0,
            "r_destination": 0,
            "time_penalty": 0,
            "ttc": 0,
        }
        shared_data.update(reward_info)
        valid_nodes = int(np.asarray(shared_data.get("node_mask", np.zeros(0))).sum())
        edge_index = np.asarray(shared_data.get("edge_index", np.zeros((2, 0))))
        RUNTIME_LOGGER.info(
            "Built reset graph info valid_nodes=%d num_edges=%d",
            valid_nodes,
            edge_index.shape[1] if edge_index.ndim == 2 else 0,
        )
        return shared_data

    def _handle_episode_end(self, terminated: bool, truncated: bool, info: Dict[str, Any]) -> Dict[str, Any]:
        del terminated, truncated
        return info

    # =========================================================
    # Environment overrides
    # =========================================================

    def _cleanup_actor_flow(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(vehicle)
            if y > -81.2 or x < -38.4 or x > 31.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
