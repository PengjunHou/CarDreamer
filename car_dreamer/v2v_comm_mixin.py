"""Reusable V2V cooperative-perception communication mixin.

``V2VCommMixin`` factors out the cooperative-perception machinery that used to live
inside ``carla_group_right_turn_auto`` so that any task env can opt into it:

* a per-vehicle camera/observer pool of cooperative vehicles,
* per-episode random participation (candidate pool) for generalization,
* a Base-Station policy lifecycle (``_update_policy_lifecycle``) with lifetime ``Td`` (§3),
* the streaming communication process (``CommunicationProcess``): sensor sampling at ``Ts``,
  per-link sender queues (proc + queue + tx delay), and a receive queue with a ``Tw`` window, and
* a cooperative GNN graph rebuilt from the **messages actually available in the receive queue**,
  which gives genuine local <-> V2V switching (§12-§14).

Cooperative vehicles are declared in the task's ``scenario_actors.vehicles`` config: any
vehicle with a ``start`` point is spawned by the reusable ``ScenarioActorManager`` and then
registered here via :py:meth:`_register_cooperative_candidate` (the manager's
``cooperative_hook``), which attaches a camera+collision observer and adds it to this episode's
candidate pool with probability ``coop_participation_prob``.

A host task opts in by:
  1. ``class CarlaXEnv(V2VCommMixin, CarlaWptEnv): ...``
  2. ``__init__``: call ``self._init_v2v()`` after ``super().__init__()``.
  3. ``on_reset``: ``self._reset_group_runtime_state(); self._destroy_group_observers();
     super().on_reset(); self.groups.setdefault(GROUP_ID, set()).add(int(self.ego.id))``.
     (Candidate vehicles are registered automatically by the base-env scenario hook.)
  4. ``on_step``: ``self._update_group_observations(); self._update_wam_runtime_state();
     self._update_policy_lifecycle(step); self._run_comm_step(step)``.
  5. ``step`` / ``reset``: merge ``self._merge_step_info(info, action)`` /
     ``self._build_reset_info()`` into the returned info dict.

The host env must provide (all already present on every ``CarlaBaseEnv``/``CarlaWptEnv``):
``self.ego``, ``self.obs``, ``self._world``, ``self._time_step``, ``self.get_state()``,
and (for graph reset info) ``self.get_wpt_dist``. The task config must provide a
``group_observation`` block (camera+collision) and may provide ``communication`` /
``graph`` / ``feature_size`` / ``coop_participation_prob`` keys (all default-tolerant).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import carla
import numpy as np
import torch
from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

from .toolkit import (
    CommConfig,
    CommPolicy,
    CommunicationProcess,
    GraphBuildConfig,
    Observer,
    SenseSnapshot,
    VehicleNodeGraphBuilder,
    _dist_m,
    get_vehicle_pos,
    make_local_policy,
    payload_fn_llm,
    shannon_rate_bps,
)
from .toolkit.observer.handlers.utils import is_fov_visible
from .toolkit.wam import (
    OBJECT_STATE_DIM,
    BevSpec,
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMPolicy,
    rasterize_bev,
    build_coop_request,
    build_wam_hetero_graph,
    hetero_graph_stats,
    predict_notable_motion,
    select_notable_objects,
)


GROUP_ID = 0
RECEIVED_BUFFER_SIZE = 256
V2V_LOGGER = get_runtime_logger("car_dreamer.v2v")


class V2VCommMixin:
    # =========================================================
    # Initialization
    # =========================================================

    def _init_v2v(self) -> None:
        """Initialize cooperative-group state, communication process, and graph builder.

        Call once from the host env ``__init__`` after ``super().__init__()``.
        """
        # --- cooperative group / candidate state ---
        self.groups: Dict[int, set] = {}
        self.group_vehs: List[carla.Actor] = []
        self.coop_participant_ids: set = set()
        self.selected_collaborators: set = set()
        self._other_observers: Dict[int, Observer] = {}
        self.group_obs: Dict[int, Dict[str, Any]] = {}
        self._prev_action = None
        self._actor_cache: Dict[int, carla.Actor] = {}
        # Per-episode probability that each camera vehicle joins cooperative perception.
        self.coop_participation_prob = float(getattr(self._config, "coop_participation_prob", 0.5))

        # --- communication config / link-rate model ---
        comm_cfg = getattr(self._config, "communication", None)
        self.group_update_period = int(getattr(comm_cfg, "group_update_period", 20))
        policy_bandwidth_hz = float(getattr(comm_cfg, "policy_bandwidth_hz", 6e6))
        bandwidth_ratio = float(getattr(comm_cfg, "bandwidth_ratio", 1.0))
        proc_delay_s = float(getattr(comm_cfg, "proc_delay_s", 0.05))
        distance_decay_m = float(getattr(comm_cfg, "distance_decay_m", 60.0))
        min_rate_factor = float(getattr(comm_cfg, "min_rate_factor", 0.2))
        overhead_bytes = int(getattr(comm_cfg, "overhead_bytes", 64))
        self._comm_policy_bandwidth_hz = policy_bandwidth_hz
        self._comm_bandwidth_ratio = float(np.clip(bandwidth_ratio, 0.0, 1.0))
        self._comm_overhead_bytes = overhead_bytes
        # FSPL / SNR parameters for the policy-conditioned per-message transmission rate (§8).
        self._comm_rate_params = dict(
            tx_power_dbm=float(getattr(comm_cfg, "tx_power_dbm", 20.0)),
            noise_figure_db=float(getattr(comm_cfg, "noise_figure_db", 9.0)),
            carrier_freq_hz=float(getattr(comm_cfg, "carrier_freq_hz", 5.9e9)),
            distance_decay_m=distance_decay_m,
            min_rate_factor=min_rate_factor,
        )

        # --- streaming communication process (Td/Ts/Tw/Ta + sender/receive queues) ---
        dt = float(getattr(getattr(self._config, "world", None), "fixed_delta_seconds", 0.1))
        self._comm_config = CommConfig.from_seconds(
            dt=dt,
            policy_duration_s=float(getattr(comm_cfg, "policy_duration_s", 2.0)),
            sensor_period_s=float(getattr(comm_cfg, "sensor_period_s", 0.5)),
            prediction_window_s=float(getattr(comm_cfg, "prediction_window_s", 2.0)),
            action_period_s=float(getattr(comm_cfg, "action_period_s", getattr(comm_cfg, "sensor_period_s", 0.5))),
            proc_delay_s=proc_delay_s,
            flush_old_policy_queue=bool(getattr(comm_cfg, "flush_old_policy_queue", False)),
            allow_cross_policy_messages=bool(getattr(comm_cfg, "allow_cross_policy_messages", False)),
        )
        # Bundled modalities every collaborator streams under a cooperative policy (one message).
        modalities = getattr(comm_cfg, "collaborator_modalities", None)
        if modalities is None:
            modalities = [str(getattr(getattr(self._config, "wam", None), "default_modality", "objlist"))]
        self._collaborator_modalities = tuple(str(m) for m in modalities) or ("objlist",)
        # `comm_period` retained as an alias of the sensor period (frequency_steps in WAMPolicy views).
        self.comm_period = int(self._comm_config.sensor_period_steps)
        self._comm_process: Optional[CommunicationProcess] = None
        self._comm_policy_counter = 0

        self.payload_fn = payload_fn_llm
        self.trans_msg_type = str(getattr(self._config, "trans_msg_type", "image"))

        # --- graph builder ---
        self.feature_size = int(getattr(self._config, "feature_size", 64))
        graph_cfg = getattr(self._config, "graph", None)
        self._graph_builder = VehicleNodeGraphBuilder(
            GraphBuildConfig(
                window_s=float(getattr(graph_cfg, "window_s", 2.0)),
                Tmax=int(getattr(graph_cfg, "tmax", 15)),
                max_nodes=int(getattr(graph_cfg, "max_nodes", 3)),
                feat_dim_max=int(getattr(graph_cfg, "feat_dim_max", 128)),
                star_graph=bool(getattr(graph_cfg, "star_graph", True)),
            )
        )
        self._init_wam_config()

    def _init_wam_config(self) -> None:
        wam_cfg = getattr(self._config, "wam", None)
        self._wam_enabled = bool(getattr(wam_cfg, "enabled", True))
        self._wam_notable_distance_m = float(getattr(wam_cfg, "notable_distance_m", 10.0))
        self._wam_max_notable_objects = int(getattr(wam_cfg, "max_notable_objects", 3))
        self._wam_reference_waypoint_count = int(getattr(wam_cfg, "reference_waypoint_count", 6))
        self._wam_prediction_horizon_steps = int(getattr(wam_cfg, "prediction_horizon_steps", 6))
        self._wam_uncertainty_threshold = float(getattr(wam_cfg, "uncertainty_threshold", 1.0))
        self._wam_visible_uncertainty = float(getattr(wam_cfg, "visible_uncertainty", 0.2))
        self._wam_invisible_uncertainty = float(getattr(wam_cfg, "invisible_uncertainty", 2.0))
        self._wam_default_modality = str(getattr(wam_cfg, "default_modality", "objlist"))
        self._wam_base_station_policy = str(getattr(wam_cfg, "base_station_policy", "placeholder_all"))
        self._wam_local_sight_fov = getattr(
            wam_cfg,
            "local_sight_fov",
            self._get_config_value(("observation", "camera", "attributes", "fov"), 120.0),
        )
        self._wam_local_sight_range = getattr(wam_cfg, "local_sight_range_m", 64.0)
        self._wam_collaborator_sight_fov = getattr(
            wam_cfg,
            "collaborator_sight_fov",
            getattr(
                getattr(getattr(self._config, "group_observation", None), "camera", None),
                "fov",
                self._get_config_value(("group_observation", "camera", "attributes", "fov"), 120.0),
            ),
        )
        self._wam_collaborator_sight_range = getattr(wam_cfg, "collaborator_sight_range_m", 64.0)

        # --- policy-conditioned hetero graph (§4-§7, §9) ---
        self._wam_build_graph = bool(getattr(wam_cfg, "build_graph", True))
        graph_wam_cfg = getattr(wam_cfg, "graph", None)
        self._wam_graph_embed = bool(getattr(graph_wam_cfg, "embed", False))
        self._wam_graph_route_waypoints = int(
            getattr(graph_wam_cfg, "route_waypoints", self._wam_reference_waypoint_count)
        )
        self._wam_graph_max_object_nodes = int(getattr(graph_wam_cfg, "max_object_nodes", 32))
        self._wam_graph_hidden_dim = int(getattr(graph_wam_cfg, "hidden_dim", 256))
        self._wam_graph_num_layers = int(getattr(graph_wam_cfg, "num_layers", 3))
        self._wam_graph_num_heads = int(getattr(graph_wam_cfg, "num_heads", 8))
        self._wam_graph_gamma_freshness = float(getattr(graph_wam_cfg, "gamma_freshness", 5.0))
        self._wam_graph_bev_channels = int(getattr(graph_wam_cfg, "bev_channels", 7))
        self._wam_graph_bev_size = int(getattr(graph_wam_cfg, "bev_size", 64))
        self._wam_graph_bev_range_m = float(getattr(graph_wam_cfg, "bev_range_m", 50.0))
        self._wam_bev_spec = BevSpec(size=self._wam_graph_bev_size, range_m=self._wam_graph_bev_range_m)
        self._wam_graph_net = None
        self._wam_graph_embeddings = None

        self._reset_wam_runtime_state()

    def _get_config_value(self, path, default):
        value = self._config
        for key in path:
            value = getattr(value, key, None)
            if value is None:
                return default
        return value

    # =========================================================
    # Actor cache helpers
    # =========================================================

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

    # =========================================================
    # Reset / observers
    # =========================================================

    def _reset_group_runtime_state(self) -> None:
        self.group_vehs = []
        self.groups = {}
        self.coop_participant_ids = set()
        self.selected_collaborators = set()
        self._prev_action = None
        self._actor_cache = {}
        self._comm_process = None  # lazily (re)created per episode once the ego exists
        self._comm_policy_counter = 0
        self._reset_wam_runtime_state()
        V2V_LOGGER.debug("V2V runtime state reset.")

    def _reset_wam_runtime_state(self) -> None:
        self._wam_notable_records = []
        self._wam_object_states = []
        self._wam_object_states_step = -1  # last step the (per-step) visibility scan ran
        self._wam_motion_predictions = {}
        self._wam_coop_request = None
        self._wam_graph = None
        self._wam_graph_embeddings = None
        self._wam_policy = WAMPolicy(
            selected_vehicle_ids=(),
            modality_by_vehicle={},
            bandwidth_by_vehicle={},
            frequency_steps=int(getattr(self, "comm_period", 1)),
            reason="not_initialized",
        )
        self._wam_last_predict_step = -1  # last step the (Ta-gated) notable/request ran

    def _destroy_group_observers(self) -> None:
        for observer in self._other_observers.values():
            observer.destroy()
        self._other_observers = {}
        self.group_obs = {}

    def _create_group_observer(self, vehicle: carla.Actor) -> None:
        group_observation = self._config.group_observation
        observer = Observer(self._world, group_observation)
        self._other_observers[int(vehicle.id)] = observer
        observer.reset(vehicle)
        self.group_obs[int(vehicle.id)], _ = observer.get_observation(self.get_state())

    def _update_group_observations(self) -> None:
        # Only participating (collaborating) vehicles' observations are consumed by
        # V2V communication and the policy graph, so only refresh those.
        for actor in self.group_vehs:
            if int(actor.id) not in self.coop_participant_ids:
                continue
            observer = self._other_observers.get(int(actor.id))
            if observer is not None:
                self.group_obs[int(actor.id)], _ = observer.get_observation(self.get_state())

    # =========================================================
    # Cooperative-vehicle registration (driven by scenario_actors `start` vehicles)
    # =========================================================

    def _register_cooperative_candidate(self, vehicle: carla.Actor) -> None:
        """Turn an already-spawned scenario vehicle into a V2V cooperative candidate.

        Invoked (as the ``ScenarioActorManager`` ``cooperative_hook``) for every
        ``scenario_actors`` vehicle that has a ``start`` point. The vehicle gets a
        camera+collision observer and joins this episode's candidate pool with probability
        ``coop_participation_prob`` (only participants enter the group, share over V2V, and
        appear in the policy graph). The vehicle itself is spawned (and destroyed on reset)
        by the scenario manager via the world's ``actor_dict``.
        """
        if vehicle is None:
            return
        self.groups.setdefault(GROUP_ID, set()).add(int(self.ego.id))
        self._create_group_observer(vehicle)  # camera+collision sensor
        self.group_vehs.append(vehicle)
        self._cache_actor(vehicle)
        if np.random.random() < float(getattr(self, "coop_participation_prob", 0.5)):
            self.coop_participant_ids.add(int(vehicle.id))
            self.groups[GROUP_ID].add(int(vehicle.id))
        V2V_LOGGER.debug(
            "Registered cooperative candidate id=%s participant=%s total_candidates=%d",
            int(vehicle.id),
            int(vehicle.id) in self.coop_participant_ids,
            len(self.group_vehs),
        )

    # =========================================================
    # Communication
    # =========================================================

    def _ensure_comm_process(self) -> CommunicationProcess:
        """Lazily create the per-episode communication process (needs the ego id)."""
        if self._comm_process is None:
            self._comm_process = CommunicationProcess(self._comm_config, int(self.ego.id))
        return self._comm_process

    def _next_policy_id(self) -> int:
        pid = int(self._comm_policy_counter)
        self._comm_policy_counter += 1
        return pid

    def _link_rate_bps(self, sender_id: int, distance_m: float, bandwidth_ratio: float) -> float:
        """Policy-conditioned Shannon link rate m -> ego (§8)."""
        ratio = float(np.clip(float(bandwidth_ratio), 0.0, 1.0))
        bandwidth_hz = float(self._comm_policy_bandwidth_hz) * ratio
        return shannon_rate_bps(float(distance_m), bandwidth_hz, **self._comm_rate_params)

    def _build_coop_policy(self, step: int, candidates: List[int]) -> CommPolicy:
        """Base-Station cooperative policy: all candidates collaborate, bundled modalities (§3)."""
        bandwidth_ratio = float(self._comm_bandwidth_ratio)
        modalities = tuple(self._collaborator_modalities)
        return CommPolicy(
            policy_id=self._next_policy_id(),
            request_vehicle_id=int(self.ego.id),
            start_step=int(step),
            duration_steps=int(self._comm_config.policy_duration_steps),
            selected_collaborators=tuple(int(c) for c in candidates),
            modalities_by_vehicle={int(c): modalities for c in candidates},
            bandwidth_by_vehicle={int(c): bandwidth_ratio for c in candidates},
            reason="coop_request",
        )

    def _sync_policy_views(self, policy: CommPolicy) -> None:
        """Mirror the active :class:`CommPolicy` into the WAMPolicy view used for info/graph."""
        self.selected_collaborators = set(int(c) for c in policy.selected_collaborators)
        self._wam_policy = WAMPolicy(
            selected_vehicle_ids=tuple(int(c) for c in policy.selected_collaborators),
            modality_by_vehicle={int(c): tuple(m) for c, m in policy.modalities_by_vehicle.items()},
            bandwidth_by_vehicle=dict(policy.bandwidth_by_vehicle),
            frequency_steps=int(self._comm_config.sensor_period_steps),
            reason=str(policy.reason),
        )

    def _update_policy_lifecycle(self, step: int) -> None:
        """Install / expire the Base-Station policy with lifetime Td (§3, §9).

        A cooperative policy runs for its full ``Td``; a ``local-only`` policy is interruptible
        by a fresh high-uncertainty request and otherwise refreshed every ``Td``.
        """
        if not bool(getattr(self, "_wam_enabled", True)):
            return
        proc = self._ensure_comm_process()
        active = proc.policy
        if active is not None and not active.is_local_only and active.active_at(step):
            return  # an active cooperative policy runs for its full duration

        request = getattr(self, "_wam_coop_request", None)
        candidates = sorted(int(v) for v in getattr(self, "coop_participant_ids", set()))
        if request is not None and candidates:
            policy = self._build_coop_policy(step, candidates)
        elif active is not None and active.is_local_only and active.active_at(step):
            return  # keep the current local-only policy (no request) to avoid churn
        else:
            policy = make_local_policy(
                policy_id=self._next_policy_id(),
                request_vehicle_id=int(self.ego.id),
                start_step=int(step),
                duration_steps=int(self._comm_config.policy_duration_steps),
            )
        proc.set_policy(policy, int(step))
        self._sync_policy_views(policy)

    def _build_sense_snapshot(
        self, sender_id: int, actor: carla.Actor, object_states: List[ObjectState], policy: CommPolicy
    ) -> Optional[SenseSnapshot]:
        """Bundle one collaborator's per-modality observation at this sensor tick (§2.2, §13)."""
        modalities = policy.modalities_by_vehicle.get(int(sender_id), self._collaborator_modalities)
        observed = [s for s in object_states if int(sender_id) in s.visible_to_collaborators]
        tf = actor.get_transform()
        vel = actor.get_velocity()
        data: Dict[str, Any] = {
            "sender_id": int(sender_id),
            "pose": {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "z": float(tf.location.z),
                "yaw": float(tf.rotation.yaw),
            },
            "vel": {"vx": float(vel.x), "vy": float(vel.y)},
            "object_states": tuple(observed),
        }
        # Legacy vehicle-node-graph feature (consumed by VehicleNodeGraphBuilder).
        if self.payload_fn is not None:
            feat_payload = self.payload_fn(actor, self.group_obs.get(int(sender_id), {}), self.feature_size)
            data["feat"] = feat_payload.get("feat")
            feat_dim = feat_payload.get("feat_dim")
            if feat_dim is not None:
                data["feat_dim"] = int(feat_dim)

        overhead = int(getattr(self, "_comm_overhead_bytes", 64))
        payload_size = overhead
        for modality in modalities:
            if modality == "bev":
                veh_pose = (float(tf.location.x), float(tf.location.y), float(tf.rotation.yaw))
                data["bev"] = rasterize_bev(veh_pose, observed, route_xy=(), spec=self._wam_bev_spec)
                payload_size += int(self._wam_bev_payload_bytes())
            else:
                data["objlist"] = {"observed_object_ids": tuple(int(s.actor_id) for s in observed)}
                payload_size += int(max(len(observed), 0) * OBJECT_STATE_DIM * 4)
        return SenseSnapshot(
            sender_id=int(sender_id),
            distance_m=float(_dist_m(actor, self.ego)),
            payload_size=int(payload_size),
            modalities=tuple(modalities),
            data=data,
        )

    def _run_comm_step(self, step: int) -> None:
        """One simulation step of the streaming comm process: deliver, then stream at sensor ticks."""
        proc = self._ensure_comm_process()
        proc.deliver(int(step))
        if not proc.is_sensor_tick(int(step)):
            return
        self._refresh_object_states(int(step))  # collaborator snapshots use current visibility (Ts)
        policy = proc.policy
        object_states = list(getattr(self, "_wam_object_states", []))
        actor_map = self._build_group_actor_map()
        snapshots: Dict[int, SenseSnapshot] = {}
        for sender_id in policy.selected_collaborators:
            actor = actor_map.get(int(sender_id)) or self._get_group_member_actor(int(sender_id))
            if actor is None:
                continue
            snapshot = self._build_sense_snapshot(int(sender_id), actor, object_states, policy)
            if snapshot is not None:
                snapshots[int(sender_id)] = snapshot
        emitted = proc.generate(int(step), snapshots, self._link_rate_bps)
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(int(step), int(runtime_cfg["step_debug_interval"]), logger=V2V_LOGGER):
            V2V_LOGGER.debug(
                "Comm tick step=%d policy=%s local_only=%s collaborators=%s emitted=%d in_flight=%d recv=%d",
                step,
                policy.policy_id,
                policy.is_local_only,
                sorted(policy.selected_collaborators),
                len(emitted),
                len(proc.in_flight),
                len(proc.receive_queue),
            )

    def _build_group_actor_map(self) -> Dict[int, carla.Actor]:
        actor_map: Dict[int, carla.Actor] = {}
        if getattr(self, "ego", None) is not None:
            actor_map[int(self.ego.id)] = self.ego
        for actor in self.group_vehs:
            actor_map[int(actor.id)] = actor
        self._actor_cache.update(actor_map)
        return actor_map

    # =========================================================
    # WAM runtime: notable objects -> request (policy decided in _update_policy_lifecycle)
    # =========================================================

    def _actor_polygon_xy(self, actor: carla.Actor):
        tf = actor.get_transform()
        bb = actor.bounding_box
        length = max(float(bb.extent.x), 0.1)
        width = max(float(bb.extent.y), 0.1)
        yaw = math.radians(float(tf.rotation.yaw))
        local = np.array(
            [
                [length, width],
                [length, -width],
                [-length, -width],
                [-length, width],
            ],
            dtype=np.float32,
        )
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        poly = local @ rot.T
        poly[:, 0] += float(tf.location.x)
        poly[:, 1] += float(tf.location.y)
        return [(float(x), float(y)) for x, y in poly]

    def _wam_reference_route_points(self):
        count = max(int(getattr(self, "_wam_reference_waypoint_count", 6)), 1)
        route = []
        for waypoint in list(getattr(self, "waypoints", []))[:count]:
            route.append((float(waypoint[0]), float(waypoint[1])))
        if route:
            return route
        ego = getattr(self, "ego", None)
        if ego is None:
            return []
        loc = ego.get_transform().location
        return [(float(loc.x), float(loc.y))]

    def _wam_object_actors(self) -> List[carla.Actor]:
        ego_id = int(self.ego.id)
        actors = self._world._world.get_actors()
        objects = []
        for actor in list(actors.filter("vehicle.*")) + list(actors.filter("walker.pedestrian.*")):
            if int(actor.id) == ego_id:
                continue
            objects.append(actor)
        return objects

    def _wam_actor_visible_from(
        self,
        observer: carla.Actor,
        target: carla.Actor,
        polygons: Dict[int, Any],
        *,
        fov,
        sight_range,
    ) -> bool:
        target_poly = polygons.get(int(target.id))
        if target_poly is None:
            return False
        tf = observer.get_transform()
        return bool(
            is_fov_visible(
                (float(tf.location.x), float(tf.location.y)),
                float(tf.rotation.yaw),
                int(observer.id),
                int(target.id),
                target_poly,
                polygons,
                fov,
                sight_range,
            )
        )

    def _wam_build_object_states(self) -> List[ObjectState]:
        object_actors = self._wam_object_actors()
        observer_actors = [self.ego] + [actor for actor in self.group_vehs if actor is not None]
        polygons = {}
        for actor in object_actors + observer_actors:
            try:
                polygons[int(actor.id)] = self._actor_polygon_xy(actor)
            except Exception:
                V2V_LOGGER.debug("Failed to build actor polygon actor_id=%s", getattr(actor, "id", None))

        object_states: List[ObjectState] = []
        participant_ids = set(int(actor_id) for actor_id in getattr(self, "coop_participant_ids", set()))
        participant_actors = [
            actor for actor in self.group_vehs if actor is not None and int(actor.id) in participant_ids
        ]
        for actor in object_actors:
            try:
                tf = actor.get_transform()
                vel = actor.get_velocity()
                bb = actor.bounding_box
                visible_to_ego = self._wam_actor_visible_from(
                    self.ego,
                    actor,
                    polygons,
                    fov=self._wam_local_sight_fov,
                    sight_range=self._wam_local_sight_range,
                )
                visible_to_collaborators = []
                for observer in participant_actors:
                    if int(observer.id) == int(actor.id):
                        continue
                    if self._wam_actor_visible_from(
                        observer,
                        actor,
                        polygons,
                        fov=self._wam_collaborator_sight_fov,
                        sight_range=self._wam_collaborator_sight_range,
                    ):
                        visible_to_collaborators.append(int(observer.id))
                actor_type = str(getattr(actor, "type_id", ""))
                object_class = "pedestrian" if "walker.pedestrian" in actor_type else "vehicle"
                object_states.append(
                    ObjectState(
                        actor_id=int(actor.id),
                        actor_type=actor_type,
                        object_class=object_class,
                        x=float(tf.location.x),
                        y=float(tf.location.y),
                        z=float(tf.location.z),
                        vx=float(vel.x),
                        vy=float(vel.y),
                        yaw=float(tf.rotation.yaw),
                        length=float(2.0 * bb.extent.x),
                        width=float(2.0 * bb.extent.y),
                        height=float(2.0 * bb.extent.z),
                        bbox=tuple(polygons.get(int(actor.id), ())),
                        visible_to_ego=visible_to_ego,
                        visible_to_collaborators=tuple(sorted(visible_to_collaborators)),
                    )
                )
            except Exception:
                V2V_LOGGER.debug("Failed to build WAM object state actor_id=%s", getattr(actor, "id", None))
        return object_states

    def _refresh_object_states(self, step: Optional[int] = None) -> None:
        """Rebuild the per-actor visibility scan (``_wam_object_states``) at most once per step.

        This is the *perception* input shared by ego's local observation and every collaborator's
        sensor snapshot, so it must stay current at each sensor tick (``Ts``) -- independent of the
        slower ``Ta`` prediction cadence.
        """
        if not bool(getattr(self, "_wam_enabled", True)):
            return
        if step is None:
            step = int(getattr(self, "_time_step", 0))
        if int(getattr(self, "_wam_object_states_step", -1)) == int(step):
            return
        self._wam_object_states = self._wam_build_object_states()
        self._wam_object_states_step = int(step)

    def _update_wam_runtime_state(self, *, force: bool = False) -> None:
        """Refresh perception every step; recompute motion uncertainty + request every ``Ta`` (§2.4).

        This is the ego's own *local sensing -> notable-object motion prediction -> request* chain.
        It no longer decides the policy (see :meth:`_update_policy_lifecycle`) nor builds the graph
        (see :meth:`_build_wam_graph`), both of which are driven by the receive queue.
        """
        if not bool(getattr(self, "_wam_enabled", True)):
            return
        step = int(getattr(self, "_time_step", 0))
        self._refresh_object_states(step)  # perception: every step (deduped)

        last = int(getattr(self, "_wam_last_predict_step", -1))
        action_period = max(int(self._comm_config.action_period_steps), 1)
        if not force and last >= 0 and (step - last) < action_period:
            return  # notable-motion prediction / request runs only every Ta

        route_points = self._wam_reference_route_points()
        self._wam_notable_records = select_notable_objects(
            self._wam_object_states,
            route_points,
            notable_distance_m=float(self._wam_notable_distance_m),
            max_notable_objects=int(self._wam_max_notable_objects),
        )
        dt = float(getattr(getattr(self._config, "world", None), "fixed_delta_seconds", 0.1))
        self._wam_motion_predictions = predict_notable_motion(
            self._wam_notable_records,
            dt=dt,
            horizon_steps=int(self._wam_prediction_horizon_steps),
            visible_uncertainty=float(self._wam_visible_uncertainty),
            invisible_uncertainty=float(self._wam_invisible_uncertainty),
        )
        self._wam_coop_request = build_coop_request(
            ego_id=int(self.ego.id),
            step=step,
            predictions=self._wam_motion_predictions,
            uncertainty_threshold=float(self._wam_uncertainty_threshold),
        )
        self._wam_last_predict_step = step
        if should_log_periodic(step, int(get_runtime_logging_config()["step_debug_interval"]), logger=V2V_LOGGER):
            V2V_LOGGER.debug(
                "WAM step=%d notable=%s max_uncertainty=%.3f triggered=%s",
                step,
                [record.object_state.actor_id for record in self._wam_notable_records],
                self._wam_max_uncertainty(),
                self._wam_coop_request is not None,
            )

    def _wam_max_uncertainty(self) -> float:
        if not getattr(self, "_wam_motion_predictions", None):
            return 0.0
        return float(max(pred.uncertainty_score for pred in self._wam_motion_predictions.values()))

    # =========================================================
    # WAM policy-conditioned hetero graph (§4-§7, §9)
    # =========================================================

    def _wam_route_xy(self):
        route = []
        for waypoint in list(getattr(self, "waypoints", []))[: int(self._wam_graph_route_waypoints)]:
            route.append((float(waypoint[0]), float(waypoint[1])))
        return tuple(route)

    def _wam_vehicle_node_input(self, actor: carla.Actor, *, is_ego: bool, agent_slot: int, route_xy=()):
        tf = actor.get_transform()
        vel = actor.get_velocity()
        return VehicleNodeInput(
            actor_id=int(actor.id),
            is_ego=bool(is_ego),
            agent_slot=int(agent_slot),
            x=float(tf.location.x),
            y=float(tf.location.y),
            z=float(tf.location.z),
            vx=float(vel.x),
            vy=float(vel.y),
            yaw=float(tf.rotation.yaw),
            q_comm=1.0,
            q_comp=1.0,
            route_xy=tuple(route_xy),
        )

    def _wam_objlist_payload_bytes(self, n_objects: int) -> float:
        overhead = int(getattr(self, "_comm_overhead_bytes", 64))
        return float(max(int(n_objects), 0) * OBJECT_STATE_DIM * 4 + overhead)

    def _wam_bev_payload_bytes(self) -> float:
        return float(int(self._wam_graph_bev_channels) * int(self._wam_graph_bev_size) ** 2)

    def _wam_collaborator_node_from_message(self, message, *, agent_slot: int) -> VehicleNodeInput:
        """Collaborator vehicle node from the message's pose snapshot at ``t_sense`` (§5.1, §13)."""
        pose = message.data.get("pose", {})
        vel = message.data.get("vel", {})
        return VehicleNodeInput(
            actor_id=int(message.sender_id),
            is_ego=False,
            agent_slot=int(agent_slot),
            x=float(pose.get("x", 0.0)),
            y=float(pose.get("y", 0.0)),
            z=float(pose.get("z", 0.0)),
            vx=float(vel.get("vx", 0.0)),
            vy=float(vel.get("vy", 0.0)),
            yaw=float(pose.get("yaw", 0.0)),
            q_comm=1.0,
            q_comp=1.0,
            route_xy=(),
        )

    def _build_wam_graph(self, step: Optional[int] = None) -> None:
        """Assemble the cooperative graph from the **receive queue** (§12-§14).

        The graph at time ``t`` is decided by the messages actually available (Tw / policy
        filtered), not by the policy directly: no available messages -> ego-only *local* graph;
        otherwise a V2V graph whose ``veh_veh`` edges carry the **measured** latency ``L_M``.
        Collaborator observations come from each message's ``t_sense`` snapshot; ego's own
        observation uses its live local sensing at ``t``.
        """
        if step is None:
            step = int(getattr(self, "_time_step", 0))
        proc = self._ensure_comm_process()
        messages = proc.available_messages(int(step)) if proc.policy is not None else []
        dt = float(self._comm_config.dt)
        live_states = list(getattr(self, "_wam_object_states", []))

        ego = self._wam_vehicle_node_input(self.ego, is_ego=True, agent_slot=0, route_xy=self._wam_route_xy())
        ego_visible = [s for s in live_states if bool(s.visible_to_ego)]
        observations = [
            ObservationNodeInput(
                vehicle_id=int(self.ego.id),
                modality="objlist",
                observed_object_ids=tuple(int(s.actor_id) for s in ego_visible),
                payload_bytes=self._wam_objlist_payload_bytes(len(ego_visible)),
                latency_s=0.0,
                freshness=1.0,
                quality=1.0,
                sample_age_s=0.0,
            )
        ]

        # Most-recent message per collaborator (§13.2): its latency drives the veh_veh edge.
        latest_by_sender: Dict[int, Any] = {}
        for message in messages:
            current = latest_by_sender.get(int(message.sender_id))
            if current is None or int(message.t_sense) >= int(current.t_sense):
                latest_by_sender[int(message.sender_id)] = message

        collaborators: List[VehicleNodeInput] = []
        latency_by_vehicle: Dict[int, float] = {}
        modality_by_vehicle: Dict[int, tuple] = {}
        objects_by_id: Dict[int, ObjectState] = {}
        slot = 1
        for sender_id, message in sorted(latest_by_sender.items()):
            collaborators.append(self._wam_collaborator_node_from_message(message, agent_slot=slot))
            slot += 1
            latency_s = float(message.total_latency)
            latency_by_vehicle[int(sender_id)] = latency_s
            freshness = math.exp(-float(self._wam_graph_gamma_freshness) * latency_s)
            sample_age = float(int(step) - int(message.t_sense)) * dt
            modalities = tuple(message.modalities)
            modality_by_vehicle[int(sender_id)] = modalities
            for snap_state in message.data.get("object_states", ()):  # snapshot @ t_sense
                objects_by_id.setdefault(int(snap_state.actor_id), snap_state)
            for modality in modalities:
                if modality == "bev":
                    observed_ids: tuple = ()
                    bev_raster = message.data.get("bev")
                    payload = self._wam_bev_payload_bytes()
                else:
                    observed_ids = tuple(message.data.get("objlist", {}).get("observed_object_ids", ()))
                    bev_raster = None
                    payload = self._wam_objlist_payload_bytes(len(observed_ids))
                observations.append(
                    ObservationNodeInput(
                        vehicle_id=int(sender_id),
                        modality=modality,
                        observed_object_ids=observed_ids,
                        payload_bytes=payload,
                        latency_s=latency_s,
                        freshness=freshness,
                        quality=1.0,
                        sample_age_s=sample_age,
                        bev_raster=bev_raster,
                    )
                )

        # Ego's fresh local states override stale snapshot states for the same object.
        for state in ego_visible:
            objects_by_id[int(state.actor_id)] = state
        objects = list(objects_by_id.values())

        policy_view = WAMPolicy(
            selected_vehicle_ids=tuple(sorted(latest_by_sender.keys())),
            modality_by_vehicle=modality_by_vehicle,
            bandwidth_by_vehicle={},
            frequency_steps=int(self._comm_config.sensor_period_steps),
            reason="receive_queue",
        )
        spec = GraphBuildSpec(
            route_waypoints=int(self._wam_graph_route_waypoints),
            max_object_nodes=int(self._wam_graph_max_object_nodes),
        )
        notable_ids = {int(record.object_state.actor_id) for record in self._wam_notable_records}
        self._wam_graph = build_wam_hetero_graph(
            ego=ego,
            collaborators=collaborators,
            objects=objects,
            observations=observations,
            policy=policy_view,
            spec=spec,
            notable_ids=notable_ids,
            latency_by_vehicle=latency_by_vehicle,
        )
        if self._wam_graph_embed:
            self._run_wam_graph_embedding()

    def _run_wam_graph_embedding(self) -> None:
        """Optional: run the §5 embedding + §9 HGT encoder on the current graph.

        Off by default (``env.wam.graph.embed``); the env builds the graph, while the GNN
        forward is intended for the world-model / training side.
        """
        from .toolkit.wam import WAMGraphModelConfig, WAMHeteroGraphNet

        if self._wam_graph is None:
            return
        if self._wam_graph_net is None:
            cfg = WAMGraphModelConfig(
                route_waypoints=int(self._wam_graph_route_waypoints),
                hidden_dim=int(self._wam_graph_hidden_dim),
                num_layers=int(self._wam_graph_num_layers),
                num_heads=int(self._wam_graph_num_heads),
                bev_channels=int(self._wam_graph_bev_channels),
                bev_size=int(self._wam_graph_bev_size),
            )
            self._wam_graph_net = WAMHeteroGraphNet(cfg).eval()
        with torch.no_grad():
            self._wam_graph_embeddings = self._wam_graph_net(self._wam_graph)

    def _wam_info(self) -> Dict[str, Any]:
        self._update_wam_runtime_state()
        if self._wam_build_graph:
            self._build_wam_graph(int(getattr(self, "_time_step", 0)))
        notable = list(getattr(self, "_wam_notable_records", []))
        policy = getattr(self, "_wam_policy", None)
        info = {
            "wam_notable_object_ids": [int(record.object_state.actor_id) for record in notable],
            "wam_visible_notable_object_ids": [
                int(record.object_state.actor_id) for record in notable if bool(record.visible)
            ],
            "wam_invisible_notable_object_ids": [
                int(record.object_state.actor_id) for record in notable if bool(record.invisible)
            ],
            "wam_uncertainty_max": float(self._wam_max_uncertainty()),
            "wam_coop_triggered": bool(getattr(self, "_wam_coop_request", None) is not None),
            "wam_policy_selected_vehicle_ids": list(policy.selected_vehicle_ids) if policy is not None else [],
            "wam_policy_modality_by_vehicle": dict(policy.modality_by_vehicle) if policy is not None else {},
        }
        graph = getattr(self, "_wam_graph", None)
        if graph is not None:
            info.update(hetero_graph_stats(graph))
        return info

    # =========================================================
    # Graph info construction
    # =========================================================

    def _build_graph_info(self) -> Dict[str, Any]:
        ego_feature = self.payload_fn(
            self.ego,
            self.obs,
            self.feature_size,
        )
        proc = self._ensure_comm_process()
        msgs = proc.available_messages(int(self._time_step)) if proc.policy is not None else []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._graph_builder.build(
            ego_actor=self.ego,
            carla_world=self._world._world,
            ego_feat=ego_feature.get("feat"),
            ego_feat_dim=ego_feature.get("feat_dim"),
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
        shared_data.update(self._wam_info())
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(int(self._time_step), int(runtime_cfg["step_debug_interval"]), logger=V2V_LOGGER):
            valid_nodes = int(np.asarray(shared_data.get("node_mask", np.zeros(0))).sum())
            edge_index = np.asarray(shared_data.get("edge_index", np.zeros((2, 0))))
            proc = self._ensure_comm_process()
            msg_count = len(proc.available_messages(int(self._time_step))) if proc.policy is not None else 0
            V2V_LOGGER.debug(
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
        shared_data.update(self._wam_info())
        valid_nodes = int(np.asarray(shared_data.get("node_mask", np.zeros(0))).sum())
        edge_index = np.asarray(shared_data.get("edge_index", np.zeros((2, 0))))
        V2V_LOGGER.info(
            "Built reset graph info valid_nodes=%d num_edges=%d",
            valid_nodes,
            edge_index.shape[1] if edge_index.ndim == 2 else 0,
        )
        return shared_data
