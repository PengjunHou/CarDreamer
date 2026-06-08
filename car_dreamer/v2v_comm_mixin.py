"""Reusable V2V cooperative-perception communication mixin.

``V2VCommMixin`` factors out the cooperative-perception machinery that used to live
inside ``carla_group_right_turn_auto`` so that any task env can opt into it:

* a per-vehicle camera/observer pool of cooperative vehicles,
* per-episode random participation (candidate pool) for generalization,
* a per-step collaboration-policy hook (``_select_collaborators``),
* V2V message passing with a wireless latency model and a delivery queue, and
* an ego-centric GNN graph built from the ego's received messages.

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
  4. ``on_step``: ``self._deliver_messages(); self._update_group_observations()`` and
     ``self._run_group_communication()`` every ``comm_period`` steps.
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
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import carla
import numpy as np
import torch
from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

from .toolkit import (
    GraphBuildConfig,
    NetResource,
    Observer,
    SimpleWirelessLatency,
    V2VMessage,
    VehicleNodeGraphBuilder,
    _dist_m,
    _tx_bytes_for_latency,
    get_vehicle_pos,
    payload_fn_llm,
)
from .toolkit.observer.handlers.utils import is_fov_visible
from .toolkit.wam import (
    OBJECT_STATE_DIM,
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VehicleNodeInput,
    WAMPolicy,
    build_coop_request,
    build_placeholder_policy,
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
        """Initialize cooperative-group state, latency model, and graph builder.

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

        # --- communication config / latency model ---
        comm_cfg = getattr(self._config, "communication", None)
        self.group_update_period = int(getattr(comm_cfg, "group_update_period", 20))
        self.comm_period = int(getattr(comm_cfg, "comm_period", 5))
        uplink_bps = float(getattr(comm_cfg, "uplink_bps", 6e6))
        downlink_bps = float(getattr(comm_cfg, "downlink_bps", 12e6))
        base_rtt_s = float(getattr(comm_cfg, "base_rtt_s", 0.02))
        proc_delay_s = float(getattr(comm_cfg, "proc_delay_s", 0.005))
        distance_decay_m = float(getattr(comm_cfg, "distance_decay_m", 60.0))
        min_rate_factor = float(getattr(comm_cfg, "min_rate_factor", 0.2))
        jitter_s = float(getattr(comm_cfg, "jitter_s", 0.0))
        overhead_bytes = int(getattr(comm_cfg, "overhead_bytes", 64))
        self._default_net_res = NetResource(uplink_bps=uplink_bps, downlink_bps=downlink_bps)
        self.latency_model = SimpleWirelessLatency(
            base_rtt_s=base_rtt_s,
            proc_delay_s=proc_delay_s,
            distance_decay_m=distance_decay_m,
            min_rate_factor=min_rate_factor,
            jitter_s=jitter_s,
            overhead_bytes=overhead_bytes,
        )
        self.payload_fn = payload_fn_llm
        self.trans_msg_type = str(getattr(self._config, "trans_msg_type", "image"))
        self._in_flight: List[V2VMessage] = []
        self._received: Dict[int, Deque[V2VMessage]] = defaultdict(
            lambda: deque(maxlen=RECEIVED_BUFFER_SIZE)
        )
        self._veh_net_res: Dict[int, NetResource] = {}

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
        self._wam_graph_bev_channels = int(getattr(graph_wam_cfg, "bev_channels", 8))
        self._wam_graph_bev_size = int(getattr(graph_wam_cfg, "bev_size", 128))
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
        self._in_flight = []
        self._received = defaultdict(lambda: deque(maxlen=RECEIVED_BUFFER_SIZE))
        self._veh_net_res = {}
        self._reset_wam_runtime_state()
        V2V_LOGGER.debug("V2V runtime state reset.")

    def _reset_wam_runtime_state(self) -> None:
        self._wam_notable_records = []
        self._wam_object_states = []
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
        self._wam_last_update_step = -1

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

    def _make_payload(self, sender: carla.Actor) -> Dict[str, Any]:
        obs = self.obs if int(sender.id) == int(self.ego.id) else self.group_obs.get(int(sender.id), {})
        payload: Dict[str, Any] = {}
        if self.payload_fn is not None:
            payload = self.payload_fn(
                sender,
                obs,
                self.feature_size,
            )

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

    def _select_collaborators(self, candidate_ids: set) -> set:
        """Pick which candidate vehicles share with the ego this communication step.

        This is the collaboration-policy hook (WAM's ``S_t``). The candidate pool
        (``coop_participant_ids``) is fixed for the episode; the policy chooses a subset
        of it to actually collaborate with at each communication step.

        The default returns the full candidate set, preserving the prior
        "every participant shares" behaviour. Override / replace this method with the
        collaboration policy; it may read any runtime state via ``self`` (ego, graph,
        ``group_obs``, distances, bandwidth budget, ...).

        :param candidate_ids: candidate vehicle ids (this episode's participants).
        :return: the subset of ``candidate_ids`` that shares with the ego this step.
        """
        if not bool(getattr(self, "_wam_enabled", True)):
            return set(candidate_ids)
        policy = getattr(self, "_wam_policy", None)
        if policy is None:
            return set()
        return set(int(vehicle_id) for vehicle_id in policy.selected_vehicle_ids) & set(candidate_ids)

    # =========================================================
    # WAM runtime: notable objects -> request -> placeholder policy
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

    def _update_wam_runtime_state(self, *, force: bool = False) -> None:
        if not bool(getattr(self, "_wam_enabled", True)):
            return
        step = int(getattr(self, "_time_step", 0))
        if not force and int(getattr(self, "_wam_last_update_step", -1)) == step:
            return

        route_points = self._wam_reference_route_points()
        object_states = self._wam_build_object_states()
        self._wam_object_states = object_states
        self._wam_notable_records = select_notable_objects(
            object_states,
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
        self._wam_policy = build_placeholder_policy(
            request=self._wam_coop_request,
            candidate_vehicle_ids=self.coop_participant_ids,
            uplink_bps=float(getattr(self._default_net_res, "uplink_bps", 0.0)),
            frequency_steps=int(self.comm_period),
            default_modality=str(self._wam_default_modality),
        )
        if self._wam_build_graph:
            self._build_wam_graph()
        self._wam_last_update_step = step
        if should_log_periodic(step, int(get_runtime_logging_config()["step_debug_interval"]), logger=V2V_LOGGER):
            V2V_LOGGER.debug(
                "WAM step=%d notable=%s max_uncertainty=%.3f triggered=%s selected=%s",
                step,
                [record.object_state.actor_id for record in self._wam_notable_records],
                self._wam_max_uncertainty(),
                self._wam_coop_request is not None,
                list(self._wam_policy.selected_vehicle_ids),
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
        overhead = int(getattr(self.latency_model, "overhead_bytes", 64))
        return float(max(int(n_objects), 0) * OBJECT_STATE_DIM * 4 + overhead)

    def _wam_bev_payload_bytes(self) -> float:
        return float(int(self._wam_graph_bev_channels) * int(self._wam_graph_bev_size) ** 2)

    def _build_wam_graph(self) -> None:
        """Assemble the policy-conditioned hetero graph for the current ``π_t`` (§7)."""
        policy = self._wam_policy
        objects = list(getattr(self, "_wam_object_states", []))
        selected = [int(vid) for vid in policy.selected_vehicle_ids]

        ego = self._wam_vehicle_node_input(
            self.ego, is_ego=True, agent_slot=0, route_xy=self._wam_route_xy()
        )
        ego_visible = [s for s in objects if bool(s.visible_to_ego)]
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

        collaborators = []
        slot = 1
        out_degree = max(len(selected), 1)
        for vid in selected:
            actor = self._get_group_member_actor(vid)
            if actor is None:
                continue
            collaborators.append(self._wam_vehicle_node_input(actor, is_ego=False, agent_slot=slot))
            slot += 1
            modality = str(policy.modality_by_vehicle.get(vid, self._wam_default_modality))
            if modality == "bev":
                observed_ids = ()
                payload = self._wam_bev_payload_bytes()
            else:
                observed_ids = tuple(int(s.actor_id) for s in objects if int(vid) in s.visible_to_collaborators)
                payload = self._wam_objlist_payload_bytes(len(observed_ids))
            latency_s = float(
                self.latency_model.compute_latency_s(
                    sender=actor,
                    receiver=self.ego,
                    payload_size_bytes=int(payload),
                    sender_res=self._veh_net_res.get(int(vid), self._default_net_res),
                    receiver_res=self._default_net_res,
                    out_degree=out_degree,
                    in_degree=out_degree,
                )
            )
            freshness = math.exp(-float(self._wam_graph_gamma_freshness) * latency_s)
            observations.append(
                ObservationNodeInput(
                    vehicle_id=int(vid),
                    modality=modality,
                    observed_object_ids=observed_ids,
                    payload_bytes=payload,
                    latency_s=latency_s,
                    freshness=freshness,
                    quality=1.0,
                    sample_age_s=0.0,
                )
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
            policy=policy,
            spec=spec,
            notable_ids=notable_ids,
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
        self._update_wam_runtime_state()
        candidate_ids = set(self.coop_participant_ids)
        # Collaboration policy selects the subset that shares with the ego this step.
        selected = set(self._select_collaborators(candidate_ids)) & candidate_ids
        self.selected_collaborators = selected
        if not selected:
            return

        # Communicating members = ego + the policy-selected collaborators (full-mesh).
        member_ids = [int(self.ego.id)] + sorted(selected)
        actor_map = self._build_group_actor_map()
        fixed_dt = float(self._world._settings.fixed_delta_seconds)
        degree = max(len(member_ids) - 1, 0)  # contention degree for this round
        enqueued_count = 0
        sender_ids = set()
        total_payload_bytes = 0

        for sender_id in member_ids:
            sender = actor_map.get(int(sender_id))
            if sender is None:
                continue
            sender_ids.add(int(sender_id))
            payload = self._make_payload(sender)
            payload_bytes = _tx_bytes_for_latency(
                payload,
                overhead_bytes=getattr(self.latency_model, "overhead_bytes", 64),
            )
            for receiver_id in member_ids:
                if int(receiver_id) == int(sender_id):
                    continue
                receiver = actor_map.get(int(receiver_id))
                if receiver is None:
                    continue
                sender_res = self._veh_net_res.get(int(sender_id), self._default_net_res)
                receiver_res = self._veh_net_res.get(int(receiver_id), self._default_net_res)
                latency_s = self.latency_model.compute_latency_s(
                    sender=sender,
                    receiver=receiver,
                    payload_size_bytes=payload_bytes,
                    sender_res=sender_res,
                    receiver_res=receiver_res,
                    out_degree=max(degree, 1),
                    in_degree=max(degree, 1),
                )
                self._enqueue_message(
                    group_id=GROUP_ID,
                    sender_id=int(sender_id),
                    receiver_id=int(receiver_id),
                    payload=payload,
                    payload_bytes=payload_bytes,
                    latency_s=latency_s,
                    distance_m=_dist_m(sender, receiver),
                    fixed_dt=fixed_dt,
                )
                enqueued_count += 1
                total_payload_bytes += int(payload_bytes)
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(int(self._time_step), int(runtime_cfg["step_debug_interval"]), logger=V2V_LOGGER):
            V2V_LOGGER.debug(
                "Communication round step=%d candidates=%s selected=%s senders=%s enqueued=%d in_flight=%d payload_bytes=%d",
                self._time_step,
                sorted(candidate_ids),
                sorted(selected),
                sorted(sender_ids),
                enqueued_count,
                len(self._in_flight),
                total_payload_bytes,
            )

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
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(current_step, int(runtime_cfg["step_debug_interval"]), logger=V2V_LOGGER):
            V2V_LOGGER.debug(
                "Delivered messages step=%d delivered=%d remaining_in_flight=%d ego_received=%d",
                current_step,
                delivered_count,
                len(self._in_flight),
                len(self._received.get(int(self.ego.id), deque())),
            )

    # =========================================================
    # Graph info construction
    # =========================================================

    def _build_graph_info(self) -> Dict[str, Any]:
        ego_feature = self.payload_fn(
            self.ego,
            self.obs,
            self.feature_size,
        )
        msgs = self._received.get(int(self.ego.id), deque())
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
            msg_count = len(self._received.get(int(self.ego.id), deque()))
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
