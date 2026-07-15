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
  4. ``on_step``: ``self._update_group_observations(); self._deliver_comm_messages(step);
     self._update_wam_runtime_state(); self._update_policy_lifecycle(step);
     self._stream_comm_messages(step)``.
  5. ``step`` / ``reset``: merge ``self._merge_step_info(info, action)`` /
     ``self._build_reset_info()`` into the returned info dict.

The host env must provide (all already present on every ``CarlaBaseEnv``/``CarlaWptEnv``):
``self.ego``, ``self.obs``, ``self._world``, ``self._time_step``, ``self.get_state()``,
and (for graph reset info) ``self.get_wpt_dist``. The task config must provide a
``group_observation`` block (camera+collision) and may provide ``communication`` /
``graph`` / ``feature_size`` / ``coop_participation_prob`` keys (all default-tolerant).
"""

from __future__ import annotations

from collections import deque
from dataclasses import is_dataclass
from pathlib import Path
import logging
import math
from typing import Any, Deque, Dict, List, Optional

import carla
import numpy as np
import torch
from runtime_logging import get_runtime_logger, get_runtime_logging_config, log_key_event, should_log_periodic

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
    OBJECT,
    OBJECT_STATE_DIM,
    BevSpec,
    CoverageConfig,
    GraphBuildSpec,
    ObjectState,
    ObservationNodeInput,
    VEHICLE,
    VehicleNodeInput,
    WAMPolicy,
    build_coverage_raster,
    rasterize_bev,
    build_wam_hetero_graph,
    detection_confidence,
    fuse_injected_objects,
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
        self._last_logged_policy_id: Optional[int] = None
        self._last_logged_request_signature = None
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
        self._wam_bev_payload_mode = str(getattr(wam_cfg, "bev_payload_mode", "feature")).lower()
        self._wam_bev_feature_dim = int(getattr(wam_cfg, "bev_feature_dim", 256))
        self._wam_bev_feature_dtype_bytes = int(getattr(wam_cfg, "bev_feature_dtype_bytes", 4))
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
        self._wam_graph_fusion_mode = str(getattr(graph_wam_cfg, "fusion_mode", "inject")).lower()
        self._wam_bev_spec = BevSpec(size=self._wam_graph_bev_size, range_m=self._wam_graph_bev_range_m)
        self._wam_graph_net = None
        self._wam_graph_embeddings = None
        stage1_cfg = getattr(wam_cfg, "stage1", None)
        self._wam_predictor_mode = str(getattr(wam_cfg, "predictor_mode", "rule")).lower()
        self._wam_predictor_checkpoint = getattr(wam_cfg, "predictor_checkpoint", None)
        self._wam_predictor_device = str(getattr(wam_cfg, "predictor_device", "auto"))
        self._wam_predictor_uncertainty_source = str(
            getattr(wam_cfg, "predictor_uncertainty_source", "notable_weighted_trace")
        )
        self._wam_predictor_history_window = int(
            getattr(wam_cfg, "predictor_history_window", getattr(stage1_cfg, "history_window", 4))
        )
        sample_period_s = float(
            getattr(
                stage1_cfg,
                "sample_period_s",
                getattr(getattr(self._config, "communication", None), "sensor_period_s", self._comm_config.dt),
            )
        )
        self._wam_predictor_sample_period_steps = max(1, int(round(sample_period_s / float(self._comm_config.dt))))
        self._wam_predictor_model = None
        self._wam_predictor_loaded_path = None
        self._wam_predictor_device_resolved = None
        # Stage-2 UWM policy sampler (policy_sampler_mode="stage2"): the BS proposes the collaboration
        # policy from the trained Unified World Model each Td instead of rule/random sampling.
        self._wam_uwm_checkpoint = getattr(wam_cfg, "uwm_checkpoint", None)
        self._wam_uwm_candidates = int(getattr(wam_cfg, "uwm_candidates", 4))
        self._wam_uwm_model = None
        self._wam_uwm_loaded_path = None
        # Lyapunov world-action policy sampler (policy_sampler_mode="lyapunov", V2X paper Sec IV). Written but
        # NOT live-verified (CARLA); the offline pipeline (scripts/run_wam_lyapunov_offline.py) is validated.
        self._wam_lyap_config_node = getattr(wam_cfg, "lyapunov", None)
        self._wam_lyapunov_scheduler = None
        self._wam_lyap_pending_segments = []
        self._wam_lyap_reference = None
        self._wam_lyap_last_ctx = None
        random_policy_cfg = getattr(wam_cfg, "random_policy", None)
        self._wam_policy_sampler_mode = str(getattr(wam_cfg, "policy_sampler_mode", "request_all")).lower()
        # Dreamer online sampler (P4): trained world-model + actor-critic checkpoints.
        self._wam_dreamer_world_model = getattr(wam_cfg, "dreamer_world_model", None)
        self._wam_dreamer_actor_critic = getattr(wam_cfg, "dreamer_actor_critic", None)
        self._wam_dreamer_policy = None
        self._wam_dreamer_scorer = None
        self._wam_dreamer_lyap = None
        self._wam_dreamer_prev_sub = None
        self._wam_random_policy_local_prob = float(getattr(random_policy_cfg, "local_prob", 0.2))
        self._wam_random_policy_counts = self._as_config_list(
            getattr(random_policy_cfg, "collaborator_counts", (1, "all")),
            default=(1, "all"),
        )
        self._wam_random_policy_modalities = self._normalize_modality_options(
            getattr(random_policy_cfg, "modalities", None)
        )
        self._wam_random_policy_bandwidth_ratios = tuple(
            float(np.clip(float(v), 0.0, 1.0))
            for v in self._as_config_list(getattr(random_policy_cfg, "bandwidth_ratios", (1.0,)), default=(1.0,))
        ) or (1.0,)
        self._wam_random_policy_respect_request = bool(getattr(random_policy_cfg, "respect_request", False))
        self._wam_random_policy_duration_grid = tuple(
            max(1, int(v))
            for v in self._as_config_list(getattr(random_policy_cfg, "duration_grid", None), default=())
        ) or (int(self._comm_config.policy_duration_steps),)
        coverage_cfg = getattr(wam_cfg, "coverage", None)
        self._wam_coverage_enabled = bool(getattr(coverage_cfg, "enabled", True))
        self._wam_coverage_config = CoverageConfig(
            past_route_distance_m=float(getattr(coverage_cfg, "past_route_distance_m", 10.0)),
            future_route_distance_m=float(getattr(coverage_cfg, "future_route_distance_m", 40.0)),
            corridor_width_m=float(getattr(coverage_cfg, "corridor_width_m", 8.0)),
            coverage_distance_scale_m=float(getattr(coverage_cfg, "coverage_distance_scale_m", 20.0)),
            route_risk_distance_scale_m=float(getattr(coverage_cfg, "route_risk_distance_scale_m", 20.0)),
            u_prior=float(getattr(coverage_cfg, "u_prior", 1.0)),
            freshness_gamma=float(getattr(coverage_cfg, "freshness_gamma", getattr(self, "_wam_graph_gamma_freshness", 5.0))),
        )
        self._wam_uncertainty_alpha_motion = float(getattr(coverage_cfg, "alpha_motion", 1.0))
        self._wam_uncertainty_beta_coverage = float(getattr(coverage_cfg, "beta_coverage", 1.0))
        # [0,1]-normalized uncertainty (reported alongside the raw total; does not drive the threshold):
        # motion_norm = 1 - exp(-motion / sigma_scale); total_norm = alpha_norm*motion_norm + (1-alpha_norm)*coverage.
        self._wam_uncertainty_sigma_scale = float(getattr(coverage_cfg, "sigma_scale", 4.0))
        self._wam_uncertainty_alpha_norm = float(getattr(coverage_cfg, "alpha_norm", 0.5))
        # Soft-gate the (no-GT) notable_prob weight: w = sigmoid(k*(p - tau)). k=0 disables (raw prob).
        self._wam_uncertainty_notable_gate_k = float(getattr(coverage_cfg, "notable_gate_k", 20.0))
        self._wam_uncertainty_notable_gate_threshold = float(getattr(coverage_cfg, "notable_gate_threshold", 0.5))
        # Denominator floor (units: notable objects): no confident notable -> motion ~0 instead of the
        # gate-cancelling mean trace. 0 disables (plain weighted mean).
        self._wam_uncertainty_notable_mass_floor = float(getattr(coverage_cfg, "notable_mass_floor", 1.0))
        seed = getattr(random_policy_cfg, "seed", None)
        self._wam_policy_rng = np.random.default_rng(None if seed is None else int(seed))

        self._reset_wam_runtime_state()

    def _as_config_list(self, value, *, default=()):
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _normalize_modality_options(self, raw_options) -> tuple:
        options = []
        for raw in self._as_config_list(raw_options, default=(self._collaborator_modalities,)):
            if isinstance(raw, str):
                parts = [p.strip() for p in raw.replace("+", ",").split(",") if p.strip()]
                modalities = tuple(parts) or (raw,)
            else:
                modalities = tuple(str(m) for m in raw)
            if modalities:
                options.append(modalities)
        return tuple(options) or (tuple(self._collaborator_modalities),)

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
        self._last_logged_policy_id = None
        self._last_logged_request_signature = None
        self._reset_wam_runtime_state()
        log_key_event(
            V2V_LOGGER,
            logging.INFO,
            "V2V runtime reset coop_participation_prob=%.3f Td=%d Ts=%d Tw=%d Ta=%d bev_payload_mode=%s bev_payload_bytes=%.0f",
            float(getattr(self, "coop_participation_prob", 0.5)),
            int(self._comm_config.policy_duration_steps),
            int(self._comm_config.sensor_period_steps),
            int(self._comm_config.prediction_window_steps),
            int(self._comm_config.action_period_steps),
            str(getattr(self, "_wam_bev_payload_mode", "feature")),
            float(self._wam_bev_payload_bytes()),
        )

    def _reset_wam_runtime_state(self) -> None:
        self._wam_notable_records = []
        self._wam_object_states = []
        self._wam_object_states_step = -1  # last step the (per-step) visibility scan ran
        self._wam_motion_predictions = {}
        self._wam_coop_request = None
        # Per-episode Lyapunov reset: clear the virtual queue Z / link backlogs so they do not carry
        # over across episodes (cached scheduler persists across env.reset).
        self._wam_lyap_pending_segments = []
        self._wam_lyap_reference = None
        self._wam_lyap_last_ctx = None
        sched = getattr(self, "_wam_lyapunov_scheduler", None)
        if sched is not None and hasattr(sched, "reset"):
            sched.reset()
        # Per-episode Dreamer reset: clear the RSSM posterior carry, the shadow queue, and the prev action.
        dreamer = getattr(self, "_wam_dreamer_policy", None)
        if dreamer is not None and hasattr(dreamer, "reset"):
            dreamer.reset()
        self._wam_dreamer_lyap = None  # reloaded lazily on next decision (fresh queue)
        self._wam_dreamer_prev_sub = None
        self._wam_graph = None
        self._wam_graph_step = -1
        graph_window_len = max(int(getattr(self, "_wam_predictor_history_window", 4)), 0) + 1
        self._wam_graph_window: Deque[Any] = deque(maxlen=graph_window_len)
        self._wam_graph_window_last_step = -1
        self._wam_slot_state_history: Dict[int, Any] = {}
        self._wam_received_message_cache: Dict[Any, Any] = {}
        self._wam_active_policy_by_step: Dict[int, Optional[int]] = {}
        self._wam_slot_history_last_step = -1
        self._wam_graph_embeddings = None
        self._wam_coverage_raster = None
        self._wam_coverage_step = -1
        self._wam_ego_pose_history: Deque[Any] = deque(maxlen=512)
        self._wam_uncertainty_breakdown = {
            "motion_uncertainty": 0.0,
            "coverage_uncertainty": 0.0,
            "total_uncertainty": 0.0,
            "route_coverage_ratio": 0.0,
            "route_coverage_quality_mean": 0.0,
            "poor_coverage_risk_mean": 0.0,
        }
        self._wam_checkpoint_prediction_stats = self._empty_wam_checkpoint_prediction_stats()
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
        # The vehicle was just spawned this step (cooperative candidates register incrementally
        # during reset_spawn), so drop the stale step cache before the observer takes its first
        # observation -- otherwise its own polygon/transform is missing when the birdeye renderer
        # draws it as ego. See WorldManager.invalidate_step_cache.
        self._world.invalidate_step_cache()
        self.group_obs[int(vehicle.id)], _ = observer.get_observation(self.get_state())

    def _update_group_observations(self) -> None:
        # Only participating (collaborating) vehicles' observations are consumed by
        # V2V communication and the policy graph, so only refresh those.
        dead_ids: List[int] = []
        for actor in self.group_vehs:
            aid = int(actor.id)
            if aid not in self.coop_participant_ids:
                continue
            # Defensive only -- this should now be rare. Collaborators used to be despawned
            # routinely right after spawn because autopilot was enabled inside the asynchronous
            # reset window, so they drove off unsupervised and crashed before step 0; that root
            # cause is fixed in WorldManager._request_autopilot. Treat a drop as a signal worth
            # investigating (scripts/trace_scenario_vehicles.py), not as normal attrition.
            if not bool(getattr(actor, "is_alive", True)):
                dead_ids.append(aid)
                continue
            observer = self._other_observers.get(aid)
            if observer is not None:
                try:
                    self.group_obs[aid], _ = observer.get_observation(self.get_state())
                except RuntimeError:
                    # Destroyed mid-observation (aliveness raced the render); drop it.
                    dead_ids.append(aid)
        for aid in dead_ids:
            self._drop_cooperative_vehicle(aid)

    def _drop_cooperative_vehicle(self, actor_id: int) -> None:
        """Remove a no-longer-alive collaborator from all cooperative-group state.

        Keeps the episode running with the surviving collaborators: the dropped vehicle stops
        being observed, streamed over V2V, and included in the policy graph. This is a safety net,
        not an expected event -- see the note in :meth:`_update_group_observations`. A run that
        keeps dropping collaborators is silently shrinking the cooperative pool, which quietly
        weakens every V2V result computed from it.
        """
        actor_id = int(actor_id)
        self.coop_participant_ids.discard(actor_id)
        self.group_obs.pop(actor_id, None)
        self._actor_cache.pop(actor_id, None)
        observer = self._other_observers.pop(actor_id, None)
        if observer is not None:
            try:
                observer.destroy()
            except Exception:  # noqa: BLE001 - observer teardown must not break the step
                pass
        self.group_vehs = [v for v in self.group_vehs if int(v.id) != actor_id]
        group = self.groups.get(GROUP_ID)
        if group is not None:
            group.discard(actor_id)
        V2V_LOGGER.warning(
            "Dropped destroyed cooperative vehicle id=%s from group; %d collaborator(s) left. "
            "This is unexpected since the reset-window autopilot fix -- investigate rather than ignore.",
            actor_id,
            len(self.coop_participant_ids),
        )

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
        log_key_event(
            V2V_LOGGER,
            logging.INFO,
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
            log_key_event(
                V2V_LOGGER,
                logging.INFO,
                "Communication process initialized ego_id=%s dt=%.3f Td=%d Ts=%d Tw=%d Ta=%d",
                int(self.ego.id),
                float(self._comm_config.dt),
                int(self._comm_config.policy_duration_steps),
                int(self._comm_config.sensor_period_steps),
                int(self._comm_config.prediction_window_steps),
                int(self._comm_config.action_period_steps),
            )
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
        cand = [int(c) for c in candidates]
        # Shared spectrum: split the total bandwidth ratio equally across the |S| members (B_m = ratio/|S|).
        bandwidth_ratio = float(self._comm_bandwidth_ratio) / max(len(cand), 1)
        modalities = tuple(self._collaborator_modalities)
        return CommPolicy(
            policy_id=self._next_policy_id(),
            request_vehicle_id=int(self.ego.id),
            start_step=int(step),
            duration_steps=int(self._comm_config.policy_duration_steps),
            selected_collaborators=tuple(cand),
            modalities_by_vehicle={int(c): modalities for c in cand},
            bandwidth_by_vehicle={int(c): bandwidth_ratio for c in cand},
            reason="coop_request",
        )

    def _sample_random_policy_duration(self) -> int:
        """Sample the sub-action duration n (env steps; comm slot == 1 env step) from the grid."""
        grid = tuple(getattr(self, "_wam_random_policy_duration_grid", ())) or (int(self._comm_config.policy_duration_steps),)
        return int(grid[int(self._wam_policy_rng.integers(0, len(grid)))])

    def _random_policy_collaborator_count(self, n_candidates: int) -> int:
        """Sample |S| from ``random_policy.collaborator_counts`` ("0" -> local-only, "all" -> every candidate)."""
        options = tuple(getattr(self, "_wam_random_policy_counts", (1, "all"))) or (1, "all")
        choice = options[int(self._wam_policy_rng.integers(0, len(options)))]
        if isinstance(choice, str):
            text = choice.strip().lower()
            if text == "all":
                return int(n_candidates)
            try:
                return int(text)
            except ValueError:
                return int(n_candidates)
        return int(choice)

    def _sample_random_comm_policy(self, step: int, candidates: List[int]) -> CommPolicy:
        """Sample one real sub-action a=(S,B,D,n) and install it for its sampled duration n.

        This is not counterfactual enumeration: the sampled policy is installed into
        ``CommunicationProcess`` and the queues evolve under it until the policy expires.
        n is drawn from ``random_policy.duration_grid`` (env steps); local-only sub-actions
        ("0" in ``collaborator_counts`` or the ``local_prob`` mixture) sample n from the same grid.
        """
        candidates = sorted(int(c) for c in candidates)
        duration = self._sample_random_policy_duration()
        local_prob = float(np.clip(float(getattr(self, "_wam_random_policy_local_prob", 0.2)), 0.0, 1.0))
        if not candidates or float(self._wam_policy_rng.random()) < local_prob:
            return make_local_policy(
                policy_id=self._next_policy_id(),
                request_vehicle_id=int(self.ego.id),
                start_step=int(step),
                duration_steps=int(duration),
            )

        k = self._random_policy_collaborator_count(len(candidates))
        k = min(max(int(k), 0), len(candidates))
        if k <= 0:
            return make_local_policy(
                policy_id=self._next_policy_id(),
                request_vehicle_id=int(self.ego.id),
                start_step=int(step),
                duration_steps=int(duration),
            )

        selected = tuple(sorted(int(v) for v in self._wam_policy_rng.choice(candidates, size=k, replace=False)))
        modality_options = getattr(self, "_wam_random_policy_modalities", None)
        if not modality_options:
            modality_options = (tuple(getattr(self, "_collaborator_modalities", ("objlist",))),)
        modality_options = tuple(modality_options)
        modalities = tuple(modality_options[int(self._wam_policy_rng.integers(0, len(modality_options)))])
        ratios = tuple(getattr(self, "_wam_random_policy_bandwidth_ratios", (1.0,))) or (1.0,)
        bandwidth_ratio = float(ratios[int(self._wam_policy_rng.integers(0, len(ratios)))])
        bandwidth_ratio = float(np.clip(bandwidth_ratio, 0.0, 1.0))
        # Shared spectrum: split the sampled total ratio equally across the |S| members (B_m = ratio/|S|).
        bandwidth_ratio = bandwidth_ratio / max(len(selected), 1)

        return CommPolicy(
            policy_id=self._next_policy_id(),
            request_vehicle_id=int(self.ego.id),
            start_step=int(step),
            duration_steps=int(duration),
            selected_collaborators=selected,
            modalities_by_vehicle={int(c): modalities for c in selected},
            bandwidth_by_vehicle={int(c): bandwidth_ratio for c in selected},
            reason="random_duration",
        )

    def _load_wam_uwm(self):
        """Load (and cache) the Stage-2 UWM for ``policy_sampler_mode=stage2``."""
        ckpt = getattr(self, "_wam_uwm_checkpoint", None)
        if ckpt in (None, "", "null"):
            raise RuntimeError("wam.policy_sampler_mode=stage2 requires wam.uwm_checkpoint")
        path = Path(str(ckpt)).expanduser()
        if self._wam_uwm_model is not None and self._wam_uwm_loaded_path == str(path):
            return self._wam_uwm_model
        from .toolkit.wam import load_wam_uwm

        device = self._wam_predictor_device_resolved or self._resolve_wam_predictor_device()
        self._wam_predictor_device_resolved = device
        self._wam_uwm_model = load_wam_uwm(path, device=device)
        self._wam_uwm_loaded_path = str(path)
        V2V_LOGGER.info("WAM Stage-2 UWM loaded from %s (device=%s)", path, device)
        return self._wam_uwm_model

    def _wampolicy_to_commpolicy(self, step: int, wam_policy) -> CommPolicy:
        """Turn a generated :class:`WAMPolicy` into an installable :class:`CommPolicy`.

        Empty selection -> a local-only policy (the BS chooses not to cooperate this Td)."""
        selected = tuple(int(v) for v in getattr(wam_policy, "selected_vehicle_ids", ()))
        if not selected:
            return make_local_policy(
                policy_id=self._next_policy_id(), request_vehicle_id=int(self.ego.id),
                start_step=int(step), duration_steps=int(self._comm_config.policy_duration_steps),
            )
        modalities = {
            int(v): (str(wam_policy.modality_by_vehicle.get(int(v), "objlist")),) for v in selected
        }
        bandwidth = {
            int(v): float(wam_policy.bandwidth_by_vehicle.get(int(v), self._comm_bandwidth_ratio)) for v in selected
        }
        return CommPolicy(
            policy_id=self._next_policy_id(), request_vehicle_id=int(self.ego.id),
            start_step=int(step), duration_steps=int(self._comm_config.policy_duration_steps),
            selected_collaborators=selected, modalities_by_vehicle=modalities,
            bandwidth_by_vehicle=bandwidth, reason="stage2_uwm",
        )

    def _sample_stage2_comm_policy(self, step: int, candidates: List[int]) -> CommPolicy:
        """Propose the collaboration policy from the Stage-2 UWM, conditioned on the current request graph.

        Builds C^BS from ``self._wam_graph`` + the notable objects, runs ``propose_policies``, decodes the
        first candidate -> :class:`WAMPolicy` -> :class:`CommPolicy`. (Hybrid Stage-1-U^pi scoring across
        the k candidates is a future refinement; it would need a Stage-1 model loaded alongside.)"""
        candidates = sorted(int(c) for c in candidates)
        graph = getattr(self, "_wam_graph", None)
        if not candidates or graph is None:
            return make_local_policy(
                policy_id=self._next_policy_id(), request_vehicle_id=int(self.ego.id),
                start_step=int(step), duration_steps=int(self._comm_config.policy_duration_steps),
            )
        uwm = self._load_wam_uwm()
        device = self._wam_predictor_device_resolved or self._resolve_wam_predictor_device()
        notable_ids = [int(r.object_state.actor_id) for r in getattr(self, "_wam_notable_records", ())]
        from .toolkit.wam import decode_chunk_to_wampolicy

        with torch.no_grad():
            cond, tids = uwm.condition_tokens([graph.clone().to(device)], 0, notable_ids)
            cond = cond.unsqueeze(0); tids = tids.unsqueeze(0)
            mask = torch.ones(1, cond.shape[1], device=cond.device)
            m_max = int(uwm.flow.config.max_members)
            mm = torch.zeros(1, m_max, device=cond.device)
            mm[0, : min(len(candidates), m_max)] = 1.0
            k = max(1, int(getattr(self, "_wam_uwm_candidates", 4)))
            cand = uwm.flow.propose_policies(cond, tids, mask, n_candidates=k, member_mask=mm)
        wam_policy = decode_chunk_to_wampolicy(cand[0, 0], candidates, num_formats=int(uwm.flow.config.num_formats))
        V2V_LOGGER.info(
            "WAM Stage-2 proposed policy step=%d selected=%s modalities=%s",
            int(step), list(wam_policy.selected_vehicle_ids), dict(wam_policy.modality_by_vehicle),
        )
        return self._wampolicy_to_commpolicy(step, wam_policy)

    def _load_lyapunov_scheduler(self):
        """Load (and cache) the Lyapunov scheduler for ``policy_sampler_mode=lyapunov`` (Sec IV).

        ``U_φ`` = the Stage-1 perception model when ``predictor_mode=checkpoint`` (else a rule fallback). No
        Stage-2 UWM; candidate chunks come from the heuristic enumerator.
        """
        if self._wam_lyapunov_scheduler is not None:
            return self._wam_lyapunov_scheduler
        from .toolkit.wam import LyapunovScheduler, SchedulerConfig, WorldActionScorer

        node = self._wam_lyap_config_node

        def g(key, default):
            return getattr(node, key, default) if node is not None else default

        model = None
        if str(getattr(self, "_wam_predictor_mode", "rule")).lower() == "checkpoint":
            try:
                model = self._load_wam_predictor()
            except Exception as exc:  # unverified online path: fall back to the rule U_φ
                V2V_LOGGER.warning("WAM lyapunov: Stage-1 U_φ load failed (%s); using rule fallback", exc)
                model = None
        cfg = SchedulerConfig(
            lam=float(g("lam", 1.0)), c0=float(g("c0", 0.5)),
            budget_bandwidth=float(g("budget_bandwidth_ratio", 0.4)),
            F_max_slots=int(g("F_max_slots", 20)), n_min_slots=int(g("n_min_slots", 5)),
            B_max_ratio=float(g("B_max_ratio", 1.0)),
            bandwidth_grid=tuple(float(x) for x in self._as_config_list(g("bandwidth_grid", (0.2, 0.5, 0.8, 1.0)), default=(0.2, 0.5, 0.8, 1.0))),
            duration_grid=tuple(int(x) for x in self._as_config_list(g("duration_grid", (5, 10, 20)), default=(5, 10, 20))),
            j_max=int(g("j_max", 2)), eps_gap=float(g("eps_gap", 0.15)),
            t_min_slots=int(g("t_min_slots", 3)), T_a_slots=int(g("T_a_slots", 5)),
            ts_seconds=float(self._comm_config.dt),
            max_score_candidates=int(g("max_score_candidates", 200)),
            candidate_seed=int(g("candidate_seed", 0)),
            uwm_candidates=int(g("uwm_candidates", 4)),
        )
        device = self._wam_predictor_device_resolved or self._resolve_wam_predictor_device()
        self._wam_predictor_device_resolved = device
        scorer = WorldActionScorer(
            perception_model=model, alpha=float(g("alpha", 0.5)),
            sigma_scale=float(getattr(self, "_wam_uncertainty_sigma_scale", 4.0)),
            freshness_gamma=float(getattr(self, "_wam_graph_gamma_freshness", 5.0)),
            device=str(device),
        )
        # Stage-2 UWM action proposer W_θ (Phase 5): loaded when env.wam.uwm_checkpoint is set.
        uwm = None
        uwm_ckpt = getattr(self, "_wam_uwm_checkpoint", None)
        if uwm_ckpt not in (None, "", "null"):
            try:
                uwm = self._load_wam_uwm()
            except Exception as exc:  # proposer is best-effort; heuristics still cover the candidate set
                V2V_LOGGER.warning("WAM lyapunov: UWM proposer load failed (%s); heuristic-only", exc)
                uwm = None
        self._wam_lyapunov_scheduler = LyapunovScheduler(cfg, scorer, request_vehicle_id=int(self.ego.id), uwm=uwm)
        V2V_LOGGER.info("WAM lyapunov scheduler ready (U_φ=%s, proposer=%s, Λ=%.3g, B̄_bgt=%.3g)",
                        "checkpoint" if model is not None else "rule",
                        "UWM" if uwm is not None else "heuristic", cfg.lam, cfg.budget_bandwidth)
        return self._wam_lyapunov_scheduler

    def _build_rollout_context(self, step: int, candidates: List[int]):
        """Build the ``C^BS`` :class:`RolloutContext` from the current slot state (reuses ``_wam_stage1_slot_state``)."""
        from .toolkit.wam import GraphBuildSpec, RolloutContext

        state = self._wam_stage1_slot_state(int(step))
        ego = state["ego"]
        cand = set(int(c) for c in candidates)
        collaborators = tuple(c for c in state["collaborators"] if int(c.actor_id) in cand)
        ego_v0 = math.hypot(float(ego.vx), float(ego.vy))
        return RolloutContext(
            ego=ego, collaborators=collaborators, objects=tuple(state["live_states"]),
            route_xy=tuple(state["route_xy"]), notable_ids=tuple(state["notable_ids"]),
            link_rate_fn=self._link_rate_bps,
            graph_spec=GraphBuildSpec(route_waypoints=int(self._wam_graph_route_waypoints)),
            bev_spec=self._wam_bev_spec, ego_v0=float(ego_v0),
            past_route_xy=tuple(state.get("past_route_xy", ())),
            dt_seconds=float(self._comm_config.dt),
            sensor_period_steps=int(self._comm_config.sensor_period_steps),
        )

    def _wam_lyap_realized_uncertainty(self, step: int):
        """Realized normalized ``U`` for the eq-8 mismatch trigger (from the uncertainty breakdown)."""
        breakdown = getattr(self, "_wam_uncertainty_breakdown", None)
        if not breakdown:
            return None
        value = breakdown.get("total_uncertainty_norm")
        return float(value) if value is not None else None

    def _wam_lyap_next_segment(self, step: int) -> Optional[CommPolicy]:
        """Pop the next buffered chunk segment whose start has arrived (mid-chunk sub-action switch)."""
        pending = getattr(self, "_wam_lyap_pending_segments", [])
        if pending and int(pending[0].start_step) <= int(step):
            return pending.pop(0)
        return None

    def _sample_lyapunov_comm_policy(self, step: int, candidates: List[int]) -> CommPolicy:
        """Solve (P2), install the first chunk segment, buffer the rest, store the reference trajectory.

        Defensive: any failure in the (unverified) online scheduler path falls back to a local-only policy."""
        from dataclasses import replace

        candidates = sorted(int(c) for c in candidates)
        try:
            sched = self._load_lyapunov_scheduler()
            ctx = self._build_rollout_context(int(step), candidates)
            self._wam_lyap_last_ctx = ctx
            segments, _breakdown, chunk, _roll = sched.plan(ctx, int(step))
            # re-id segments with the mixin's policy-id counter (the scheduler used its own)
            segments = [replace(s, policy_id=self._next_policy_id()) for s in segments]
            self._wam_lyap_reference = sched.ref
            self._wam_lyap_pending_segments = list(segments[1:])
            V2V_LOGGER.info(
                "WAM lyapunov step=%d installed chunk J=%d F=%d first_selected=%s",
                int(step), chunk.num_subepochs, chunk.horizon_slots,
                list(segments[0].selected_collaborators),
            )
            return segments[0]
        except Exception as exc:
            V2V_LOGGER.warning("WAM lyapunov sampler failed (%s); installing local-only", exc)
            self._wam_lyap_pending_segments = []
            return make_local_policy(
                policy_id=self._next_policy_id(), request_vehicle_id=int(self.ego.id),
                start_step=int(step), duration_steps=int(self._comm_config.policy_duration_steps),
            )

    def _load_dreamer_policy(self):
        """Return the P4 Dreamer sampler. Policy + safety scorer are cached; the shadow queue is
        (re)created per episode (``_wam_dreamer_lyap`` is cleared to ``None`` on ``_reset_wam_runtime_state``)."""
        from .toolkit.wam import (
            LyapunovConfig, LyapunovState, WorldActionScorer, load_dreamer_policy,
        )

        node = self._wam_lyap_config_node

        def g(key, default):
            return getattr(node, key, default) if node is not None else default

        if self._wam_dreamer_policy is None:
            device = self._wam_predictor_device_resolved or self._resolve_wam_predictor_device()
            self._wam_predictor_device_resolved = device
            self._wam_dreamer_policy = load_dreamer_policy(
                str(self._wam_dreamer_world_model), str(self._wam_dreamer_actor_critic), device=str(device)
            )
            model = None
            if str(getattr(self, "_wam_predictor_mode", "rule")).lower() == "checkpoint":
                try:
                    model = self._load_wam_predictor()
                except Exception as exc:  # unverified online path
                    V2V_LOGGER.warning("WAM dreamer: Stage-1 U_φ load failed (%s); rule fallback for safety P2", exc)
                    model = None
            self._wam_dreamer_scorer = WorldActionScorer(
                perception_model=model, alpha=float(g("alpha", 0.5)),
                sigma_scale=float(getattr(self, "_wam_uncertainty_sigma_scale", 4.0)),
                freshness_gamma=float(getattr(self, "_wam_graph_gamma_freshness", 5.0)), device=str(device),
            )
            V2V_LOGGER.info("WAM dreamer sampler ready (world_model=%s, actor=%s)",
                            self._wam_dreamer_world_model, self._wam_dreamer_actor_critic)
        if self._wam_dreamer_lyap is None:  # fresh queue at episode start
            self._wam_dreamer_lyap = LyapunovState(LyapunovConfig(
                lam=float(g("lam", 1.0)), c0=float(g("c0", 0.5)),
                budget_bandwidth=float(g("budget_bandwidth_ratio", 0.4)), ts_seconds=float(self._comm_config.dt)))
            self._wam_dreamer_prev_sub = None
        return self._wam_dreamer_policy

    def _sample_dreamer_comm_policy(self, step: int, candidates: List[int]) -> CommPolicy:
        """Dreamer actor picks ``(S,B,D)`` from the RSSM posterior; local-only safety keeps no-degradation.

        Defensive: any failure in this (unverified) online path falls back to a local-only policy."""
        import torch

        from .toolkit.wam import MDPConfig, WAMPolicy, subaction_cost
        from .toolkit.wam.action_chunk import SubAction, sub_action_to_comm_policy
        from .toolkit.wam.graph import OBJECT, VEHICLE
        from .toolkit.wam.stage1_policy import build_stage1_policy_graph

        candidates = sorted(int(c) for c in candidates)
        try:
            policy_obj = self._load_dreamer_policy()
            ctx = self._build_rollout_context(int(step), candidates)
            model = self._load_wam_predictor()
            device = self._wam_predictor_device_resolved
            hidden = int(model.config.hidden_dim)

            # observation graph embedding: build the graph under the currently-observed cooperation state
            sel_ids = set(int(x) for x in getattr(self, "selected_collaborators", set()))
            sel = [c for c in ctx.collaborators if int(c.actor_id) in sel_ids]
            obs_policy = WAMPolicy(
                selected_vehicle_ids=tuple(int(c.actor_id) for c in sel),
                modality_by_vehicle={int(c.actor_id): "objlist" for c in sel},
                bandwidth_by_vehicle={int(c.actor_id): 1.0 for c in sel},
                frequency_steps=1, reason="dreamer_obs")
            graph = build_stage1_policy_graph(
                ego=ctx.ego, collaborators=list(sel), objects=list(ctx.objects),
                policy=obs_policy, spec=ctx.graph_spec, notable_ids=ctx.notable_ids).to(device)
            with torch.no_grad():
                H = model.graph_net(graph)

            def pool(key):
                return H[key].mean(0) if (key in H and H[key].numel()) else torch.zeros(hidden, device=device)

            graph_embed = torch.cat([pool(VEHICLE), pool(OBJECT)])
            lyap = self._wam_dreamer_lyap
            scalars = torch.tensor(
                [lyap.z.value, lyap.total_backlog(), ctx.ego_v0, float(len(candidates)), float(len(ctx.notable_ids))],
                dtype=torch.float32)
            is_first = self._wam_dreamer_prev_sub is None
            sub = policy_obj.act(graph_embed, scalars, candidate_ids=candidates, is_first=is_first)

            # local-only safety: install whichever of {actor sub-action, local-only} has lower P2 cost
            cfg = MDPConfig(lam=lyap.config.lam, c0=lyap.config.c0, budget_bandwidth=lyap.config.budget_bandwidth)
            c_sub, roll_sub, _ = subaction_cost(ctx, sub, self._wam_dreamer_scorer, cfg,
                                                z=lyap.z.value, link_backlogs=lyap.backlogs())
            local = SubAction(selected=(), duration_slots=int(sub.duration_slots))
            c_loc, roll_loc, _ = subaction_cost(ctx, local, self._wam_dreamer_scorer, cfg,
                                                z=lyap.z.value, link_backlogs=lyap.backlogs())
            if c_loc < c_sub:
                sub, roll = local, roll_loc
            else:
                roll = roll_sub

            # evolve the queue over the installed sub-action's slots (frame totals spread evenly)
            nslot = max(int(sub.duration_slots), 1)
            svc = {int(k): float(v) / nslot for k, v in roll.per_member_predicted_service_bits.items()}
            arr = {int(k): float(v) / nslot for k, v in roll.per_member_predicted_load_bits.items()}
            for _ in range(nslot):
                lyap.advance_slot(per_member_service_bits=svc, per_member_arrival_bits=arr,
                                  allocated_bandwidth=float(sub.total_bandwidth()), budget_bandwidth=cfg.budget_bandwidth)
            self._wam_dreamer_prev_sub = sub
            V2V_LOGGER.info("WAM dreamer step=%d selected=%s bw=%s local=%s",
                            int(step), list(sub.selected), dict(sub.bandwidth_by_vehicle), bool(sub.is_local_only))
            return sub_action_to_comm_policy(sub, policy_id=self._next_policy_id(),
                                             request_vehicle_id=int(self.ego.id), start_step=int(step))
        except Exception as exc:
            V2V_LOGGER.warning("WAM dreamer sampler failed (%s); installing local-only", exc)
            return make_local_policy(
                policy_id=self._next_policy_id(), request_vehicle_id=int(self.ego.id),
                start_step=int(step), duration_steps=int(self._comm_config.policy_duration_steps))

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
        if int(policy.policy_id) != int(getattr(self, "_last_logged_policy_id", -1) or -1):
            self._last_logged_policy_id = int(policy.policy_id)
            log_key_event(
                V2V_LOGGER,
                logging.INFO,
                "WAM policy installed id=%d step=%d end=%d reason=%s local_only=%s selected=%s modalities=%s bandwidth=%s",
                int(policy.policy_id),
                int(policy.start_step),
                int(policy.end_step),
                str(policy.reason),
                bool(policy.is_local_only),
                list(int(v) for v in policy.selected_collaborators),
                {int(k): tuple(v) for k, v in policy.modalities_by_vehicle.items()},
                {int(k): float(v) for k, v in policy.bandwidth_by_vehicle.items()},
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
        request = getattr(self, "_wam_coop_request", None)
        candidates = sorted(int(v) for v in getattr(self, "coop_participant_ids", set()))
        sampler_mode = str(getattr(self, "_wam_policy_sampler_mode", "request_all")).lower()
        if sampler_mode == "local_only":
            # Ego-only baseline: never cooperate (for the vs-lyapunov comparison, Phase 3).
            if active is not None and active.is_local_only and active.active_at(step):
                return
            policy = make_local_policy(
                policy_id=self._next_policy_id(), request_vehicle_id=int(self.ego.id),
                start_step=int(step), duration_steps=int(self._comm_config.policy_duration_steps),
            )
            proc.set_policy(policy, int(step))
            self._sync_policy_views(policy)
            return

        if sampler_mode == "random_duration":
            if active is not None and active.active_at(step):
                return  # sampled policies, including local-only, persist for the full Td
            if bool(getattr(self, "_wam_random_policy_respect_request", False)) and request is None:
                policy = make_local_policy(
                    policy_id=self._next_policy_id(),
                    request_vehicle_id=int(self.ego.id),
                    start_step=int(step),
                    duration_steps=self._sample_random_policy_duration(),
                )
            else:
                policy = self._sample_random_comm_policy(step, candidates)
            proc.set_policy(policy, int(step))
            self._sync_policy_views(policy)
            return

        if sampler_mode == "stage2":
            if active is not None and active.active_at(step):
                return  # the UWM-proposed policy runs its full Td
            policy = self._sample_stage2_comm_policy(step, candidates)
            proc.set_policy(policy, int(step))
            self._sync_policy_views(policy)
            return

        if sampler_mode == "dreamer":
            # trained Dreamer actor picks (S,B,D) per decision epoch; the sub-action runs its short duration
            if active is not None and active.active_at(step):
                return
            policy = self._sample_dreamer_comm_policy(int(step), candidates)
            proc.set_policy(policy, int(step))
            self._sync_policy_views(policy)
            return

        if sampler_mode == "lyapunov":
            sched = self._load_lyapunov_scheduler()
            # evolve the queues (Q_m, Z) for the just-elapsed slot using the last-planned context
            if getattr(self, "_wam_lyap_last_ctx", None) is not None and sched.chunk is not None:
                try:
                    sched.observe_slot(self._wam_lyap_last_ctx, int(step))
                except Exception:  # unverified online path
                    pass
            if sched.is_decision_epoch(int(step), self._wam_lyap_realized_uncertainty):
                policy = self._sample_lyapunov_comm_policy(int(step), candidates)  # plan (P2) + install first segment
                proc.set_policy(policy, int(step))
                self._sync_policy_views(policy)
            elif active is None or not active.active_at(step):
                nxt = self._wam_lyap_next_segment(int(step))  # mid-chunk sub-action switch
                if nxt is not None:
                    proc.set_policy(nxt, int(step))
                    self._sync_policy_views(nxt)
            return

        if active is not None and not active.is_local_only and active.active_at(step):
            return  # an active cooperative policy runs for its full duration

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

    def _wam_comm_slot_stats(self, step: int) -> Dict[int, Dict[str, float]]:
        """Per-member queue/link primitives at env step ``step`` (V2X (P2) inputs; {} if local-only)."""
        proc = self._ensure_comm_process()
        policy = proc.policy
        if policy is None or policy.is_local_only:
            return {}
        distances: Dict[int, float] = {}
        for member_id in policy.selected_collaborators:
            try:
                actor = self._get_group_member_actor(int(member_id))
            except RuntimeError:
                actor = None
            if actor is not None:
                distances[int(member_id)] = float(_dist_m(actor, self.ego))
        return proc.comm_stats(int(step), link_rate_bps=self._link_rate_bps, distance_by_sender=distances)

    def _deliver_comm_messages(self, step: int) -> None:
        """Deliver messages whose transmission completed by this simulation step."""
        proc = self._ensure_comm_process()
        proc.deliver(int(step))

    def _stream_comm_messages(self, step: int) -> None:
        """Generate collaborator messages at sensor ticks under the active policy."""
        proc = self._ensure_comm_process()
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
        if emitted:
            log_key_event(
                V2V_LOGGER,
                logging.INFO,
                "Communication emitted step=%d policy=%s messages=%d senders=%s payload_bytes=%s recv_steps=%s",
                int(step),
                int(policy.policy_id),
                len(emitted),
                [int(m.sender_id) for m in emitted],
                [int(m.payload_size) for m in emitted],
                [int(m.t_recv) for m in emitted],
            )
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

    def _run_comm_step(self, step: int) -> None:
        """One simulation step of the streaming comm process: deliver, then stream at sensor ticks."""
        self._deliver_comm_messages(step)
        self._stream_comm_messages(step)

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

    def _record_wam_ego_pose_history(self, step: int) -> None:
        ego = getattr(self, "ego", None)
        if ego is None:
            return
        tf = ego.get_transform()
        pose = (int(step), float(tf.location.x), float(tf.location.y), float(tf.rotation.yaw))
        hist = getattr(self, "_wam_ego_pose_history", None)
        if hist is None:
            self._wam_ego_pose_history = deque(maxlen=512)
            hist = self._wam_ego_pose_history
        if hist and int(hist[-1][0]) == int(step):
            hist[-1] = pose
        else:
            hist.append(pose)

    def _wam_past_route_xy(self):
        hist = list(getattr(self, "_wam_ego_pose_history", ()))
        if len(hist) >= 2:
            return tuple((float(item[1]), float(item[2])) for item in hist[:-1])
        ego = getattr(self, "ego", None)
        if ego is None:
            return ()
        tf = ego.get_transform()
        dist = float(getattr(getattr(self, "_wam_coverage_config", None), "past_route_distance_m", 10.0))
        yaw = math.radians(float(tf.rotation.yaw))
        return ((float(tf.location.x) - math.cos(yaw) * dist, float(tf.location.y) - math.sin(yaw) * dist),)

    def _wam_actor_polygons(self) -> Dict[int, Any]:
        actors = []
        if getattr(self, "ego", None) is not None:
            actors.append(self.ego)
        actors.extend(actor for actor in getattr(self, "group_vehs", []) if actor is not None)
        actors.extend(self._wam_object_actors())
        polygons = {}
        for actor in actors:
            try:
                polygons[int(actor.id)] = self._actor_polygon_xy(actor)
            except Exception:
                V2V_LOGGER.debug("Failed to build coverage polygon actor_id=%s", getattr(actor, "id", None))
        return polygons

    def _available_wam_messages(self, step: int):
        proc = self._ensure_comm_process()
        return proc.available_messages(int(step)) if proc.policy is not None else []

    def _latest_wam_messages_by_sender(self, messages):
        latest: Dict[int, Any] = {}
        for message in messages:
            current = latest.get(int(message.sender_id))
            if current is None or int(message.t_sense) >= int(current.t_sense):
                latest[int(message.sender_id)] = message
        return latest

    def _build_wam_coverage(self, step: Optional[int] = None) -> Dict[str, float]:
        if step is None:
            step = int(getattr(self, "_time_step", 0))
        if (
            not bool(getattr(self, "_wam_coverage_enabled", False))
            or getattr(self, "ego", None) is None
            or not bool(getattr(self, "_wam_build_graph", True))
        ):
            self._wam_coverage_raster = None
            return {
                "coverage_uncertainty": 0.0,
                "route_coverage_ratio": 0.0,
                "route_coverage_quality_mean": 0.0,
                "poor_coverage_risk_mean": 0.0,
            }
        if int(getattr(self, "_wam_coverage_step", -1)) == int(step) and self._wam_coverage_raster is not None:
            return {
                key: float(getattr(self, "_wam_uncertainty_breakdown", {}).get(key, 0.0))
                for key in (
                    "coverage_uncertainty",
                    "route_coverage_ratio",
                    "route_coverage_quality_mean",
                    "poor_coverage_risk_mean",
                )
            }

        ego_tf = self.ego.get_transform()
        ego_pose = (float(ego_tf.location.x), float(ego_tf.location.y), float(ego_tf.rotation.yaw))
        messages = self._available_wam_messages(int(step))
        latest = self._latest_wam_messages_by_sender(messages)
        cfg = getattr(self, "_wam_coverage_config", CoverageConfig())
        gamma = float(getattr(cfg, "freshness_gamma", 5.0))
        dt = float(self._comm_config.dt)
        collaborators = []
        collaborator_freshness = []
        for sender_id, message in sorted(latest.items()):
            pose = message.data.get("pose", {})
            collaborators.append(
                (
                    int(sender_id),
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("yaw", 0.0)),
                )
            )
            age_s = max(float(int(step) - int(getattr(message, "t_sense", step))) * dt, 0.0)
            collaborator_freshness.append(math.exp(-gamma * age_s))
        raster, metrics = build_coverage_raster(
            ego_pose=ego_pose,
            route_xy=self._wam_route_xy(),
            past_route_xy=self._wam_past_route_xy(),
            ego_observer=(int(self.ego.id), ego_pose[0], ego_pose[1], ego_pose[2]),
            collaborator_observers=tuple(collaborators),
            actor_polygons=self._wam_actor_polygons(),
            ego_fov=float(self._wam_local_sight_fov),
            ego_sight_range=float(self._wam_local_sight_range),
            collaborator_fov=float(self._wam_collaborator_sight_fov),
            collaborator_sight_range=float(self._wam_collaborator_sight_range),
            config=cfg,
            spec=getattr(self, "_wam_bev_spec", BevSpec()),
            collaborator_freshness=tuple(collaborator_freshness),
        )
        self._wam_coverage_raster = raster
        self._wam_coverage_step = int(step)
        return metrics

    def _update_wam_uncertainty_breakdown(self, step: int, *, motion_uncertainty: Optional[float] = None) -> None:
        if motion_uncertainty is None:
            vals = [float(pred.uncertainty_score) for pred in getattr(self, "_wam_motion_predictions", {}).values()]
            motion_uncertainty = max(vals) if vals else 0.0
        coverage = self._build_wam_coverage(int(step))
        alpha = float(getattr(self, "_wam_uncertainty_alpha_motion", 1.0))
        beta = float(getattr(self, "_wam_uncertainty_beta_coverage", 1.0))
        coverage_uncertainty = float(coverage.get("coverage_uncertainty", 0.0))
        total = alpha * float(motion_uncertainty) + beta * coverage_uncertainty
        # [0,1]-normalized view: same saturation 1-exp(-./tau) as the offline evaluator, but applied to
        # the online (union, notable_prob-weighted) motion uncertainty -- online has no GT so it cannot
        # restrict to the fixed GT-notable set the offline ``*_norm_notable`` columns use. Hence these
        # keep the plain ``_norm`` name. Reported only; the coop threshold below still uses raw ``total``.
        tau = max(float(getattr(self, "_wam_uncertainty_sigma_scale", 4.0)), 1e-6)
        a_norm = float(min(max(getattr(self, "_wam_uncertainty_alpha_norm", 0.5), 0.0), 1.0))
        motion_norm = 1.0 - math.exp(-max(float(motion_uncertainty), 0.0) / tau)
        cov01 = min(max(coverage_uncertainty, 0.0), 1.0)
        self._wam_uncertainty_breakdown = {
            "motion_uncertainty": float(motion_uncertainty),
            "coverage_uncertainty": coverage_uncertainty,
            "total_uncertainty": float(total),
            "motion_uncertainty_norm": float(motion_norm),
            "total_uncertainty_norm": float(a_norm * motion_norm + (1.0 - a_norm) * cov01),
            "route_coverage_ratio": float(coverage.get("route_coverage_ratio", 0.0)),
            "route_coverage_quality_mean": float(coverage.get("route_coverage_quality_mean", 0.0)),
            "poor_coverage_risk_mean": float(coverage.get("poor_coverage_risk_mean", 0.0)),
        }
        if float(total) > float(getattr(self, "_wam_uncertainty_threshold", 1.0)):
            high_ids = tuple(
                sorted(
                    int(actor_id)
                    for actor_id, pred in getattr(self, "_wam_motion_predictions", {}).items()
                    if float(pred.uncertainty_score) > float(getattr(self, "_wam_uncertainty_threshold", 1.0))
                )
            )
            if not high_ids:
                high_ids = tuple(
                    int(record.object_state.actor_id) for record in getattr(self, "_wam_notable_records", ())
                )
            from .toolkit.wam import CoopRequest

            self._wam_coop_request = CoopRequest(
                ego_id=int(self.ego.id),
                step=int(step),
                high_uncertainty_object_ids=high_ids,
                uncertainty_threshold=float(getattr(self, "_wam_uncertainty_threshold", 1.0)),
                reason="total_uncertainty_above_threshold",
            )
        else:
            self._wam_coop_request = None

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

    def _update_wam_notable_records(self) -> None:
        route_points = self._wam_reference_route_points()
        self._wam_notable_records = select_notable_objects(
            self._wam_object_states,
            route_points,
            notable_distance_m=float(self._wam_notable_distance_m),
            max_notable_objects=int(self._wam_max_notable_objects),
        )

    def _update_wam_runtime_state(self, *, force: bool = False) -> None:
        """Refresh perception/graph every step; recompute request every ``Ta`` (§2.4)."""
        if not bool(getattr(self, "_wam_enabled", True)):
            return
        step = int(getattr(self, "_time_step", 0))
        self._record_wam_ego_pose_history(step)
        self._refresh_object_states(step)  # perception: every step (deduped)
        self._update_wam_notable_records()
        if bool(getattr(self, "_wam_build_graph", True)):
            self._update_wam_graph(step)
            self._update_wam_graph_window(step)
        self._update_wam_checkpoint_slot_cache(step)

        last = int(getattr(self, "_wam_last_predict_step", -1))
        action_period = max(int(self._comm_config.action_period_steps), 1)
        if not force and last >= 0 and (step - last) < action_period:
            return  # notable-motion prediction / request runs only every Ta

        mode = str(getattr(self, "_wam_predictor_mode", "rule")).lower()
        if mode == "checkpoint":
            self._predict_wam_with_checkpoint(step)
        else:
            self._predict_wam_with_rule(step)
        self._wam_last_predict_step = step
        request = getattr(self, "_wam_coop_request", None)
        high_ids = tuple() if request is None else tuple(int(v) for v in request.high_uncertainty_object_ids)
        request_signature = (bool(request is not None), high_ids)
        if request_signature != getattr(self, "_last_logged_request_signature", None):
            self._last_logged_request_signature = request_signature
            log_key_event(
                V2V_LOGGER,
                logging.INFO,
                "WAM request state step=%d predictor=%s triggered=%s high_uncertainty_ids=%s max_uncertainty=%.3f threshold=%.3f notable=%s visible=%s invisible=%s candidates=%s",
                int(step),
                mode,
                bool(request is not None),
                list(high_ids),
                float(self._wam_max_uncertainty()),
                float(self._wam_uncertainty_threshold),
                [int(record.object_state.actor_id) for record in self._wam_notable_records],
                [int(record.object_state.actor_id) for record in self._wam_notable_records if bool(record.visible)],
                [int(record.object_state.actor_id) for record in self._wam_notable_records if bool(record.invisible)],
                sorted(int(v) for v in getattr(self, "coop_participant_ids", set())),
            )
        if should_log_periodic(step, int(get_runtime_logging_config()["step_debug_interval"]), logger=V2V_LOGGER):
            V2V_LOGGER.debug(
                "WAM step=%d predictor=%s notable=%s max_uncertainty=%.3f triggered=%s",
                step,
                mode,
                [record.object_state.actor_id for record in self._wam_notable_records],
                self._wam_max_uncertainty(),
                self._wam_coop_request is not None,
            )

    def _predict_wam_with_rule(self, step: int) -> None:
        dt = float(getattr(getattr(self._config, "world", None), "fixed_delta_seconds", 0.1))
        self._wam_motion_predictions = predict_notable_motion(
            self._wam_notable_records,
            dt=dt,
            horizon_steps=int(self._wam_prediction_horizon_steps),
            visible_uncertainty=float(self._wam_visible_uncertainty),
            invisible_uncertainty=float(self._wam_invisible_uncertainty),
        )
        self._update_wam_uncertainty_breakdown(step)

    def _resolve_wam_predictor_device(self) -> torch.device:
        requested = str(getattr(self, "_wam_predictor_device", "auto")).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(requested)

    def _load_wam_predictor(self):
        checkpoint = getattr(self, "_wam_predictor_checkpoint", None)
        if checkpoint in (None, "", "null"):
            raise RuntimeError("wam.predictor_mode=checkpoint requires wam.predictor_checkpoint")
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if self._wam_predictor_model is not None and self._wam_predictor_loaded_path == str(checkpoint_path):
            return self._wam_predictor_model

        from .toolkit.wam import WAMPerceptionConfig, WAMPerceptionModel

        device = self._resolve_wam_predictor_device()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("perception_config") if isinstance(ckpt, dict) else None
        if cfg is None:
            cfg = WAMPerceptionConfig(
                route_waypoints=int(self._wam_graph_route_waypoints),
                hidden_dim=int(self._wam_graph_hidden_dim),
                num_layers=int(self._wam_graph_num_layers),
                num_heads=int(self._wam_graph_num_heads),
                bev_channels=int(self._wam_graph_bev_channels),
                bev_size=int(self._wam_graph_bev_size),
            )
        elif isinstance(cfg, dict):
            cfg = WAMPerceptionConfig(**cfg)
        elif is_dataclass(cfg):
            # Saved checkpoints store the dataclass directly.
            pass
        else:
            raise TypeError(f"Unsupported perception_config type: {type(cfg)!r}")

        model = WAMPerceptionModel(cfg).to(device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        model.eval()
        self._wam_predictor_model = model
        self._wam_predictor_loaded_path = str(checkpoint_path)
        self._wam_predictor_device_resolved = device
        return model

    def _update_wam_checkpoint_slot_cache(self, step: int) -> None:
        proc = self._ensure_comm_process()
        self._wam_active_policy_by_step[int(step)] = proc.active_policy_id
        for message in getattr(proc.receive_queue, "messages", ()):
            msg_id = getattr(message, "msg_id", None)
            if msg_id is None:
                msg_id = (
                    int(getattr(message, "policy_id", -1)),
                    int(getattr(message, "sender_id", -1)),
                    int(getattr(message, "t_sense", -1)),
                    int(getattr(message, "t_recv", -1)),
                )
            self._wam_received_message_cache[msg_id] = message

        sample_period = max(int(getattr(self, "_wam_predictor_sample_period_steps", 1)), 1)
        if int(step) % sample_period == 0 and int(getattr(self, "_wam_slot_history_last_step", -1)) != int(step):
            self._wam_slot_state_history[int(step)] = self._wam_stage1_slot_state(int(step))
            self._wam_slot_history_last_step = int(step)
        self._prune_wam_checkpoint_slot_cache(int(step))

    def _prune_wam_checkpoint_slot_cache(self, step: int) -> None:
        history = max(int(getattr(self, "_wam_predictor_history_window", 4)), 0)
        sample_period = max(int(getattr(self, "_wam_predictor_sample_period_steps", 1)), 1)
        keep_from = int(step) - history * sample_period - int(self._comm_config.prediction_window_steps) - sample_period
        for slot_step in list(getattr(self, "_wam_slot_state_history", {})):
            if int(slot_step) < keep_from:
                del self._wam_slot_state_history[slot_step]
        for key, message in list(getattr(self, "_wam_received_message_cache", {}).items()):
            if int(getattr(message, "t_sense", keep_from)) < keep_from:
                del self._wam_received_message_cache[key]
        for policy_step in list(getattr(self, "_wam_active_policy_by_step", {})):
            if int(policy_step) < keep_from:
                del self._wam_active_policy_by_step[policy_step]

    def _wam_checkpoint_anchor_step(self, step: int) -> int:
        sample_period = max(int(getattr(self, "_wam_predictor_sample_period_steps", 1)), 1)
        return int(step) - (int(step) % sample_period)

    def _wam_checkpoint_window_steps(self, step: int):
        history = max(int(getattr(self, "_wam_predictor_history_window", 4)), 0)
        sample_period = max(int(getattr(self, "_wam_predictor_sample_period_steps", 1)), 1)
        anchor = self._wam_checkpoint_anchor_step(int(step))
        return [anchor - sample_period * idx for idx in range(history, -1, -1)]

    def _messages_for_checkpoint_slot(self, slot_step: int, prediction_step: int):
        oldest = int(prediction_step) - int(self._comm_config.prediction_window_steps)
        active_policy_id = self._wam_active_policy_by_step.get(int(prediction_step), self._ensure_comm_process().active_policy_id)
        out = []
        for message in getattr(self, "_wam_received_message_cache", {}).values():
            t_sense = int(getattr(message, "t_sense"))
            if t_sense != int(slot_step):
                continue
            if int(getattr(message, "t_recv")) > int(prediction_step):
                continue
            if not (oldest <= t_sense <= int(prediction_step)):
                continue
            if (
                not bool(self._comm_config.allow_cross_policy_messages)
                and active_policy_id is not None
                and int(getattr(message, "policy_id")) != int(active_policy_id)
            ):
                continue
            out.append(message)
        return out

    def _checkpoint_graph_window_ready(self, step: int) -> bool:
        return all(int(slot_step) in self._wam_slot_state_history for slot_step in self._wam_checkpoint_window_steps(step))

    def _build_checkpoint_graph_window(self, step: int):
        window = []
        for slot_step in self._wam_checkpoint_window_steps(step):
            state = self._wam_slot_state_history[int(slot_step)]
            messages = self._messages_for_checkpoint_slot(int(slot_step), int(step))
            window.append(self._build_wam_graph_for_stage1_slot(state, messages, int(step)))
        return window

    def _empty_wam_checkpoint_prediction_stats(self) -> Dict[str, float]:
        return {
            "checkpoint_window_slots": 0.0,
            "checkpoint_window_has_v2v_graph": 0.0,
            "checkpoint_window_v2v_slots": 0.0,
            "checkpoint_window_v2v_slot_rate": 0.0,
            "checkpoint_final_has_v2v_graph": 0.0,
            "checkpoint_final_graph_objects": 0.0,
            "checkpoint_final_ego_visible_objects": 0.0,
            "checkpoint_final_collab_only_objects": 0.0,
            "checkpoint_final_collab_object_ratio": 0.0,
            "checkpoint_window_union_objects": 0.0,
            "checkpoint_window_union_ego_visible_objects": 0.0,
            "checkpoint_window_union_collab_only_objects": 0.0,
            "checkpoint_window_union_collab_object_ratio": 0.0,
            "checkpoint_prediction_query_objects": 0.0,
        }

    def _wam_graph_object_visibility_sets(self, graph) -> Dict[str, set]:
        if graph is None or not hasattr(graph, "node_types") or OBJECT not in graph.node_types:
            return {"all": set(), "ego_visible": set(), "collab_only": set()}
        obj = graph[OBJECT]
        node_id = getattr(obj, "node_id", None)
        if node_id is None:
            return {"all": set(), "ego_visible": set(), "collab_only": set()}
        valid = node_id >= 0
        node_mask = getattr(obj, "node_mask", None)
        if node_mask is not None:
            valid = valid & (node_mask > 0.5)
        visible = getattr(obj, "visible", torch.zeros_like(node_id, dtype=torch.float32)) > 0.5
        invisible = getattr(obj, "invisible", torch.zeros_like(node_id, dtype=torch.float32)) > 0.5
        ids = [int(v) for v in node_id[valid].detach().cpu().tolist()]
        visible_ids = [int(v) for v in node_id[valid & visible].detach().cpu().tolist()]
        invisible_ids = [int(v) for v in node_id[valid & invisible].detach().cpu().tolist()]
        return {
            "all": set(ids),
            "ego_visible": set(visible_ids),
            "collab_only": set(invisible_ids),
        }

    def _wam_graph_has_v2v_vehicle(self, graph) -> bool:
        if graph is None or not hasattr(graph, "node_types") or VEHICLE not in graph.node_types:
            return False
        veh = graph[VEHICLE]
        node_id = getattr(veh, "node_id", None)
        if node_id is None:
            return False
        valid = node_id >= 0
        node_mask = getattr(veh, "node_mask", None)
        if node_mask is not None:
            valid = valid & (node_mask > 0.5)
        return int(valid.sum().item()) > 1

    def _summarize_wam_checkpoint_graph_window(self, window) -> Dict[str, float]:
        stats = self._empty_wam_checkpoint_prediction_stats()
        graphs = list(window or ())
        if not graphs:
            return stats
        slot_count = len(graphs)
        v2v_slots = sum(1 for graph in graphs if self._wam_graph_has_v2v_vehicle(graph))
        union_ids, union_ego_visible, union_collab_only = set(), set(), set()
        for graph in graphs:
            sets = self._wam_graph_object_visibility_sets(graph)
            union_ids |= sets["all"]
            union_ego_visible |= sets["ego_visible"]
            union_collab_only |= sets["collab_only"]
        final_sets = self._wam_graph_object_visibility_sets(graphs[-1])
        final_total = len(final_sets["all"])
        union_total = len(union_ids)
        stats.update(
            {
                "checkpoint_window_slots": float(slot_count),
                "checkpoint_window_has_v2v_graph": 1.0 if v2v_slots else 0.0,
                "checkpoint_window_v2v_slots": float(v2v_slots),
                "checkpoint_window_v2v_slot_rate": float(v2v_slots) / float(slot_count),
                "checkpoint_final_has_v2v_graph": 1.0 if self._wam_graph_has_v2v_vehicle(graphs[-1]) else 0.0,
                "checkpoint_final_graph_objects": float(final_total),
                "checkpoint_final_ego_visible_objects": float(len(final_sets["ego_visible"])),
                "checkpoint_final_collab_only_objects": float(len(final_sets["collab_only"])),
                "checkpoint_final_collab_object_ratio": (
                    float(len(final_sets["collab_only"])) / float(final_total) if final_total else 0.0
                ),
                "checkpoint_window_union_objects": float(union_total),
                "checkpoint_window_union_ego_visible_objects": float(len(union_ego_visible)),
                "checkpoint_window_union_collab_only_objects": float(len(union_collab_only)),
                "checkpoint_window_union_collab_object_ratio": (
                    float(len(union_collab_only)) / float(union_total) if union_total else 0.0
                ),
            }
        )
        return stats

    def _graph_window_ready(self) -> bool:
        step = int(getattr(self, "_time_step", 0))
        return self._checkpoint_graph_window_ready(step)

    def _predict_wam_with_checkpoint(self, step: int) -> None:
        self._wam_motion_predictions = {}
        self._wam_coop_request = None
        self._wam_checkpoint_prediction_stats = self._empty_wam_checkpoint_prediction_stats()
        if not self._checkpoint_graph_window_ready(int(step)):
            return
        model = self._load_wam_predictor()
        device = self._wam_predictor_device_resolved or self._resolve_wam_predictor_device()
        raw_window = self._build_checkpoint_graph_window(int(step))
        self._wam_checkpoint_prediction_stats = self._summarize_wam_checkpoint_graph_window(raw_window)
        window = [graph.clone().to(device) for graph in raw_window]
        with torch.no_grad():
            out = model(window)
        object_ids = [int(v) for v in out["object_node_ids"].detach().cpu().tolist()]
        self._wam_checkpoint_prediction_stats["checkpoint_prediction_query_objects"] = float(len(object_ids))
        if not object_ids:
            self._update_wam_uncertainty_breakdown(step, motion_uncertainty=0.0)
            return
        notable_prob = out["notable_prob"].detach()
        # Soft-gate the notable_prob weight so objects the model is unsure about are suppressed before
        # the notable-weighted average (online has no GT, so it uses the model's predicted prob).
        gate_k = float(getattr(self, "_wam_uncertainty_notable_gate_k", 0.0))
        gate_thr = float(getattr(self, "_wam_uncertainty_notable_gate_threshold", 0.5))
        weight = torch.sigmoid(gate_k * (notable_prob - gate_thr)) if gate_k > 0 else notable_prob
        mass_floor = float(getattr(self, "_wam_uncertainty_notable_mass_floor", 0.0))
        trace = torch.exp(out["traj_log_var"].detach()).sum(dim=-1)  # [Q, H]
        uncertainty = trace.mean(dim=-1)
        motion_uncertainty = float(
            (weight * uncertainty).sum() / (weight.sum() + mass_floor).clamp_min(1e-6)
        )
        source = str(getattr(self, "_wam_predictor_uncertainty_source", "notable_weighted_trace"))
        score = weight * uncertainty if source == "notable_weighted_trace" else uncertainty
        mu = out["traj_mu"].detach().cpu()
        score_cpu = score.detach().cpu()
        uncertainty_cpu = uncertainty.detach().cpu()
        from .toolkit.wam import MotionPredictionRecord

        self._wam_motion_predictions = {}
        for idx, actor_id in enumerate(object_ids):
            future_xy = tuple((float(x), float(y)) for x, y in mu[idx].tolist())
            u = float(score_cpu[idx])
            raw_u = float(uncertainty_cpu[idx])
            cov = tuple((raw_u, raw_u) for _ in future_xy)
            self._wam_motion_predictions[int(actor_id)] = MotionPredictionRecord(
                actor_id=int(actor_id),
                future_xy=future_xy,
                covariance_diag=cov,
                uncertainty_score=u,
            )
        self._update_wam_uncertainty_breakdown(step, motion_uncertainty=motion_uncertainty)

    def _wam_max_uncertainty(self) -> float:
        breakdown = getattr(self, "_wam_uncertainty_breakdown", None) or {}
        if "total_uncertainty" in breakdown:
            return float(breakdown.get("total_uncertainty", 0.0))
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
        if str(getattr(self, "_wam_bev_payload_mode", "feature")).lower() == "feature":
            return float(max(int(self._wam_bev_feature_dim), 0) * max(int(self._wam_bev_feature_dtype_bytes), 1))
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

    def _assemble_wam_graph_from_inputs(
        self,
        *,
        ego: VehicleNodeInput,
        ego_pose,
        live_states: List[ObjectState],
        route_xy,
        messages,
        step: int,
        notable_ids,
    ):
        dt = float(self._comm_config.dt)
        ego_visible = [s for s in live_states if bool(s.visible_to_ego)]

        # Most-recent message per collaborator (§13.2): its latency drives the veh_veh edge.
        latest_by_sender: Dict[int, Any] = {}
        for message in messages:
            current = latest_by_sender.get(int(message.sender_id))
            if current is None or int(message.t_sense) >= int(current.t_sense):
                latest_by_sender[int(message.sender_id)] = message

        if str(getattr(self, "_wam_graph_fusion_mode", "inject")).lower() == "inject":
            return self._assemble_wam_graph_inject(
                ego=ego, ego_visible=ego_visible, messages=messages,
                latest_by_sender=latest_by_sender, notable_ids=notable_ids,
            )

        observations = [
            ObservationNodeInput(
                vehicle_id=int(ego.actor_id),
                modality="objlist",
                observed_object_ids=tuple(int(s.actor_id) for s in ego_visible),
                payload_bytes=self._wam_objlist_payload_bytes(len(ego_visible)),
                latency_s=0.0,
                freshness=1.0,
                quality=1.0,
                sample_age_s=0.0,
            ),
            ObservationNodeInput(
                vehicle_id=int(ego.actor_id),
                modality="bev",
                observed_object_ids=tuple(int(s.actor_id) for s in ego_visible),
                payload_bytes=self._wam_bev_payload_bytes(),
                latency_s=0.0,
                freshness=1.0,
                quality=1.0,
                sample_age_s=0.0,
                bev_raster=rasterize_bev(
                    ego_pose,
                    ego_visible,
                    route_xy=route_xy,
                    spec=self._wam_bev_spec,
                ),
            ),
        ]

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
            snap_states = tuple(message.data.get("object_states", ()))  # snapshot @ t_sense
            for snap_state in snap_states:
                objects_by_id.setdefault(int(snap_state.actor_id), snap_state)
            for modality in modalities:
                if modality == "bev":
                    observed_ids = tuple(int(s.actor_id) for s in snap_states)
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
        graph = build_wam_hetero_graph(
            ego=ego,
            collaborators=collaborators,
            objects=objects,
            observations=observations,
            policy=policy_view,
            spec=spec,
            notable_ids={int(v) for v in notable_ids},
            latency_by_vehicle=latency_by_vehicle,
        )
        return graph, latest_by_sender, observations, ego_visible

    def _assemble_wam_graph_inject(self, *, ego, ego_visible, messages, latest_by_sender, notable_ids):
        """Injection (early-fusion) graph: ego-only structure + collaborator-revealed objects injected
        as object nodes (Phase 1). No collaborator vehicle/observation nodes, no veh_veh edges."""
        # Collect collaborator detections: each message's snapshot objects @ t_sense, with a
        # distance-based confidence from the sender's pose. fuse_injected_objects dedups (ego wins,
        # else highest confidence).
        detections = []
        for message in messages:
            pose = message.data.get("pose", {})
            observer_xy = (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
            for snap_state in message.data.get("object_states", ()):
                detections.append((snap_state, detection_confidence(observer_xy, snap_state)))
        fused = fuse_injected_objects(ego_visible, detections)

        ego_obs = ObservationNodeInput(
            vehicle_id=int(ego.actor_id),
            modality="objlist",
            observed_object_ids=tuple(int(s.actor_id) for s in fused.object_states),
            payload_bytes=self._wam_objlist_payload_bytes(len(fused.object_states)),
            latency_s=0.0,
            freshness=1.0,
            quality=1.0,
            sample_age_s=0.0,
            det_confidence_by_object=dict(fused.det_confidence_by_object),
        )
        policy_view = WAMPolicy(
            selected_vehicle_ids=(),
            modality_by_vehicle={},
            bandwidth_by_vehicle={},
            frequency_steps=int(self._comm_config.sensor_period_steps),
            reason="inject",
        )
        spec = GraphBuildSpec(
            route_waypoints=int(self._wam_graph_route_waypoints),
            max_object_nodes=int(self._wam_graph_max_object_nodes),
        )
        graph = build_wam_hetero_graph(
            ego=ego,
            collaborators=[],
            objects=fused.object_states,
            observations=[ego_obs],
            policy=policy_view,
            spec=spec,
            notable_ids={int(v) for v in notable_ids},
            object_visibility={int(oid): True for oid in fused.ego_visible_ids},
        )
        return graph, latest_by_sender, [ego_obs], ego_visible

    def _wam_stage1_slot_state(self, step: int):
        """Capture the slot-local graph inputs used later by Stage-1 recording."""
        self._refresh_object_states(int(step))
        self._update_wam_notable_records()
        route_xy = self._wam_route_xy()
        ego = self._wam_vehicle_node_input(self.ego, is_ego=True, agent_slot=0, route_xy=route_xy)
        ego_tf = self.ego.get_transform()
        ego_pose = (
            float(ego_tf.location.x),
            float(ego_tf.location.y),
            float(ego_tf.rotation.yaw),
        )
        collaborators = tuple(
            self._wam_vehicle_node_input(actor, is_ego=False, agent_slot=idx + 1, route_xy=())
            for idx, actor in enumerate(getattr(self, "group_vehs", ()))
            if actor is not None
        )
        return {
            "step": int(step),
            "ego": ego,
            "collaborators": collaborators,
            "ego_pose": ego_pose,
            "live_states": tuple(getattr(self, "_wam_object_states", ())),
            "route_xy": tuple(route_xy),
            "past_route_xy": tuple(self._wam_past_route_xy()),
            "actor_polygons": self._wam_actor_polygons(),
            "notable_ids": tuple(int(record.object_state.actor_id) for record in getattr(self, "_wam_notable_records", ())),
        }

    def _build_wam_graph_for_stage1_slot(self, state, messages, prediction_step: int):
        """Rebuild one Stage-1 history slot using only messages selected by the recorder."""
        slot_step = int(state.get("step", prediction_step))
        graph, _, _, _ = self._assemble_wam_graph_from_inputs(
            ego=state["ego"],
            ego_pose=state["ego_pose"],
            live_states=list(state.get("live_states", ())),
            route_xy=tuple(state.get("route_xy", ())),
            messages=list(messages),
            step=slot_step,
            notable_ids=state.get("notable_ids", ()),
        )
        return graph

    def _build_wam_coverage_for_stage1_slot(self, state, messages, prediction_step: int):
        """Rebuild one Stage-1 coverage raster using only messages selected by the recorder.

        Each collaborator's coverage is discounted by its message freshness ``exp(-gamma * age_s)``,
        ``age_s = (prediction_step - t_sense) * dt`` -- a stale snapshot covers a region less reliably.
        """
        if not bool(getattr(self, "_wam_coverage_enabled", False)):
            return None
        ego_pose = tuple(state["ego_pose"])
        cfg = getattr(self, "_wam_coverage_config", CoverageConfig())
        gamma = float(getattr(cfg, "freshness_gamma", 5.0))
        dt = float(self._comm_config.dt)
        latest = self._latest_wam_messages_by_sender(messages)
        collaborators = []
        collaborator_freshness = []
        for sender_id, message in sorted(latest.items()):
            pose = message.data.get("pose", {})
            collaborators.append(
                (
                    int(sender_id),
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("yaw", 0.0)),
                )
            )
            age_s = max(float(int(prediction_step) - int(getattr(message, "t_sense", prediction_step))) * dt, 0.0)
            collaborator_freshness.append(math.exp(-gamma * age_s))
        raster, _ = build_coverage_raster(
            ego_pose=ego_pose,
            route_xy=tuple(state.get("route_xy", ())),
            past_route_xy=tuple(state.get("past_route_xy", ())),
            ego_observer=(int(self.ego.id), float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])),
            collaborator_observers=tuple(collaborators),
            actor_polygons=state.get("actor_polygons", {}),
            ego_fov=float(self._wam_local_sight_fov),
            ego_sight_range=float(self._wam_local_sight_range),
            collaborator_fov=float(self._wam_collaborator_sight_fov),
            collaborator_sight_range=float(self._wam_collaborator_sight_range),
            config=cfg,
            spec=getattr(self, "_wam_bev_spec", BevSpec()),
            collaborator_freshness=tuple(collaborator_freshness),
        )
        return raster

    def _build_wam_coverage_for_stage1_policy(self, state, policy):
        """Build one Stage-1 coverage raster for a counterfactual fixed policy."""
        if not bool(getattr(self, "_wam_coverage_enabled", False)):
            return None
        ego_pose = tuple(state["ego_pose"])
        collaborator_by_id = {int(v.actor_id): v for v in state.get("collaborators", ())}
        collaborators = []
        for vid in sorted(int(v) for v in getattr(policy, "selected_vehicle_ids", ())):
            node = collaborator_by_id.get(int(vid))
            if node is None:
                continue
            collaborators.append((int(vid), float(node.x), float(node.y), float(node.yaw)))
        raster, _ = build_coverage_raster(
            ego_pose=ego_pose,
            route_xy=tuple(state.get("route_xy", ())),
            past_route_xy=tuple(state.get("past_route_xy", ())),
            ego_observer=(int(self.ego.id), float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])),
            collaborator_observers=tuple(collaborators),
            actor_polygons=state.get("actor_polygons", {}),
            ego_fov=float(self._wam_local_sight_fov),
            ego_sight_range=float(self._wam_local_sight_range),
            collaborator_fov=float(self._wam_collaborator_sight_fov),
            collaborator_sight_range=float(self._wam_collaborator_sight_range),
            config=getattr(self, "_wam_coverage_config", CoverageConfig()),
            spec=getattr(self, "_wam_bev_spec", BevSpec()),
        )
        return raster

    def _update_wam_graph(self, step: Optional[int] = None) -> None:
        if step is None:
            step = int(getattr(self, "_time_step", 0))
        if int(getattr(self, "_wam_graph_step", -1)) == int(step) and self._wam_graph is not None:
            return
        self._build_wam_graph(int(step))
        self._wam_graph_step = int(step)

    def _update_wam_graph_window(self, step: Optional[int] = None) -> None:
        if step is None:
            step = int(getattr(self, "_time_step", 0))
        if self._wam_graph is None:
            return
        if int(getattr(self, "_wam_graph_window_last_step", -1)) == int(step):
            return
        self._wam_graph_window.append(self._wam_graph)
        self._wam_graph_window_last_step = int(step)

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
        live_states = list(getattr(self, "_wam_object_states", []))

        ego_route_xy = self._wam_route_xy()
        ego = self._wam_vehicle_node_input(self.ego, is_ego=True, agent_slot=0, route_xy=ego_route_xy)
        ego_tf = self.ego.get_transform()
        ego_pose = (
            float(ego_tf.location.x),
            float(ego_tf.location.y),
            float(ego_tf.rotation.yaw),
        )
        notable_ids = {int(record.object_state.actor_id) for record in self._wam_notable_records}
        self._wam_graph, latest_by_sender, observations, ego_visible = self._assemble_wam_graph_from_inputs(
            ego=ego,
            ego_pose=ego_pose,
            live_states=live_states,
            route_xy=ego_route_xy,
            messages=messages,
            step=int(step),
            notable_ids=notable_ids,
        )
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(
            int(step),
            int(runtime_cfg["step_debug_interval"]),
            logger=V2V_LOGGER,
            level=logging.INFO,
        ):
            modality_counts: Dict[str, int] = {}
            for obs in observations:
                modality_counts[str(obs.modality)] = modality_counts.get(str(obs.modality), 0) + 1
            stats = hetero_graph_stats(self._wam_graph)
            log_key_event(
                V2V_LOGGER,
                logging.INFO,
                "WAM graph built step=%d vehicles=%d objects=%d observations=%d veh_obs_edges=%d obs_obj_edges=%d veh_veh_edges=%d ego_visible=%d messages=%d collaborators=%s modalities=%s ego_bev=%s",
                int(step),
                int(stats["wam_graph_num_vehicle_nodes"]),
                int(stats["wam_graph_num_object_nodes"]),
                int(stats["wam_graph_num_observation_nodes"]),
                int(stats["wam_graph_num_veh_obs_edges"]),
                int(stats["wam_graph_num_obs_obj_edges"]),
                int(stats["wam_graph_num_veh_veh_edges"]),
                len(ego_visible),
                len(messages),
                sorted(int(v) for v in latest_by_sender.keys()),
                modality_counts,
                any(int(obs.vehicle_id) == int(self.ego.id) and str(obs.modality) == "bev" for obs in observations),
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
        step = int(getattr(self, "_time_step", 0))
        self._refresh_object_states(step)
        self._update_wam_notable_records()
        if self._wam_build_graph:
            self._update_wam_graph(step)
        notable = list(getattr(self, "_wam_notable_records", []))
        policy = getattr(self, "_wam_policy", None)
        uncertainty = dict(getattr(self, "_wam_uncertainty_breakdown", {}) or {})
        info = {
            "wam_notable_object_ids": [int(record.object_state.actor_id) for record in notable],
            "wam_visible_notable_object_ids": [
                int(record.object_state.actor_id) for record in notable if bool(record.visible)
            ],
            "wam_invisible_notable_object_ids": [
                int(record.object_state.actor_id) for record in notable if bool(record.invisible)
            ],
            "wam_uncertainty_max": float(self._wam_max_uncertainty()),
            "wam_motion_uncertainty": float(uncertainty.get("motion_uncertainty", 0.0)),
            "wam_coverage_uncertainty": float(uncertainty.get("coverage_uncertainty", 0.0)),
            "wam_total_uncertainty": float(uncertainty.get("total_uncertainty", self._wam_max_uncertainty())),
            "wam_route_coverage_ratio": float(uncertainty.get("route_coverage_ratio", 0.0)),
            "wam_route_coverage_quality_mean": float(uncertainty.get("route_coverage_quality_mean", 0.0)),
            "wam_poor_coverage_risk_mean": float(uncertainty.get("poor_coverage_risk_mean", 0.0)),
            "wam_coop_triggered": bool(getattr(self, "_wam_coop_request", None) is not None),
            "wam_policy_selected_vehicle_ids": list(policy.selected_vehicle_ids) if policy is not None else [],
            "wam_policy_modality_by_vehicle": dict(policy.modality_by_vehicle) if policy is not None else {},
        }
        # Allocated bandwidth this step = Σ_m B_m of the active BS policy (0 for local-only).
        active_policy = getattr(self._ensure_comm_process(), "policy", None) if hasattr(self, "_ensure_comm_process") else None
        if active_policy is not None:
            info["wam_allocated_bandwidth"] = float(sum(float(v) for v in active_policy.bandwidth_by_vehicle.values()))
        else:
            info["wam_allocated_bandwidth"] = 0.0
        # Lyapunov queue state (only populated in policy_sampler_mode=lyapunov).
        sched = getattr(self, "_wam_lyapunov_scheduler", None)
        if sched is not None and getattr(sched, "lyap", None) is not None:
            try:
                info["wam_lyap_z"] = float(sched.lyap.z.value)
                info["wam_lyap_total_backlog"] = float(sum(float(v) for v in sched.lyap.backlogs().values()))
            except Exception:  # unverified online scheduler path: never break the info dict
                info["wam_lyap_z"] = 0.0
                info["wam_lyap_total_backlog"] = 0.0
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
