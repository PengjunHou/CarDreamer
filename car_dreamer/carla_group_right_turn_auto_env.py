from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import carla
from runtime_logging import get_runtime_logger, should_log_periodic

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .right_turn_auto_runtime import RECEIVED_BUFFER_SIZE, RightTurnAutoRuntimeMixin
from .toolkit import (
    DEFAULT_PAYLOAD_TYPE,
    GraphBuildConfig,
    LatencyModel,
    NetResource,
    Observer,
    RuleBasedPayloadSelector,
    SimpleWirelessLatency,
    V2VMessage,
    VehicleNodeGraphBuilder,
    build_default_payload_registry,
    canonicalize_payload_type,
)
from .toolkit.policy import RuleBasedPolicySelector, build_default_policy_registry


AUTO_ENV_LOGGER = get_runtime_logger("car_dreamer.env.right_turn_auto")


class CarlaGroupRightTurnAutoEnv(RightTurnAutoRuntimeMixin, CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn right) and avoid collision.

    This environment intentionally ignores the external action for ego control and
    instead drives the ego vehicle with CARLA's `BasicAgent`. The action is still
    accepted to preserve the Gym interface expected by the training stack.
    """

    def __init__(self, config):
        super().__init__(config)
        self._apply_speed_preset()
        self._init_group_state()
        self._init_communication_config()
        self._init_graph_builder()
        self._init_collaboration_policy()
        self._init_runtime_flags()

    # =========================================================
    # Initialization helpers
    # =========================================================

    def _apply_speed_preset(self) -> None:
        self._runtime_step_debug_interval_override = 0
        preset = str(getattr(self._config, "speed_preset", "")).strip().lower()
        self._speed_preset = preset
        if not preset or preset in {"default", "none"}:
            return
        if preset != "fast_episode":
            AUTO_ENV_LOGGER.warning("Unknown speed preset '%s'; ignoring.", preset)
            return
        self._config = self._config.update(
            {
                "communication.comm_period": 2,
            }
        )
        self._runtime_step_debug_interval_override = 50
        AUTO_ENV_LOGGER.info(
            "Applied speed preset '%s' with faster communication settings.",
            preset,
        )

    def _init_group_state(self) -> None:
        self.groups: Dict[int, set[int]] = {}
        self.group_vehs: List[carla.Actor] = []
        self.background_vehs: List[carla.Actor] = []
        self.pedestrians: List[carla.Actor] = []
        self.num_group_vehs = int(getattr(self._config, "num_group_vehs", 2))
        self._other_observers: Dict[int, Observer] = {}
        self.group_obs: Dict[int, Dict[str, Any]] = {}
        self._prev_action = None
        self._actor_cache: Dict[int, carla.Actor] = {}

    def _init_communication_config(self) -> None:
        comm_cfg = getattr(self._config, "communication", None)
        self.group_update_period = int(getattr(comm_cfg, "group_update_period", 20))
        self.comm_period = int(getattr(comm_cfg, "comm_period", 5))

        bandwidth_hz = float(getattr(comm_cfg, "bandwidth_hz", 10e6))
        overhead_base_s = float(getattr(comm_cfg, "overhead_base_s", 0.030))
        overhead_per_kb_s = float(getattr(comm_cfg, "overhead_per_kb_s", 0.0015))
        pathloss_model = str(getattr(comm_cfg, "pathloss_model", "urban_los"))
        margin_db = float(getattr(comm_cfg, "margin_db", 10.0))
        margin_sigma_db = float(getattr(comm_cfg, "margin_sigma_db", 0.0))
        jitter_s = float(getattr(comm_cfg, "jitter_s", 0.0))
        overhead_bytes = int(getattr(comm_cfg, "overhead_bytes", 64))
        self._drop_on_capacity_exceeded = bool(getattr(comm_cfg, "drop_on_capacity_exceeded", False))
        self._log_dropped_messages = bool(getattr(comm_cfg, "log_dropped_messages", True))
        payload_cfg = getattr(self._config, "payload", None)
        enabled_types = list(getattr(payload_cfg, "enabled_types", [DEFAULT_PAYLOAD_TYPE]))
        self._payload_enabled_types = [canonicalize_payload_type(item) for item in enabled_types]
        if not self._payload_enabled_types:
            self._payload_enabled_types = [DEFAULT_PAYLOAD_TYPE]
        self._payload_selector_id = str(getattr(payload_cfg, "selector_id", "default"))
        self._payload_override_type = str(getattr(payload_cfg, "override_type", "")).strip()
        self._payload_image_jpeg_quality = int(getattr(payload_cfg, "image_jpeg_quality", 80))
        self._payload_registry = build_default_payload_registry()
        self._payload_selector = RuleBasedPayloadSelector()
        self._payload_override = canonicalize_payload_type(self._payload_override_type) if self._payload_override_type else ""

        self._default_net_res = NetResource(bandwidth_hz=bandwidth_hz)
        self.latency_model: LatencyModel = SimpleWirelessLatency(
            overhead_base_s=overhead_base_s,
            overhead_per_kb_s=overhead_per_kb_s,
            pathloss_model=pathloss_model,
            margin_db=margin_db,
            margin_sigma_db=margin_sigma_db,
            jitter_s=jitter_s,
            overhead_bytes=overhead_bytes,
        )
        self.trans_msg_type = str(getattr(self._config, "trans_msg_type", "image")) # "image_emb"
        self._in_flight: List[V2VMessage] = []
        self._received: Dict[int, Deque[V2VMessage]] = defaultdict(
            lambda: deque(maxlen=RECEIVED_BUFFER_SIZE)
        )
        self._veh_net_res: Dict[int, NetResource] = {}

    def _init_graph_builder(self) -> None:
        self.feature_size = int(getattr(self._config, "feature_size", 64))
        graph_cfg = getattr(self._config, "graph", None)
        cfg = GraphBuildConfig(
            window_s=float(getattr(graph_cfg, "window_s", 2.0)),
            Tmax=int(getattr(graph_cfg, "tmax", 15)),
            max_nodes=int(getattr(graph_cfg, "max_nodes", 3)),
            feat_dim_max=int(getattr(graph_cfg, "feat_dim_max", 128)),
            star_graph=bool(getattr(graph_cfg, "star_graph", True)),
        )
        self._graph_builder = VehicleNodeGraphBuilder(cfg)

    def _init_runtime_flags(self) -> None:
        self.agent = None

    def _init_collaboration_policy(self) -> None:
        self._policy_mode = str(getattr(self._config, "policy_mode", "fixed")).strip().lower() or "fixed"
        if self._policy_mode not in {"fixed", "adaptive"}:
            AUTO_ENV_LOGGER.warning("Unknown policy_mode '%s'; falling back to fixed.", self._policy_mode)
            self._policy_mode = "fixed"
        self._policy_selector_id = str(getattr(self._config, "policy_selector_id", "default"))
        self._policy_override_id = str(getattr(self._config, "policy_override", "")).strip()
        self._policy_registry = build_default_policy_registry()
        self._policy_selector = RuleBasedPolicySelector()
        self._collaboration_policy_id = str(getattr(self._config, "policy_id", "P3"))
        self._collaboration_policy = self._policy_registry.get(self._collaboration_policy_id)
        self._collaboration_policy_seed = int(getattr(self._config, "policy_seed", 0))
        self._collaboration_bandwidth_floor = float(
            getattr(self._config, "policy_bandwidth_floor", 0.1)
        )
        AUTO_ENV_LOGGER.info(
            "Configured collaboration policy mode=%s policy_id=%s selector=%s override=%s seed=%d bandwidth_floor=%.3f",
            self._policy_mode,
            self._collaboration_policy_id,
            self._policy_selector_id,
            self._policy_override_id or "<none>",
            self._collaboration_policy_seed,
            self._collaboration_bandwidth_floor,
        )

    # =========================================================
    # Environment overrides
    # =========================================================

    def on_reset(self) -> None:
        self._reset_group_runtime_state()
        self._destroy_group_observers()
        self._configure_traffic_lights()
        super().on_reset()
        self._setup_basic_agent()
        self.generate_group_vehicles()
        self.generate_background_actors()
        self._refresh_actor_cache()
        AUTO_ENV_LOGGER.info(
            "Right-turn auto reset complete ego_id=%s group_vehicle_ids=%s",
            getattr(self.ego, "id", None),
            [int(actor.id) for actor in self.group_vehs],
        )

    def on_step(self) -> None:
        self._deliver_messages()
        self._update_group_observations()
        if self._time_step % max(self.comm_period, 1) == 0:
            self._run_group_communication()
        self._cleanup_actor_flow()
        if should_log_periodic(
            int(self._time_step),
            int(self._get_runtime_debug_interval()),
            logger=AUTO_ENV_LOGGER,
        ):
            AUTO_ENV_LOGGER.debug(
                "Right-turn auto step=%d in_flight=%d received_for_ego=%d",
                self._time_step,
                len(self._in_flight),
                len(self._received.get(int(self.ego.id), deque())),
            )
        super().on_step()

    def apply_control(self, action):
        del action
        if self.agent is None:
            raise RuntimeError("BasicAgent is not initialized. Call reset() before step().")
        control = self.agent.run_step()
        self.ego.apply_control(control)

    def get_state(self):
        self._state = {"ego_waypoints": self.waypoints, "timesteps": self._time_step}
        return self._state

    def step(self, action):
        self.get_state()
        _, reward, terminated, truncated, info = super().step(action)
        info = self._merge_step_info(info, requested_action=action)
        info = self._handle_episode_end(terminated, truncated, info)
        if terminated or truncated:
            AUTO_ENV_LOGGER.info(
                "Episode ended step=%d reward=%.4f terminated=%s truncated=%s info_keys=%s",
                self._time_step,
                reward,
                terminated,
                truncated,
                sorted(info.keys()),
            )
        return self.obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        del options
        _, _ = super().reset(seed=seed)
        info = self._build_reset_info()
        AUTO_ENV_LOGGER.debug("Right-turn auto reset info keys=%s", sorted(info.keys()))
        return self.obs, info
