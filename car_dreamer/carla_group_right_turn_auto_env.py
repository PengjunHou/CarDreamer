from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import carla
from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .right_turn_auto_runtime import RECEIVED_BUFFER_SIZE, RightTurnAutoRuntimeMixin
from .toolkit import (
    GraphBuildConfig,
    LatencyModel,
    NetResource,
    Observer,
    SimpleWirelessLatency,
    V2VMessage,
    VehicleNodeGraphBuilder,
    payload_fn_llm,
)


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
        self._init_group_state()
        self._init_communication_config()
        self._init_graph_builder()
        self._init_runtime_flags()

    # =========================================================
    # Initialization helpers
    # =========================================================

    def _init_group_state(self) -> None:
        self.groups: Dict[int, set[int]] = {}
        self.group_vehs: List[carla.Actor] = []
        self.num_group_vehs = int(getattr(self._config, "num_group_vehs", 2))
        self._other_observers: Dict[int, Observer] = {}
        self.group_obs: Dict[int, Dict[str, Any]] = {}
        self._prev_action = None
        self._actor_cache: Dict[int, carla.Actor] = {}

    def _init_communication_config(self) -> None:
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
        self.latency_model: LatencyModel = SimpleWirelessLatency(
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
        runtime_cfg = get_runtime_logging_config()
        print(f"Step {self._time_step}: in_flight={len(self._in_flight)} received_for_ego={len(self._received.get(int(self.ego.id), []))}")
        if should_log_periodic(int(self._time_step), int(runtime_cfg["step_debug_interval"]), logger=AUTO_ENV_LOGGER):
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
