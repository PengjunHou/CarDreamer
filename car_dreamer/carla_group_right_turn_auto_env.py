from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import carla
from transformers import AutoProcessor
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
from .toolkit.vlm import RightTurnAutoVLMMixin
from .toolkit.emulation.policy import get_policy


AUTO_ENV_LOGGER = get_runtime_logger("car_dreamer.env.right_turn_auto")


class CarlaGroupRightTurnAutoEnv(RightTurnAutoRuntimeMixin, RightTurnAutoVLMMixin, CarlaWptFixedEnv):
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
        self._init_vlm_config()
        self._init_collaboration_policy()
        self._init_runtime_flags()
        if self._vlm_enabled:
            self._init_vlm()

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
                "vlm.eval_period": 3,
                "vlm.max_images_per_sender_for_inference": 2,
                "vlm.max_total_shared_images": 6,
                "vlm.max_msgs_per_sender": 6,
            }
        )
        self._runtime_step_debug_interval_override = 50
        AUTO_ENV_LOGGER.info(
            "Applied speed preset '%s' with faster VLM/communication settings.",
            preset,
        )

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
        self._drop_on_capacity_exceeded = bool(getattr(comm_cfg, "drop_on_capacity_exceeded", False))
        self._log_dropped_messages = bool(getattr(comm_cfg, "log_dropped_messages", True))

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

    def _init_vlm_config(self) -> None:
        vlm_cfg = getattr(self._config, "vlm", None)
        self._vlm_enabled = bool(getattr(vlm_cfg, "enabled", True))
        self._vlm_model_name = str(getattr(vlm_cfg, "model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))
        self._vlm_image_template = str(getattr(vlm_cfg, "image_template", "Analyze the driving scene."))
        self._vlm_eval_period = int(getattr(vlm_cfg, "eval_period", 1))
        self._vlm_image_obs_key = str(getattr(vlm_cfg, "image_obs_key", "camera"))
        self._vlm_local_files_only = bool(getattr(vlm_cfg, "local_files_only", False))
        self._vlm_shared_source = str(getattr(vlm_cfg, "shared_source", "received_feat"))
        self._vlm_received_window_s = float(getattr(vlm_cfg, "received_window_s", 2.0))
        self._vlm_max_msgs_per_sender = int(getattr(vlm_cfg, "max_msgs_per_sender", 20))
        self._vlm_max_images_per_sender_for_inference = int(
            getattr(vlm_cfg, "max_images_per_sender_for_inference", 4)
        )
        self._vlm_sampling_strategy = str(getattr(vlm_cfg, "sampling_strategy", "uniform"))
        self._vlm_max_total_shared_images = int(getattr(vlm_cfg, "max_total_shared_images", 12))

        self._vlm_ego_conf_weight = float(getattr(vlm_cfg, "ego_conf_weight", 1.0))
        self._vlm_default_shared_conf_weight = float(getattr(vlm_cfg, "shared_conf_weight", 1.0))
        self._vlm_shared_conf_weights: Dict[int, float] = {}
        shared_weights_cfg = getattr(vlm_cfg, "shared_weights", None)
        if shared_weights_cfg is not None:
            try:
                self._vlm_shared_conf_weights = {
                    int(k): float(v) for k, v in dict(shared_weights_cfg).items()
                }
            except Exception:
                self._vlm_shared_conf_weights = {}

        self._vlm_importance_distance_tau = float(getattr(vlm_cfg, "importance_distance_tau", 1.0))
        self._vlm_importance_region_weight = float(getattr(vlm_cfg, "importance_region_weight", 1.0))
        self._vlm_importance_facing_weight = float(getattr(vlm_cfg, "importance_facing_weight", 1.0))
        self._vlm_importance_distance_weight = float(getattr(vlm_cfg, "importance_distance_weight", 1.0))
        self._vlm_importance_ego_bias = float(getattr(vlm_cfg, "importance_ego_bias", 0.0))
        self._vlm_sc_beta = float(getattr(vlm_cfg, "sc_beta", 1.0))
        self._vlm_sensor_fov_deg = float(getattr(vlm_cfg, "sensor_fov_deg", 120.0))

        self._vlm_do_sample = bool(getattr(vlm_cfg, "do_sample", False))
        self._vlm_score_max_new_tokens = int(getattr(vlm_cfg, "score_max_new_tokens", 128))
        self._vlm_scene_description_max_new_tokens = int(
            getattr(vlm_cfg, "scene_description_max_new_tokens", 96)
        )
        self._vlm_temperature = float(getattr(vlm_cfg, "temperature", 0.0))
        self._vlm_top_p = float(getattr(vlm_cfg, "top_p", 0.9))
        self._vlm_enable_step_cache = bool(getattr(vlm_cfg, "enable_step_cache", True))
        self._vlm_enable_multi_query_scoring = bool(
            getattr(vlm_cfg, "enable_multi_query_scoring", True)
        )

        self._vlm_model = None
        self._vlm_processor: Optional[AutoProcessor] = None
        self._vlm_records: List[Dict[str, Any]] = []
        self._vlm_last_eval: Dict[str, Any] = {}
        self._vlm_step_cache: Dict[str, Any] = {}
        self._vlm_questions = self._build_vlm_questions()

    def _init_runtime_flags(self) -> None:
        self._dump_vlm_records_on_episode_end = bool(
            getattr(self._config, "dump_vlm_records_on_episode_end", True)
        )
        self._vlm_dump_dir = str(getattr(self._config, "vlm_dump_dir", "data"))
        self._dump_vlm_records_each_step = bool(
            getattr(self._config, "dump_vlm_records_each_step", False)
        )
        self._dump_emulation_records_on_episode_end = bool(
            getattr(self._config, "dump_emulation_records_on_episode_end", True)
        )
        self._emulation_dump_dir = str(
            getattr(self._config, "emulation_dump_dir", self._vlm_dump_dir)
        )
        self._vlm_episode_dumped = False
        self._emulation_episode_dumped = False
        self._emulation_episode_steps: List[Any] = []
        self._emulation_step_counter = 0
        self._emulation_episode_index = 0
        self._emulation_scene_type = "right_turn"
        self._emulation_scene_id = str(getattr(self._config, "scene_id", "right_turn_scene"))
        self._emulation_episode_id = ""
        self.agent = None

    def _init_collaboration_policy(self) -> None:
        self._collaboration_policy_id = str(getattr(self._config, "policy_id", "P3"))
        self._collaboration_policy = get_policy(self._collaboration_policy_id)
        self._collaboration_policy_seed = int(getattr(self._config, "policy_seed", 0))
        self._collaboration_bandwidth_floor = float(
            getattr(self._config, "policy_bandwidth_floor", 0.1)
        )
        AUTO_ENV_LOGGER.info(
            "Configured collaboration policy policy_id=%s seed=%d bandwidth_floor=%.3f",
            self._collaboration_policy_id,
            self._collaboration_policy_seed,
            self._collaboration_bandwidth_floor,
        )

    def _begin_emulation_logging_episode(self) -> None:
        self._emulation_episode_index += 1
        self._emulation_episode_id = (
            f"{self._emulation_scene_type}_episode_{self._emulation_episode_index:06d}"
        )
        self._emulation_episode_steps = []
        self._emulation_step_counter = 0
        self._vlm_episode_dumped = False
        self._emulation_episode_dumped = False

    # =========================================================
    # Environment overrides
    # =========================================================

    def on_reset(self) -> None:
        self._reset_group_runtime_state()
        self._begin_emulation_logging_episode()
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
        if self._vlm_enabled and self._time_step % max(self._vlm_eval_period, 1) == 0:
            try:
                AUTO_ENV_LOGGER.debug("Starting VLM evaluation at step=%d", self._time_step)
                self._evaluate_vlm_questions()
                AUTO_ENV_LOGGER.debug("Completed VLM evaluation at step=%d", self._time_step)
                self._maybe_dump_vlm_records_step()
            except Exception as exc:
                AUTO_ENV_LOGGER.exception("VLM evaluation failed at step=%d", self._time_step)
                self._vlm_last_eval = {
                    "step": int(self._time_step),
                    "status": "error",
                    "error": str(exc),
                }
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

    def dump_vlm_records(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._vlm_records, handle, ensure_ascii=False, indent=2)

    def dump_emulation_episode(self, path: str) -> None:
        from .toolkit.emulation import episode_to_dict, validate_episode_record
        from .toolkit.vlm.right_turn_auto_predictor_logging import (
            build_runtime_emulation_episode,
        )

        if not self._emulation_episode_steps:
            raise ValueError("No predictor-ready emulation steps are available for dumping.")
        episode = build_runtime_emulation_episode(
            scene_id=str(self._emulation_scene_id),
            episode_id=str(self._emulation_episode_id),
            scene_type=str(self._emulation_scene_type),
            dt=float(self._config.world.fixed_delta_seconds),
            policy_id=str(getattr(self, "_collaboration_policy_id", "")),
            steps=self._emulation_episode_steps,
            metadata={
                "source": "right_turn_auto_runtime_logging",
                "env_step_final": int(self._time_step),
                "vlm_record_count": int(len(self._vlm_records)),
            },
        )
        validate_episode_record(episode)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(episode_to_dict(episode), handle, ensure_ascii=False, indent=2)
