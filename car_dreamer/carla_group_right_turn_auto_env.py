from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .right_turn_auto_runtime import GROUP_ID, RightTurnAutoRuntimeMixin


AUTO_ENV_LOGGER = get_runtime_logger("car_dreamer.env.right_turn_auto")


class CarlaGroupRightTurnAutoEnv(RightTurnAutoRuntimeMixin, CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn right) and avoid collision.

    This environment intentionally ignores the external action for ego control and
    instead drives the ego vehicle with CARLA's `BasicAgent`. The action is still
    accepted to preserve the Gym interface expected by the training stack.

    Cooperative perception (V2V) is provided by :class:`V2VCommMixin` via
    :class:`RightTurnAutoRuntimeMixin`. Cooperative vehicles are declared in the task's
    ``scenario_actors`` config (any vehicle with a ``start`` point) and registered as
    candidates by the base-env scenario hook.
    """

    def __init__(self, config):
        super().__init__(config)
        self._init_v2v()
        self._init_runtime_flags()

    # =========================================================
    # Initialization helpers
    # =========================================================

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
        # Cooperative vehicles are declared in scenario_actors (vehicles with a `start` point)
        # and registered by the base-env scenario hook, which runs right after on_reset. Here we
        # only seed the cooperative group with the ego.
        self.groups.setdefault(GROUP_ID, set()).add(int(self.ego.id))
        self._refresh_actor_cache()
        AUTO_ENV_LOGGER.info(
            "Right-turn auto reset complete ego_id=%s",
            getattr(self.ego, "id", None),
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
