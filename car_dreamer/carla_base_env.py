from abc import abstractmethod
from typing import Dict, Tuple, Optional, Any

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic, summarize_keys

from .toolkit import EnvMonitorOpenCV, Observer, ScenarioActorManager, WorldManager


ENV_LOGGER = get_runtime_logger("car_dreamer.env")


class CarlaBaseEnv(gym.Env):
    def __init__(self, config):
        self._config = config

        self._monitor = EnvMonitorOpenCV(self._config)
        self._world = WorldManager(self._config)
        # Config-driven background actors (vehicles + pedestrians); no-op without config.
        # V2V-enabled envs expose `_register_cooperative_candidate` so that scenario vehicles with a
        # `start` point become cooperative candidates; other envs just place them (hook is None).
        self._scenario_actors = ScenarioActorManager(
            self._world,
            getattr(self._config, "scenario_actors", None),
            cooperative_hook=getattr(self, "_register_cooperative_candidate", None),
        )
        self._world.on_reset(self._on_reset_hook)
        self._world.on_step(self._on_step_hook)
        self._ego_observer = Observer(self._world, self._config.observation)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self._time_step = 0

    @abstractmethod
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        pass

    @abstractmethod
    def apply_control(self, action) -> None:
        """
        Override this method to apply control to actors.
        This method will be called before the simulator ticks.
        """
        pass

    @abstractmethod
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        pass

    def _on_reset_hook(self) -> None:
        """Internal: run the task's ``on_reset`` then spawn config-driven scenario actors."""
        self.on_reset()
        self._scenario_actors.reset_spawn()

    def _on_step_hook(self) -> None:
        """Internal: run the task's ``on_step`` then maintain scenario actors."""
        self.on_step()
        self._scenario_actors.step_update()

    @abstractmethod
    def reward(self) -> Tuple[float, Dict]:
        """
        Override this method to define the reward function.
        """
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, bool]:
        """
        Override this method to define the terminal condition.
        If one of the keys in the returned dictionary gives True, the episode will be terminated.
        """
        pass

    def get_ego_vehicle(self) -> carla.Actor:
        """
        Override this method to return the ego vehicle.
        The default behavior is to return self.ego
        """
        return self.ego

    def get_state(self) -> Dict:
        """Return the environment state. Implement this method to define the env state."""
        return self._state

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            return spaces.Box(
                low=np.array([action_config.continuous_acc[0], action_config.continuous_steer[0]]),
                high=np.array([action_config.continuous_acc[1], action_config.continuous_steer[1]]),
                dtype=np.float32,
            )

    def _get_observation_space(self):
        return self._ego_observer.get_observation_space()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset environment (Gymnasium API): accepts `seed` and `options`.

        Returns (obs, info).
        """
        ENV_LOGGER.info("Reset environment start seed=%s", seed)
        super().reset(seed=seed)

        # Keep behavior unchanged: seed is accepted but not applied here.
        self._ego_observer.destroy()
        self._world.reset()
        self._ego_observer.reset(self.get_ego_vehicle())

        self._time_step = 0

        self.obs, info = self._ego_observer.get_observation(self.get_state())
        ENV_LOGGER.info(
            "Reset environment complete ego_id=%s obs_keys=[%s] info_keys=[%s]",
            getattr(self.get_ego_vehicle(), "id", None),
            summarize_keys(self.obs),
            summarize_keys(info),
        )
        return self.obs, info

    def get_vehicle_control(self, action):
        """
        Convert actions in the action space to vehicle control in CARLA
        """
        action_config = self._config.action
        # Calculate acceleration and steering
        if action_config.discrete:
            acc = action_config.discrete_acc[action // self.n_steer]
            steer = action_config.discrete_steer[action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]
        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)
        # throttle（油门）: 0 to 1, where 0 means no throttle and 1 means full throttle.
        # steer（转向）: -1 to 1, where -1 means full
        # brake（刹车）: 0 to 1, where 0 means no brake and 1 means full brake.
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(int(self._time_step), int(runtime_cfg["step_debug_interval"]), logger=ENV_LOGGER):
            ENV_LOGGER.debug(
                "Control conversion step=%d action=%s throttle=%.3f steer=%.3f brake=%.3f",
                self._time_step,
                action,
                throttle,
                steer,
                brake,
            )
        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = False
        for k, v in terminal_conds.items():
            if v:
                ENV_LOGGER.info("Terminal condition triggered step=%d condition=%s", self._time_step, k)
                terminal = True
            terminal_conds[k] = np.array([v], dtype=np.bool_)
        if terminal:
            terminal_conds["episode_timesteps"] = self._time_step
        terminal_conds["terminal"] = terminal
        return terminal, terminal_conds

    def step(self, action):
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(int(self._time_step), int(runtime_cfg["step_debug_interval"]), logger=ENV_LOGGER):
            ENV_LOGGER.debug("Env step start step=%d action=%s", self._time_step, action)
        self.apply_control(action)
        self._world.step()
        self._time_step += 1

        env_state = self.get_state()
        is_terminal, terminal_conds = self._is_terminal()
        self.obs, obs_info = self._ego_observer.get_observation(env_state)
        reward, reward_info = self.reward()

        info = {
            **env_state,
            **terminal_conds,
            **obs_info,
            **reward_info,
            "action": action,
        }
        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}
        if self._config.display.enable:
            self._render(self.obs, info)

        # Gymnasium API: return (obs, reward, terminated, truncated, info)
        # terminated: episode ended naturally (goal/failure)
        # truncated: episode was cut short (time limit, etc.)
        terminated = is_terminal
        truncated = False  # CarDreamer doesn't use truncated separately
        if should_log_periodic(int(self._time_step), int(runtime_cfg["step_debug_interval"]), logger=ENV_LOGGER):
            ENV_LOGGER.debug(
                "Env step complete step=%d reward=%.4f terminated=%s truncated=%s info_keys=[%s]",
                self._time_step,
                reward,
                terminated,
                truncated,
                summarize_keys(info),
            )
        return self.obs, reward, terminated, truncated, info

    def is_collision(self):
        """
        Check if the ego vehicle is in collision.
        You must include 'collsion' in observation.names to use this method.
        """
        return self.obs["collision"][0] > 0

    def _render(self, obs, info):
        self._monitor.render(obs, info)
