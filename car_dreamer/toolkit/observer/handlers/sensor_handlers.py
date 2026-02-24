from abc import abstractmethod
from typing import Dict, Tuple

import carla
import numpy as np
from gymnasium import spaces
import os
import cv2

from ...carla_manager import WorldManager
from .base_handler import BaseHandler


class SensorHandler(BaseHandler):
    """
    Base handler for sensor data endpoints.
    """

    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        blueprint = self._world.get_blueprint(config.blueprint)
        if "transform" in config:
            self._transform = carla.Transform(carla.Location(**config.transform))
        else:
            self._transform = carla.Transform()
        if "attributes" in config:
            for attr_name, attr_value in config.attributes.items():
                blueprint.set_attribute(attr_name, str(attr_value))
        self._blueprint = blueprint
        self._sensor = None
        self._data = None

    @property
    def _default_obs_type(self) -> np.dtype:
        obs_space = self._get_observation_space()
        if isinstance(obs_space, spaces.Box):
            return obs_space.dtype
        return np.uint8

    @property
    def _default_obs(self) -> np.ndarray:
        return np.zeros(self._config.shape, dtype=self._default_obs_type)

    @abstractmethod
    def _get_observation_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def _update_data(self, data) -> None:
        pass

    def get_observation_space(self) -> Dict:
        return {self._config.key: self._get_observation_space()}

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        obs = {self._config.key: (self._data if self._data is not None else self._default_obs)}
        info = {}
        return obs, info

    def destroy(self) -> None:
        self._data = None
        if self._sensor is not None:
            self._sensor.destroy()

    def reset(self, ego: carla.Actor) -> None:
        self._sensor = self._world.spawn_unmanaged_actor(self._transform, self._blueprint, attach_to=ego)
        self._sensor.listen(self._update_data)
        # print(f"[SensorHandler] Spawned sensor {self._config.key} for ego id {ego.id}")


class CameraHandler(SensorHandler):
    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

    def _update_data(self, data) -> None:
        camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)
        camera_data = np.reshape(camera_data, (data.height, data.width, 4))
        camera_data = camera_data[:, :, :3]
        camera_data = camera_data[:, :, ::-1].copy()
        self._data = camera_data
        
    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        if self._data is None:
            return {self._config.key: self._default_obs}, {}

        cur_time_step = self._world.get_time_step()
        actor_id = self._sensor.parent.id if self._sensor is not None else 0
        os.makedirs(f"data/camera_frames/vehicle_{actor_id}", exist_ok=True)
        cv2.imwrite(f"data/camera_frames/vehicle_{actor_id}/camera_{cur_time_step:06d}.png", self._data)

        obs = {self._config.key: self._data}
        info = {}
        return obs, info


class LidarHandler(SensorHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._obs_range = config.attributes.range
        self._lidar_z = config.transform.z
        self._lidar_bin = config.lidar_bin
        self._ego_offset = config.ego_offset

    def _update_data(self, data) -> None:
        self._data = data

    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

    def get_observation(self, env_state: Dict) -> Tuple[Dict]:
        if self._data is None:
            return {self._config.key: self._default_obs}, {}

        points = np.frombuffer(self._data.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
        points = points[np.linalg.norm(points[:, :3], axis=1) <= self._obs_range]
        points[1, :] = -points[1, :]

        intensities = np.interp(points[:, 3], (points[:, 3].min(), points[:, 3].max()), (0, 1))
        colors = (intensities[:, np.newaxis] * np.array([[255, 0, 0]])).astype(np.uint8)

        y_bins = np.arange(
            -(self._obs_range - self._ego_offset),
            self._ego_offset + self._lidar_bin,
            self._lidar_bin,
        )
        x_bins = np.arange(-self._obs_range / 2, self._obs_range / 2 + self._lidar_bin, self._lidar_bin)
        z_bins = [-self._lidar_z - 1, -self._lidar_z + 0.25, 1]
        lidar, _ = np.histogramdd(points[:, :3], bins=(x_bins, y_bins, z_bins))

        lidar = lidar[: self._config.shape[0], : self._config.shape[1], :2]
        ground_mask = lidar[:, :, 0] > 0
        obstacle_mask = lidar[:, :, 1] > 0

        image = np.zeros((lidar.shape[0], lidar.shape[1], 3), dtype=np.uint8)
        image[ground_mask] = colors[: ground_mask.sum()]
        image[obstacle_mask] = np.array([0, 255, 0], dtype=np.uint8)
        image = np.flip(image, axis=0)
        
        cur_time_step = self._world.get_time_step()
        actor_id = self._sensor.parent.id if self._sensor is not None else 0
        os.makedirs(f"data/lidar_frames/vehicle_{actor_id}", exist_ok=True)
        cv2.imwrite(f"data/lidar_frames/vehicle_{actor_id}/lidar_{cur_time_step:06d}.png", image)

        obs = {self._config.key: image}
        info = {}
        return obs, info


class CollisionHandler(SensorHandler):
    def get_observation(self, env_state):
        # 把obs保存到本地文件，方便调试
        cur_time_step = self._world.get_time_step()
        actor_id = self._sensor.parent.id if self._sensor is not None else 0
        os.makedirs(f"data/collision_frames/vehicle_{actor_id}", exist_ok=True)
        np.save(f"data/collision_frames/vehicle_{actor_id}/collision_{cur_time_step:06d}.npy", self._data if self._data is not None else np.zeros(self._config.shape, dtype=self._default_obs_type))
        return super().get_observation(env_state)   
    
    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=np.inf, shape=self._config.shape, dtype=np.float32)

    def _update_data(self, data) -> None:
        impulse = data.normal_impulse
        collision_intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._data = collision_intensity * np.ones(self._config.shape)