from __future__ import annotations

import carla
import numpy as np
from agents.navigation.basic_agent import BasicAgent
from runtime_logging import get_runtime_logger

from .toolkit import get_vehicle_pos
from .v2v_comm_mixin import GROUP_ID, RECEIVED_BUFFER_SIZE, V2VCommMixin  # noqa: F401  (re-exported)


RUNTIME_LOGGER = get_runtime_logger("car_dreamer.runtime")


class RightTurnAutoRuntimeMixin(V2VCommMixin):
    """Right-turn-auto specifics layered on the reusable V2V communication mixin.

    Keeps only what is specific to this task: ``BasicAgent`` ego control, fixed
    ``group_spawn_points`` cooperative vehicles (overriding the generic near-ego
    spawner), all-green traffic lights, and right-turn actor-flow cleanup.
    """

    def _configure_traffic_lights(self) -> None:
        for tl in self._world.carla_actors(actor_type="traffic_light"):
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(9999)
            tl.set_red_time(0)
            tl.set_yellow_time(0)

    def _spawn_cooperative_vehicles(self) -> None:
        """Override the near-ego default: spawn cooperative vehicles at the fixed
        ``group_spawn_points`` (stationary). Every vehicle is camera-equipped; each
        independently joins the cooperative candidate pool with probability
        ``coop_participation_prob``.
        """
        self.groups.setdefault(GROUP_ID, set())
        self.groups[GROUP_ID].add(int(self.ego.id))
        spawn_points = self._config.group_spawn_points
        assert spawn_points is not None and len(spawn_points) >= self.num_group_vehs, (
            "Not enough spawn points for the number of group vehicles"
        )
        participation_prob = float(getattr(self, "coop_participation_prob", 0.5))
        for spawn_point in spawn_points[: self.num_group_vehs]:
            transform = carla.Transform(
                carla.Location(*spawn_point[:3]),
                carla.Rotation(yaw=spawn_point[3]),
            )
            vehicle = self._world.spawn_actor(transform=transform)
            # Every non-flow vehicle is camera-equipped (observer attached here).
            self._create_group_observer(vehicle)
            self.group_vehs.append(vehicle)
            self._cache_actor(vehicle)
            # Randomly decide if this camera vehicle participates in cooperative
            # perception this episode. Only participants become candidate collaborators
            # (i.e. enter the group, share over V2V, and appear in the policy graph).
            if np.random.random() < participation_prob:
                self.coop_participant_ids.add(int(vehicle.id))
                self.groups[GROUP_ID].add(int(vehicle.id))
        RUNTIME_LOGGER.info(
            "Generated group vehicles count=%d ids=%s participants=%s group_members=%s",
            len(self.group_vehs),
            [int(vehicle.id) for vehicle in self.group_vehs],
            sorted(self.coop_participant_ids),
            sorted(self.groups.get(GROUP_ID, set())),
        )

    def _setup_basic_agent(self) -> None:
        self.ego_end = self._config.lane_end_point
        ego_transform = carla.Transform(
            carla.Location(*self.ego_end[:3]),
            carla.Rotation(yaw=self.ego_end[3]),
        )
        self.agent = BasicAgent(self.ego)
        self.agent.set_destination(ego_transform.location)
        self._cache_actor(self.ego)

    def _cleanup_actor_flow(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(vehicle)
            if y > -81.2 or x < -38.4 or x > 31.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
