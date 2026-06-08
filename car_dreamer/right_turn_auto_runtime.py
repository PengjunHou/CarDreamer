from __future__ import annotations

import carla
from agents.navigation.basic_agent import BasicAgent
from runtime_logging import get_runtime_logger

from .toolkit import get_vehicle_pos
from .v2v_comm_mixin import GROUP_ID, RECEIVED_BUFFER_SIZE, V2VCommMixin  # noqa: F401  (re-exported)


RUNTIME_LOGGER = get_runtime_logger("car_dreamer.runtime")


class RightTurnAutoRuntimeMixin(V2VCommMixin):
    """Right-turn-auto specifics layered on the reusable V2V communication mixin.

    Keeps only what is specific to this task: ``BasicAgent`` ego control, all-green
    traffic lights, and right-turn actor-flow cleanup. Cooperative vehicles are declared
    in ``scenario_actors`` (any vehicle with a ``start`` point) and registered via
    ``V2VCommMixin._register_cooperative_candidate``.
    """

    def _configure_traffic_lights(self) -> None:
        for tl in self._world.carla_actors(actor_type="traffic_light"):
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(9999)
            tl.set_red_time(0)
            tl.set_yellow_time(0)

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
