from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos


class CarlaRoundaboutEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the roundabout and avoid collision.

    **Provided Tasks**: ``carla_roundabout``

    In addition to the single car flow from :class:`CarlaWptFixedEnv`, this env can
    populate the scene with scattered background traffic and pedestrians, controlled
    by ``num_vehicles`` and ``num_pedestrians`` in the task config (0 disables each).
    """

    def on_reset(self) -> None:
        super().on_reset()
        num_vehicles = int(getattr(self._config, "num_vehicles", 0))
        if num_vehicles > 0:
            self._world.spawn_auto_actors(num_vehicles)
        num_pedestrians = int(getattr(self._config, "num_pedestrians", 0))
        if num_pedestrians > 0:
            self._world.spawn_walkers(num_pedestrians)

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(vehicle)
            if (y < 0.0 and x < -39.8) or y < -47.2 or y > 46.0 or x > 44.8:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
        super().on_step()
