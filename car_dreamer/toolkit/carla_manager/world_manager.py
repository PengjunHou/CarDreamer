import time
from functools import wraps
from typing import Callable, Dict, List, Union

import carla
import numpy as np
from runtime_logging import get_runtime_logger

from .utils import ActorActionDict, ActorPolygonDict, ActorTransformDict, Command
from .vehicle_manager import VehicleManager


WORLD_LOGGER = get_runtime_logger("car_dreamer.world")


def cached_step_wise(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        cache_key = (func.__name__,) + tuple(args) + tuple(kwargs.items())
        if not hasattr(self, "_cache") or self._cache["step"] != self._time_step:
            self._cache = {"step": self._time_step}
        if cache_key not in self._cache:
            self._cache[cache_key] = func(self, *args, **kwargs)
        return self._cache[cache_key]

    return wrapper


class WorldManager:
    """
    The class to manage the world in CARLA.
    You can spawn various actors using this class.
    The actors spawned by this class will be automatically destroyed when reset.
    This class also provides methods to get information about these actors.
    """

    def __init__(self, env_config):
        self._config = env_config.world
        self._env_config = env_config

        WORLD_LOGGER.info("Connecting to Carla server at port=%s", self._config.carla_port)
        self._client = carla.Client("127.0.0.1", self._config.carla_port)
        self._client.set_timeout(20.0)
        self._world = self._client.load_world(self._config.town)
        self._map = self._world.get_map()
        WORLD_LOGGER.info("Loaded CARLA map town=%s", self._config.town)

        settings = self._world.get_settings()
        settings.synchronous_mode = False
        settings.actor_active_distance = self._config.actor_active_distance
        settings.fixed_delta_seconds = self._config.fixed_delta_seconds
        self._world.apply_settings(settings)
        self._settings = settings

        self._tm_port = self._config.carla_port + 6000
        self._vehicle_manager = VehicleManager(self._client, self._tm_port, self._config.traffic)

        self._on_reset = None
        self._apply_control = None
        self._on_step = None
        self.actor_dict = {}
        self._autopilot_actor_ids = set()
        # Pedestrians (walker bodies + their AI controllers) are tracked separately
        # from actor_dict so they never enter the vehicle-oriented BEV / visibility
        # pipeline (which assumes vehicle bounding boxes), but are still destroyed on reset.
        self._walker_actors = {}
        self._time_step = 0

        self._ego_planner = None

    def on_reset(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to be called when the environment is reset.
        If called multiple times, it will overwrite the previous callback.
        """
        self._on_reset = callback

    def on_step(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to be called when the environment steps.
        If called multiple times, it will overwrite the previous callback.
        """
        self._on_step = callback

    def reset(self) -> None:
        # destroy all actors
        self._time_step = 0
        self._cache = {"step": self._time_step}
        self._destroy_walkers()
        self._client.apply_batch_sync([carla.command.DestroyActor(id) for id in self.actor_dict])
        self.actor_dict = {}
        self._autopilot_actor_ids = set()

        self._set_synchronous_mode(False)

        if self._on_reset is not None:
            self._on_reset()

        self._set_synchronous_mode(True)
        # This prevents some synchronization bugs
        time.sleep(1)

    def step(self) -> None:
        self._time_step += 1
        self._world.tick()
        if self._on_step is not None:
            self._on_step()

    def warmup(self, ticks: int) -> None:
        """
        Advance the CARLA world without env-level callbacks or logical step accounting.

        This lets Traffic Manager/autopilot actors settle after reset while keeping the
        first Gym step, WAM/V2V process, rewards, observations, and frame numbering at zero.
        """
        ticks = max(int(ticks), 0)
        if ticks <= 0:
            return
        WORLD_LOGGER.info("Warmup CARLA world ticks=%d without env callbacks", ticks)
        for _ in range(ticks):
            self._world.tick()
        self._cache = {"step": self._time_step}

    def get_time_step(self) -> int:
        """
        Get the current time step of the world.
        """
        return self._time_step 

    def get_blueprint_library(self, pattern_filter: str, attribute_filter: Dict[str, str] = None) -> carla.BlueprintLibrary:
        """
        Get blueprint library based on the pattern filter and attribute filter.
        """
        bps = self._world.get_blueprint_library().filter(pattern_filter)
        if attribute_filter is not None:
            for name, value in attribute_filter.items():
                bps = bps.filter_by_attribute(name, value)
        return bps

    def get_blueprint(self, pattern_filter: str, attribute_filter: Dict[str, str] = None) -> carla.ActorBlueprint:
        """
        Randomly get a blueprint from the library based on the pattern filter and attribute filter.
        """
        bps = self.get_blueprint_library(pattern_filter, attribute_filter)
        assert len(bps) > 0, f"No blueprint found for filter {pattern_filter} {attribute_filter}"
        return np.random.choice(bps)

    def get_spawn_points(self) -> List[carla.Transform]:
        """
        Get spawn points of the map.
        """
        return self._map.get_spawn_points()

    def get_random_spawn_point(self) -> carla.Transform:
        """
        Get a random spawn point of the map.
        """
        spawn_points = self.get_spawn_points()
        assert len(spawn_points) > 0, "No spawn points found"
        return np.random.choice(spawn_points)

    def try_spawn_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
    ) -> Union[carla.Actor, None]:
        """
        Spawn an actor with the given blueprint and transform.

        :param transform: if None, use a random spawn point.
        :param blueprint: if None, use vehicle.audi* with number_of_wheels in 4 as default.

        :return: the spawned actor. If fails, return None.
        """
        if transform is None:
            transform = self.get_random_spawn_point()
        if blueprint is None:
            blueprint = self.get_blueprint("vehicle.audi*", {"number_of_wheels": "4"})
            if blueprint.has_attribute("color"):
                color = np.random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            blueprint.set_attribute("role_name", "hero")
        actor = self._world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self.actor_dict[actor.id] = actor
        return actor

    def spawn_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
        max_try_time: int = None,
    ) -> carla.Actor:
        """
        Equivalent to ``try_spawn_actor(transform, blueprint)``, but retry if failed.

        :param max_try_time: if None, try until success, else raise an exception after ``max_try_time``.

        .. seealso:: :py:meth:`try_spawn_actor`
        """
        actor = self.try_spawn_actor(transform, blueprint)
        try_time = 0
        while actor is None and (max_try_time is None or try_time < max_try_time):
            WORLD_LOGGER.warning("Failed to spawn actor, retrying attempt=%d", try_time + 1)
            time.sleep(0.1)
            actor = self.try_spawn_actor(transform, blueprint)
            try_time += 1
        if actor is None:
            raise Exception("Failed to spawn actor")
        return actor

    def spawn_unmanaged_actor(self, transform: carla.Transform, blueprint: carla.ActorBlueprint, **kwargs) -> carla.Actor:
        """
        Spawn an actor with the given blueprint and transform.
        Actors spawned by this method will be omitted by this manager.
        That is, they will not be included when retrieving actor information or destroyed when reset.
        This is useful when creating sensors for :py:class:`car_dreamer.toolkit.observer.handlers.SensorHandler`.
        """
        return self._world.spawn_actor(blueprint, transform, **kwargs)

    def spawn_auto_actors(
        self,
        n: int,
        transforms: List[carla.Transform] = None,
        blueprints: carla.BlueprintLibrary = None,
    ) -> List[carla.Actor]:
        """
        Spawn ``n`` actors that are automatically controlled by autopilot.

        :param n: number of actors to spawn.
        :param transforms: if None, use random spawn points.
        :param blueprints: if None, use vehicle.* with number_of_wheels 4 as default.

        :return: a list of spawned actors, note that the length of the list may be less than n.
        """
        if transforms is None:
            transforms = self.get_spawn_points()
        if blueprints is None:
            blueprints = self.get_blueprint_library("vehicle.*", {"number_of_wheels": "4"})
        batch = []
        actor_list = []
        np.random.shuffle(transforms)
        for transform in transforms[: min(n, len(transforms))]:
            bp = np.random.choice(blueprints)
            if bp.has_attribute("color"):
                color = np.random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            if bp.has_attribute("driver_id"):
                driver_id = np.random.choice(bp.get_attribute("driver_id").recommended_values)
                bp.set_attribute("driver_id", driver_id)
                bp.set_attribute("role_name", "autopilot")
            batch.append(carla.command.SpawnActor(bp, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True, self._tm_port)))
        for response in self._client.apply_batch_sync(batch, False):
            if response.error:
                WORLD_LOGGER.warning("Batch spawn response error: %s", response.error)
            else:
                actor = self._world.get_actor(response.actor_id)
                actor_list.append(actor)
                self.actor_dict[actor.id] = actor
                self._autopilot_actor_ids.add(actor.id)
                self._vehicle_manager.set_auto_lane_change(actor, self._config.auto_lane_change)
                self._vehicle_manager.set_lane_change_percent(actor, left=100.0, right=100.0)
                if "background_speed" in self._config:
                    self._vehicle_manager.set_desired_speed(actor, self._config.background_speed)
        return actor_list

    def _destroy_walkers(self) -> None:
        """Stop walker AI controllers and destroy all tracked pedestrians."""
        if not self._walker_actors:
            return
        for actor in self._walker_actors.values():
            if "controller.ai.walker" in actor.type_id:
                try:
                    actor.stop()
                except Exception:  # noqa: BLE001
                    pass
        self._client.apply_batch_sync([carla.command.DestroyActor(id) for id in self._walker_actors])
        self._walker_actors = {}

    def spawn_walkers(
        self,
        n: int,
        run_speed: float = 1.4,
        cross_factor: float = 0.1,
    ) -> List[carla.Actor]:
        """
        Spawn ``n`` pedestrians that wander the navigation mesh via AI controllers.

        Walker bodies and their ``controller.ai.walker`` actors are tracked in a
        dedicated registry (not ``actor_dict``), so they stay out of the vehicle-only
        BEV/visibility pipeline but are still stopped and destroyed on the next reset.
        Intended to be called from ``on_reset`` (the world is asynchronous there).

        :param n: number of pedestrians to spawn.
        :param run_speed: maximum walking speed in m/s.
        :param cross_factor: probability that pedestrians cross roads (0..1).

        :return: a list of spawned walker actors (length may be less than ``n``).
        """
        if n <= 0:
            return []

        walker_bps = self.get_blueprint_library("walker.pedestrian.*")
        if len(walker_bps) == 0:
            WORLD_LOGGER.warning("No walker blueprints found; skipping pedestrian spawn.")
            return []
        controller_bp = self._world.get_blueprint_library().find("controller.ai.walker")
        self._world.set_pedestrians_cross_factor(float(cross_factor))

        # 1) spawn walker bodies at random navigation-mesh points, retrying the
        #    shortfall since random points often collide (occupied / too close).
        walker_ids = []
        attempts = 0
        max_attempts = n * 5
        while len(walker_ids) < n and attempts < max_attempts:
            batch = []
            for _ in range(n - len(walker_ids)):
                location = self._world.get_random_location_from_navigation()
                if location is None:
                    continue
                bp = np.random.choice(walker_bps)
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                batch.append(carla.command.SpawnActor(bp, carla.Transform(location)))
            attempts += len(batch)
            if not batch:
                break
            for response in self._client.apply_batch_sync(batch, True):
                if response.error:
                    WORLD_LOGGER.debug("Walker spawn skipped: %s", response.error)
                else:
                    walker_ids.append(response.actor_id)

        # 2) batch-spawn an AI controller attached to each walker body
        controller_batch = [
            carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id)
            for walker_id in walker_ids
        ]
        controller_ids = []
        for response in self._client.apply_batch_sync(controller_batch, True):
            if response.error:
                WORLD_LOGGER.debug("Walker controller spawn skipped: %s", response.error)
            else:
                controller_ids.append(response.actor_id)

        # 3) let the server register the new actors before starting the controllers
        self._world.wait_for_tick()

        # 4) track pedestrians (separately from vehicles), send each to a random target
        walkers = []
        for walker_id in walker_ids:
            actor = self._world.get_actor(walker_id)
            if actor is not None:
                self._walker_actors[actor.id] = actor
                walkers.append(actor)
        for controller_id in controller_ids:
            controller = self._world.get_actor(controller_id)
            if controller is None:
                continue
            self._walker_actors[controller.id] = controller
            try:
                controller.start()
                controller.go_to_location(self._world.get_random_location_from_navigation())
                controller.set_max_speed(float(run_speed))
            except Exception as exc:  # noqa: BLE001
                WORLD_LOGGER.debug("Failed to start walker controller %s: %s", controller_id, exc)

        WORLD_LOGGER.info(
            "Spawned pedestrians requested=%d walkers=%d controllers=%d",
            n,
            len(walkers),
            len(controller_ids),
        )
        return walkers

    def spawn_scenario_vehicle(
        self,
        start: carla.Transform = None,
        destination: carla.Location = None,
        target_speed: float = None,
        ignore_lights: bool = False,
        stationary: bool = False,
        blueprint: carla.ActorBlueprint = None,
    ) -> Union[carla.Actor, None]:
        """
        Spawn one vehicle for the config-driven scenario-actor module.

        :param start: spawn transform; if None, a random map spawn point is used.
        :param destination: if given, the Traffic Manager routes the vehicle toward it
            (``tm.set_path``); the vehicle drives there legally (lanes, lights) and keeps
            going afterwards. If None, the vehicle roams (TM default).
        :param target_speed: desired speed (km/h); falls back to ``world.background_speed``.
        :param ignore_lights: if True, the vehicle runs red lights (100%).
        :param stationary: if True, the vehicle is parked (no autopilot / no routing) -- used
            for cooperative observer vehicles that should stay put.
        :param blueprint: if None, a random ``vehicle.*`` 4-wheel blueprint is used.

        The vehicle is registered in ``actor_dict`` and destroyed on the next reset.
        """
        if blueprint is None:
            blueprint = self.get_blueprint("vehicle.*", {"number_of_wheels": "4"})
        transform = start if start is not None else self.get_random_spawn_point()
        vehicle = self.try_spawn_actor(transform, blueprint)
        if vehicle is None:
            WORLD_LOGGER.debug("Scenario vehicle spawn failed (occupied?) at %s", transform.location)
            return None
        if stationary:
            return vehicle  # parked observer: no autopilot, no route
        vehicle.set_autopilot(True, self._tm_port)
        self._autopilot_actor_ids.add(vehicle.id)
        tm = self._vehicle_manager._tm
        self._vehicle_manager.set_auto_lane_change(vehicle, self._config.auto_lane_change)
        if target_speed is not None:
            self._vehicle_manager.set_desired_speed(vehicle, float(target_speed))
        elif "background_speed" in self._config:
            self._vehicle_manager.set_desired_speed(vehicle, self._config.background_speed)
        if ignore_lights:
            tm.ignore_lights_percentage(vehicle, 100.0)
        if destination is not None:
            try:
                tm.set_path(vehicle, [destination])
            except Exception as exc:  # noqa: BLE001
                WORLD_LOGGER.warning("set_path failed for scenario vehicle %s: %s", vehicle.id, exc)
        return vehicle

    def spawn_scenario_walkers(
        self,
        specs: List[Dict],
        cross_factor: float = 0.1,
    ) -> List[Dict]:
        """
        Spawn one pedestrian per spec for the scenario-actor module (batched).

        Each spec is a dict with optional ``start`` (``carla.Transform``/``carla.Location``;
        random navigation point if absent), ``destination`` (``carla.Location``; random walk
        if absent), ``max_speed`` (m/s) and ``on_arrival`` (passed through to the manager).
        Reuses the proven body -> controller -> wait_for_tick -> start sequence; bodies and
        controllers are tracked in ``_walker_actors`` (destroyed on reset).

        :return: a list of manager records ``{walker, controller, destination, on_arrival,
            max_speed}`` for per-step maintenance.
        """
        if not specs:
            return []
        walker_bps = self.get_blueprint_library("walker.pedestrian.*")
        if len(walker_bps) == 0:
            WORLD_LOGGER.warning("No walker blueprints found; skipping pedestrian spawn.")
            return []
        controller_bp = self._world.get_blueprint_library().find("controller.ai.walker")
        self._world.set_pedestrians_cross_factor(float(cross_factor))

        # 1) spawn walker bodies (explicit start transform or a random navigation point)
        spawn_specs: List[Dict] = []
        batch = []
        for spec in specs:
            start = spec.get("start")
            if start is None:
                location = self._world.get_random_location_from_navigation()
                if location is None:
                    continue
                transform = carla.Transform(location)
            elif isinstance(start, carla.Transform):
                transform = start
            else:
                transform = carla.Transform(start)
            bp = np.random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            batch.append(carla.command.SpawnActor(bp, transform))
            spawn_specs.append(spec)

        walker_ids = []
        walker_spec_by_id: Dict[int, Dict] = {}
        for spec, response in zip(spawn_specs, self._client.apply_batch_sync(batch, True)):
            if response.error:
                WORLD_LOGGER.debug("Walker spawn skipped: %s", response.error)
            else:
                walker_ids.append(response.actor_id)
                walker_spec_by_id[response.actor_id] = spec

        # 2) attach an AI controller to each surviving walker body
        controller_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
        controller_walker_pairs = []
        for wid, response in zip(walker_ids, self._client.apply_batch_sync(controller_batch, True)):
            if response.error:
                WORLD_LOGGER.debug("Walker controller spawn skipped: %s", response.error)
            else:
                controller_walker_pairs.append((response.actor_id, wid))

        # 3) let the server register the new actors before starting the controllers
        self._world.wait_for_tick()

        for wid in walker_ids:
            actor = self._world.get_actor(wid)
            if actor is not None:
                self._walker_actors[actor.id] = actor

        # 4) start each controller and send it to its destination (or a random point)
        records: List[Dict] = []
        for controller_id, walker_id in controller_walker_pairs:
            controller = self._world.get_actor(controller_id)
            if controller is None:
                continue
            self._walker_actors[controller.id] = controller
            spec = walker_spec_by_id.get(walker_id, {})
            destination = spec.get("destination")
            target = destination if destination is not None else self._world.get_random_location_from_navigation()
            max_speed = float(spec.get("max_speed", 1.4))
            try:
                controller.start()
                controller.go_to_location(target)
                controller.set_max_speed(max_speed)
            except Exception as exc:  # noqa: BLE001
                WORLD_LOGGER.debug("Failed to start walker controller %s: %s", controller_id, exc)
                continue
            records.append(
                {
                    "walker": self._world.get_actor(walker_id),
                    "controller": controller,
                    "destination": destination,
                    "on_arrival": spec.get("on_arrival", "keep"),
                    "max_speed": max_speed,
                }
            )
        WORLD_LOGGER.info("Spawned scenario pedestrians requested=%d bodies=%d managed=%d", len(specs), len(walker_ids), len(records))
        return records

    def try_spawn_aggresive_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
    ) -> Union[carla.Actor, None]:
        """
        Similar to ``try_spawn_actor(transform, blueprint)``.
        But the actor will be automatically controlled by autopilot and ignore traffic lights and other vehicles.

        .. seealso:: :py:meth:`try_spawn_actor`
        """
        vehicle = self.try_spawn_actor(transform, blueprint)
        if vehicle is None:
            return None
        vehicle.set_autopilot(True, self._tm_port)
        self._autopilot_actor_ids.add(vehicle.id)
        self._vehicle_manager.set_auto_lane_change(vehicle, True)
        if "background_speed" in self._config:
            self._vehicle_manager.set_desired_speed(vehicle, self._config.background_speed)
        self._vehicle_manager._tm.ignore_lights_percentage(vehicle, 100)
        self._vehicle_manager._tm.ignore_vehicles_percentage(vehicle, 0)
        return vehicle

    def destroy_actor(self, actor_id: int) -> None:
        """
        Destroy an actor. Call this method if you want to manually destroy an actor spawned by this manager.

        .. warning::
           Do not call this method for actors spawned by :py:meth:`spawn_unmanaged_actor`.
           Directly call :py:meth:`carla.Actor.destroy` instead.
        """
        actor = self.actor_dict.pop(actor_id)
        self._autopilot_actor_ids.discard(actor_id)
        actor.destroy()

    @property
    def traffic_manager_port(self) -> int:
        """
        Get the Traffic Manager port used for autopilot vehicles.
        """
        return self._tm_port

    def is_autopilot_actor(self, actor_id: int) -> bool:
        """
        Return whether this manager enabled Traffic Manager autopilot for the actor.
        """
        return int(actor_id) in self._autopilot_actor_ids

    def set_actor_autopilot(self, actor: carla.Actor, enabled: bool) -> None:
        """
        Toggle Traffic Manager autopilot and keep local bookkeeping in sync.
        """
        actor.set_autopilot(bool(enabled), self._tm_port)
        if enabled:
            self._autopilot_actor_ids.add(actor.id)
        else:
            self._autopilot_actor_ids.discard(actor.id)

    @property
    def actor_ids(self) -> List[int]:
        """
        Get the ids of all actors spawned by this manager.
        """
        return list(self.actor_dict.keys())

    @property
    def actors(self) -> List[carla.Actor]:
        """
        Get all actors spawned by this manager.
        """
        return list(self.actor_dict.values())

    @cached_step_wise
    def _get_actor_polygons(self) -> ActorPolygonDict:
        actor_polygons: ActorPolygonDict = {}

        for actor in self.actors:
            actor_transform = actor.get_transform()
            x = actor_transform.location.x
            y = actor_transform.location.y

            yaw = actor_transform.rotation.yaw * np.pi / 180

            # Get length and width of the bounding box
            bb = actor.bounding_box
            l, w = bb.extent.x, bb.extent.y

            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).T

            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).T + np.repeat([[x, y]], 4, axis=0)
            actor_polygons[actor.id] = poly.tolist()

        return actor_polygons

    @property
    def actor_polygons(self) -> ActorPolygonDict:
        """
        Get the bounding box polygons of all actors spawned by this manager.

        :return: a dictionary mapping actor IDs to their bounding box polygons.
        :rtype: dict[int, list[tuple[float, float]]]
        """
        return self._get_actor_polygons()

    @cached_step_wise
    def _get_actor_actions(self) -> ActorActionDict:
        actor_actions: ActorActionDict = {}

        for actor in self.actor_dict.values():
            try:
                actions = self._vehicle_manager._tm.get_all_actions(actor)
                actor_actions[actor.id] = [(Command(command), waypoint) for command, waypoint in actions]
            except Exception as e:  # noqa: F841
                pass

        return actor_actions

    @property
    def actor_actions(self) -> ActorActionDict:
        """
        Get the actions of all actors spawned by this manager.

        :return: a dictionary mapping vehicle IDs to their known actions.
        :rtype: dict[int, list[tuple[Command, carla.Waypoint]]]

        .. warning::
           Actors not controlled by autopilot will not have actions.
           They will not be included in the returned dictionary.
           And some actors may have an empty list if there is no known action.
        """
        return self._get_actor_actions()

    @cached_step_wise
    def _get_actor_transforms(self) -> ActorTransformDict:
        return {actor.id: actor.get_transform() for actor in self.actor_dict.values()}

    @property
    def actor_transforms(self) -> ActorTransformDict:
        """
        Get the transforms of all actors spawned by this manager.

        :return: a dictionary mapping actor IDs to their transforms.
        :rtype: dict[int, carla.Transform]
        """
        return self._get_actor_transforms()

    def _set_synchronous_mode(self, synchronous=True):
        self._settings.synchronous_mode = synchronous
        self._world.apply_settings(self._settings)
        self._vehicle_manager.set_synchronous_mode(synchronous)

    def _get_world(self):
        return self._world

    @property
    def carla_world(self):
        return self._get_world()

    def _get_map(self):
        return self._map

    @property
    def carla_map(self):
        return self._get_map()

    @cached_step_wise
    def _get_carla_actors(self, actor_type: str = "") -> List[carla.Actor]:
        filtered_actors = []
        carla_actors = self._world.get_actors()
        for actor in carla_actors:
            if actor_type in actor.type_id:
                filtered_actors.append(actor)
        return filtered_actors

    def carla_actors(self, actor_type: str = "") -> List[carla.Actor]:
        """
        Get all actors of a specific type directly through CARLA APIs.

        :param actor_type: the type of the actors to retrieve (e.g., 'vehicle', 'traffic_light').
        :return: a list of actors of the specified type.
        """
        return self._get_carla_actors(actor_type)
