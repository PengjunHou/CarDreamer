"""Config-driven background actors (vehicles + pedestrians) for any task.

A task declares how many background vehicles and pedestrians to add — and optionally their
start / destination — under ``env.scenario_actors`` (typically in
``car_dreamer/configs/tasks/<task>.yaml``). :class:`ScenarioActorManager` spawns them on reset and
maintains pedestrian arrival behavior each step. It is wired into ``CarlaBaseEnv`` once, so every task
supports it via config (no-op when no config is present).

Control:
* vehicles -> CARLA Traffic Manager autopilot; with a ``destination`` the TM routes them there legally
  (``tm.set_path``) and they keep flowing afterwards.
* pedestrians -> ``WalkerAIController.go_to_location`` on the navigation mesh.

Schema (each list entry is a group; omit empty groups — the ``Config`` class disallows empty lists)::

    env:
      scenario_actors:
        vehicles:
          - count: 10                       # random spawn, TM roam
          - count: 3
            destination: [x, y, z]          # random starts, routed toward destination
          - start: [x, y, z, yaw]           # one scripted car
            destination: [x, y, z]
            target_speed: 30                # optional km/h
            ignore_lights: false            # optional
        pedestrians:
          - count: 20                       # random nav spawn + random walk
          - start: [x, y, z]
            destination: [x, y, z]          # go_to_location
            max_speed: 1.4                  # optional m/s
            on_arrival: stop                # optional: keep (default) | stop | loop
        cross_factor: 0.1                   # optional global jaywalk probability

Rule: if ``start`` is present the spec is a single actor at that point; otherwise ``count`` (default 1)
actors are placed at random points. ``destination`` omitted -> CARLA default (roam / random walk).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import carla
from runtime_logging import get_runtime_logger

SCENARIO_LOGGER = get_runtime_logger("car_dreamer.scenario")

ARRIVAL_THRESHOLD_M = 2.0


def _as_float_list(value) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(x) for x in value]


DEFAULT_TARGET_SPEED = 25.0


def _parse_vehicle_groups(groups) -> List[Dict]:
    specs: List[Dict] = []
    for group in groups or ():
        g = dict(group)
        start = _as_float_list(g.get("start"))
        destination = _as_float_list(g.get("destination"))
        target_speed = g.get("target_speed")
        is_candidate = start is not None
        specs.append(
            {
                "count": 1 if is_candidate else int(g.get("count", 1)),
                "start": start,
                "destination": destination,
                "target_speed": float(target_speed) if target_speed is not None else DEFAULT_TARGET_SPEED,
                "ignore_lights": bool(g.get("ignore_lights", False)),
                # A vehicle with an explicit start is a V2V cooperative candidate (camera-equipped,
                # selectable by the policy). Without a destination it stays parked.
                "is_candidate": is_candidate,
                "stationary": is_candidate and destination is None,
            }
        )
    return specs


def _parse_pedestrian_groups(groups) -> List[Dict]:
    """Expand pedestrian groups into a flat per-walker spec list."""
    specs: List[Dict] = []
    for group in groups or ():
        g = dict(group)
        start = _as_float_list(g.get("start"))
        destination = _as_float_list(g.get("destination"))
        max_speed = float(g.get("max_speed", 1.4))
        on_arrival = str(g.get("on_arrival", "keep"))
        count = 1 if start is not None else int(g.get("count", 1))
        for _ in range(max(count, 0)):
            specs.append(
                {"start": start, "destination": destination, "max_speed": max_speed, "on_arrival": on_arrival}
            )
    return specs


def parse_scenario_specs(scenario_config) -> Dict[str, object]:
    """Normalize an ``env.scenario_actors`` config (``Config`` or plain dict) into specs.

    :return: ``{"vehicles": [group...], "pedestrians": [per-walker...], "cross_factor": float}``.
    """
    cfg = dict(scenario_config or {})
    return {
        "vehicles": _parse_vehicle_groups(cfg.get("vehicles", ())),
        "pedestrians": _parse_pedestrian_groups(cfg.get("pedestrians", ())),
        "cross_factor": float(cfg.get("cross_factor", 0.1)),
    }


def _to_transform(xyzyaw: Optional[List[float]]) -> Optional[carla.Transform]:
    if xyzyaw is None:
        return None
    x, y, z = float(xyzyaw[0]), float(xyzyaw[1]), float(xyzyaw[2]) if len(xyzyaw) > 2 else 0.0
    yaw = float(xyzyaw[3]) if len(xyzyaw) > 3 else 0.0
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))


def _to_location(xyz: Optional[List[float]]) -> Optional[carla.Location]:
    if xyz is None:
        return None
    z = float(xyz[2]) if len(xyz) > 2 else 0.0
    return carla.Location(x=float(xyz[0]), y=float(xyz[1]), z=z)


class ScenarioActorManager:
    """Spawns and maintains the config-declared background vehicles and pedestrians."""

    def __init__(self, world_manager, scenario_config, cooperative_hook=None):
        self._world = world_manager
        self._specs = parse_scenario_specs(scenario_config)
        self.enabled = bool(self._specs["vehicles"] or self._specs["pedestrians"])
        # Optional callback invoked with each spawned "candidate" vehicle (one with a `start`).
        # V2V-enabled envs use it to attach a camera observer and add the vehicle to the
        # cooperative candidate pool; non-V2V envs pass None (the vehicle is simply placed).
        self._cooperative_hook = cooperative_hook
        self._walker_records: List[Dict] = []
        self._arrived: set = set()

    def reset_spawn(self) -> None:
        """Spawn all configured actors. Call from the env ``on_reset`` (async window)."""
        self._walker_records = []
        self._arrived = set()
        if not self.enabled:
            return

        spawned_vehicles = 0
        spawned_candidates = 0
        for spec in self._specs["vehicles"]:
            start = _to_transform(spec["start"])
            destination = _to_location(spec["destination"])
            if spec["is_candidate"]:
                # One placed vehicle; stationary if no destination, else TM-routed.
                actor = self._world.spawn_scenario_vehicle(
                    start=start,
                    destination=destination,
                    target_speed=spec["target_speed"],
                    ignore_lights=spec["ignore_lights"],
                    stationary=spec["stationary"],
                )
                if actor is not None:
                    spawned_vehicles += 1
                    if self._cooperative_hook is not None:
                        self._cooperative_hook(actor)
                        spawned_candidates += 1
                continue
            # count-only background traffic (no camera, not a candidate)
            for _ in range(max(int(spec["count"]), 0)):
                actor = self._world.spawn_scenario_vehicle(
                    start=None,
                    destination=destination,
                    target_speed=spec["target_speed"],
                    ignore_lights=spec["ignore_lights"],
                )
                if actor is not None:
                    spawned_vehicles += 1

        pedestrian_specs = [
            {
                "start": _to_transform(spec["start"]),
                "destination": _to_location(spec["destination"]),
                "max_speed": spec["max_speed"],
                "on_arrival": spec["on_arrival"],
            }
            for spec in self._specs["pedestrians"]
        ]
        if pedestrian_specs:
            self._walker_records = self._world.spawn_scenario_walkers(
                pedestrian_specs, cross_factor=self._specs["cross_factor"]
            )

        SCENARIO_LOGGER.info(
            "Scenario actors spawned vehicles=%d (cooperative candidates=%d) pedestrians=%d",
            spawned_vehicles,
            spawned_candidates,
            len(self._walker_records),
        )

    def step_update(self) -> None:
        """Pedestrian arrival handling. Call from the env ``on_step`` (after tick)."""
        if not self._walker_records:
            return
        for record in self._walker_records:
            on_arrival = record.get("on_arrival", "keep")
            if on_arrival == "keep":
                continue
            destination = record.get("destination")
            controller = record.get("controller")
            walker = record.get("walker")
            if destination is None or controller is None or walker is None:
                continue
            try:
                location = walker.get_location()
            except Exception:  # noqa: BLE001
                continue
            if math.hypot(location.x - destination.x, location.y - destination.y) > ARRIVAL_THRESHOLD_M:
                continue
            controller_id = controller.id
            try:
                if on_arrival == "stop":
                    if controller_id not in self._arrived:
                        controller.stop()
                        self._arrived.add(controller_id)
                elif on_arrival == "loop":
                    controller.go_to_location(destination)
            except Exception:  # noqa: BLE001
                pass
