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
        vehicle_start_mode: road          # exact | road | spawn_point
        vehicle_destination_mode: road    # exact | road | spawn_point
        preserve_vehicle_start_yaw: true
        vehicle_spawn_z_offset: 0.0       # optional; add to resolved vehicle start z before spawn
        vehicles:
          - count: 10                       # random spawn, TM roam
          - count: 3
            destination: [x, y, z]          # random starts, routed toward destination
          - start: [x, y, z, yaw]           # one scripted car
            destination: [x, y, z]
            target_speed: 30                # optional km/h
            ignore_lights: false            # optional
            autopilot_roam: false            # optional; no destination -> TM roam instead of parked
            vehicle_start_mode: exact        # optional per-vehicle override
            vehicle_spawn_z_offset: 0.5      # optional per-vehicle override
        pedestrian_start_mode: navmesh    # exact | navmesh
        pedestrian_destination_mode: navmesh
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
        autopilot_roam = bool(g.get("autopilot_roam", False))
        vehicle_start_mode = g.get("vehicle_start_mode", g.get("start_mode"))
        preserve_vehicle_start_yaw = g.get("preserve_vehicle_start_yaw")
        vehicle_spawn_z_offset = g.get("vehicle_spawn_z_offset")
        specs.append(
            {
                "count": 1 if is_candidate else int(g.get("count", 1)),
                "start": start,
                "destination": destination,
                "target_speed": float(target_speed) if target_speed is not None else DEFAULT_TARGET_SPEED,
                "ignore_lights": bool(g.get("ignore_lights", False)),
                # A vehicle with an explicit start is a V2V cooperative candidate (camera-equipped,
                # selectable by the policy). Without a destination it stays parked unless
                # autopilot_roam is enabled, in which case Traffic Manager lets it follow the road.
                "autopilot_roam": autopilot_roam,
                "vehicle_start_mode": str(vehicle_start_mode) if vehicle_start_mode is not None else None,
                "preserve_vehicle_start_yaw": (
                    bool(preserve_vehicle_start_yaw) if preserve_vehicle_start_yaw is not None else None
                ),
                "vehicle_spawn_z_offset": (
                    float(vehicle_spawn_z_offset) if vehicle_spawn_z_offset is not None else None
                ),
                "is_candidate": is_candidate,
                "stationary": is_candidate and destination is None and not autopilot_roam,
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
        "vehicle_start_mode": str(cfg.get("vehicle_start_mode", "exact")),
        "vehicle_destination_mode": str(cfg.get("vehicle_destination_mode", "exact")),
        "preserve_vehicle_start_yaw": bool(cfg.get("preserve_vehicle_start_yaw", True)),
        "vehicle_spawn_z_offset": float(cfg.get("vehicle_spawn_z_offset", 0.0)),
        "pedestrian_start_mode": str(cfg.get("pedestrian_start_mode", "exact")),
        "pedestrian_destination_mode": str(cfg.get("pedestrian_destination_mode", "exact")),
        "pedestrian_navmesh_samples": int(cfg.get("pedestrian_navmesh_samples", 200)),
        "snap_warn_distance_m": float(cfg.get("snap_warn_distance_m", 0.5)),
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


def _distance_xy(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _format_location(location: Optional[carla.Location]) -> str:
    if location is None:
        return "None"
    return f"({float(location.x):.2f}, {float(location.y):.2f}, {float(location.z):.2f})"


def _format_transform(transform: Optional[carla.Transform]) -> str:
    if transform is None:
        return "None"
    return (
        f"({_format_location(transform.location)}, "
        f"yaw={float(transform.rotation.yaw):.1f})"
    )


def _nearest_spawn_point(carla_map, location: carla.Location) -> Optional[carla.Transform]:
    spawn_points = list(carla_map.get_spawn_points())
    if not spawn_points:
        return None
    return min(spawn_points, key=lambda transform: _distance_xy(transform.location, location))


def _transform_with_yaw(transform: carla.Transform, yaw: float) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=float(transform.location.x), y=float(transform.location.y), z=float(transform.location.z)),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def _transform_with_z_offset(transform: carla.Transform, z_offset: float) -> carla.Transform:
    if abs(float(z_offset)) <= 1e-9:
        return transform
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z) + float(z_offset),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def _nearest_navigation_location(carla_world, location: carla.Location, samples: int) -> Optional[carla.Location]:
    best = None
    best_dist = math.inf
    for _ in range(max(int(samples), 1)):
        candidate = carla_world.get_random_location_from_navigation()
        if candidate is None:
            continue
        dist = _distance_xy(location, candidate)
        if dist < best_dist:
            best = candidate
            best_dist = dist
    return best


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
        candidate_specs = [spec for spec in self._specs["vehicles"] if spec["is_candidate"]]
        background_specs = [spec for spec in self._specs["vehicles"] if not spec["is_candidate"]]

        # Spawn camera-equipped cooperative candidates before random background traffic so
        # count-only vehicles cannot occupy the scripted candidate positions first.
        for spec in candidate_specs:
            start = _to_transform(spec["start"])
            destination = _to_location(spec["destination"])
            start = self._resolve_vehicle_start(start, spec)
            destination = self._resolve_vehicle_destination(destination)
            # One placed vehicle; stationary if no destination, else TM-routed.
            actor = self._world.spawn_scenario_vehicle(
                start=start,
                destination=destination,
                target_speed=spec["target_speed"],
                ignore_lights=spec["ignore_lights"],
                stationary=spec["stationary"],
            )
            if actor is None:
                SCENARIO_LOGGER.warning(
                    "Scenario candidate vehicle spawn failed start=%s destination=%s "
                    "stationary=%s target_speed=%.1f ignore_lights=%s",
                    _format_transform(start),
                    _format_location(destination),
                    bool(spec["stationary"]),
                    float(spec["target_speed"]),
                    bool(spec["ignore_lights"]),
                )
                continue
            spawned_vehicles += 1
            if self._cooperative_hook is not None:
                self._cooperative_hook(actor)
                spawned_candidates += 1

        for spec in background_specs:
            destination = self._resolve_vehicle_destination(_to_location(spec["destination"]))
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
                "start": self._resolve_pedestrian_start(_to_transform(spec["start"])),
                "destination": self._resolve_pedestrian_destination(_to_location(spec["destination"])),
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

    def _resolve_vehicle_start(self, transform: Optional[carla.Transform], spec: Optional[Dict] = None) -> Optional[carla.Transform]:
        if transform is None:
            return None
        mode = str((spec or {}).get("vehicle_start_mode") or self._specs.get("vehicle_start_mode", "exact")).lower()
        z_offset = (spec or {}).get("vehicle_spawn_z_offset")
        if z_offset is None:
            z_offset = self._specs.get("vehicle_spawn_z_offset", 0.0)
        if mode == "exact":
            return _transform_with_z_offset(transform, float(z_offset))
        requested = transform.location
        resolved = None
        if mode == "road":
            waypoint = self._world._map.get_waypoint(
                requested,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            resolved = waypoint.transform if waypoint is not None else None
        elif mode in ("spawn", "spawn_point", "nearest_spawn_point"):
            resolved = _nearest_spawn_point(self._world._map, requested)
        else:
            SCENARIO_LOGGER.warning("Unknown vehicle_start_mode=%s; using exact scenario start.", mode)
            return transform
        if resolved is None:
            SCENARIO_LOGGER.warning(
                "Failed to resolve scenario vehicle start mode=%s at %s; using exact.",
                mode,
                requested,
            )
            return transform
        preserve_yaw = (spec or {}).get("preserve_vehicle_start_yaw")
        if preserve_yaw is None:
            preserve_yaw = self._specs.get("preserve_vehicle_start_yaw", True)
        if bool(preserve_yaw):
            resolved = _transform_with_yaw(resolved, transform.rotation.yaw)
        resolved = _transform_with_z_offset(resolved, float(z_offset))
        self._log_snap("vehicle_start", mode, requested, resolved.location)
        return resolved

    def _resolve_vehicle_destination(self, location: Optional[carla.Location]) -> Optional[carla.Location]:
        if location is None:
            return None
        mode = str(self._specs.get("vehicle_destination_mode", "exact")).lower()
        if mode == "exact":
            return location
        resolved = None
        if mode == "road":
            waypoint = self._world._map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            resolved = waypoint.transform.location if waypoint is not None else None
        elif mode in ("spawn", "spawn_point", "nearest_spawn_point"):
            spawn_point = _nearest_spawn_point(self._world._map, location)
            resolved = spawn_point.location if spawn_point is not None else None
        else:
            SCENARIO_LOGGER.warning("Unknown vehicle_destination_mode=%s; using exact destination.", mode)
            return location
        if resolved is None:
            SCENARIO_LOGGER.warning(
                "Failed to resolve scenario vehicle destination mode=%s at %s; using exact.",
                mode,
                location,
            )
            return location
        self._log_snap("vehicle_destination", mode, location, resolved)
        return resolved

    def _resolve_pedestrian_start(self, transform: Optional[carla.Transform]) -> Optional[carla.Transform]:
        if transform is None:
            return None
        mode = str(self._specs.get("pedestrian_start_mode", "exact")).lower()
        if mode == "exact":
            return transform
        if mode not in ("navmesh", "navigation"):
            SCENARIO_LOGGER.warning("Unknown pedestrian_start_mode=%s; using exact scenario start.", mode)
            return transform
        requested = transform.location
        resolved = _nearest_navigation_location(
            self._world._world,
            requested,
            int(self._specs.get("pedestrian_navmesh_samples", 200)),
        )
        if resolved is None:
            SCENARIO_LOGGER.warning("Failed to resolve pedestrian start to navmesh at %s; using exact.", requested)
            return transform
        self._log_snap("pedestrian_start", mode, requested, resolved)
        return carla.Transform(resolved, transform.rotation)

    def _resolve_pedestrian_destination(self, location: Optional[carla.Location]) -> Optional[carla.Location]:
        if location is None:
            return None
        mode = str(self._specs.get("pedestrian_destination_mode", "exact")).lower()
        if mode == "exact":
            return location
        if mode not in ("navmesh", "navigation"):
            SCENARIO_LOGGER.warning("Unknown pedestrian_destination_mode=%s; using exact destination.", mode)
            return location
        resolved = _nearest_navigation_location(
            self._world._world,
            location,
            int(self._specs.get("pedestrian_navmesh_samples", 200)),
        )
        if resolved is None:
            SCENARIO_LOGGER.warning("Failed to resolve pedestrian destination to navmesh at %s; using exact.", location)
            return location
        self._log_snap("pedestrian_destination", mode, location, resolved)
        return resolved

    def _log_snap(self, label: str, mode: str, requested: carla.Location, resolved: carla.Location) -> None:
        dist = _distance_xy(requested, resolved)
        warn_distance = float(self._specs.get("snap_warn_distance_m", 0.5))
        log = SCENARIO_LOGGER.warning if dist > warn_distance else SCENARIO_LOGGER.info
        log(
            "Resolved scenario %s mode=%s requested=(%.2f, %.2f, %.2f) resolved=(%.2f, %.2f, %.2f) delta_xy=%.2fm",
            label,
            mode,
            float(requested.x),
            float(requested.y),
            float(requested.z),
            float(resolved.x),
            float(resolved.y),
            float(resolved.z),
            dist,
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
