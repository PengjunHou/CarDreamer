from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

try:
    import carla
except ModuleNotFoundError:  # pragma: no cover - allows CARLA-free geometry tests.
    carla = None


SAFETY_LOGGER = get_runtime_logger("car_dreamer.pedestrian_safety")


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _xy_from_vector(value: Any) -> Tuple[float, float]:
    return float(getattr(value, "x", 0.0)), float(getattr(value, "y", 0.0))


@dataclass(frozen=True)
class PedestrianSafetyConfig:
    enabled: bool = False
    max_distance_m: float = 12.0
    front_angle_deg: float = 50.0
    ttc_threshold_s: float = 2.0
    brake_distance_m: float = 5.0
    restore_autopilot: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> "PedestrianSafetyConfig":
        return cls(
            enabled=bool(_cfg_get(cfg, "enabled", cls.enabled)),
            max_distance_m=float(_cfg_get(cfg, "max_distance_m", cls.max_distance_m)),
            front_angle_deg=float(_cfg_get(cfg, "front_angle_deg", cls.front_angle_deg)),
            ttc_threshold_s=float(_cfg_get(cfg, "ttc_threshold_s", cls.ttc_threshold_s)),
            brake_distance_m=float(_cfg_get(cfg, "brake_distance_m", cls.brake_distance_m)),
            restore_autopilot=bool(_cfg_get(cfg, "restore_autopilot", cls.restore_autopilot)),
        )


@dataclass(frozen=True)
class PedestrianHazard:
    vehicle_id: int
    walker_id: int
    distance_m: float
    angle_deg: float
    ttc_s: float


def evaluate_pedestrian_hazard(
    vehicle_location: Any,
    vehicle_forward: Any,
    vehicle_velocity: Any,
    walker_location: Any,
    walker_velocity: Any,
    *,
    max_distance_m: float,
    front_angle_deg: float,
    ttc_threshold_s: float,
    brake_distance_m: float,
) -> Optional[Tuple[float, float, float]]:
    """Return (distance, angle, ttc) if a walker is a frontal hazard."""
    vx, vy = _xy_from_vector(vehicle_location)
    wx, wy = _xy_from_vector(walker_location)
    rel_x = wx - vx
    rel_y = wy - vy
    distance = math.hypot(rel_x, rel_y)
    if distance <= 1e-6 or distance > max_distance_m:
        return None

    forward_x, forward_y = _xy_from_vector(vehicle_forward)
    forward_norm = math.hypot(forward_x, forward_y)
    if forward_norm <= 1e-6:
        return None
    rel_unit_x = rel_x / distance
    rel_unit_y = rel_y / distance
    cos_angle = max(-1.0, min(1.0, (forward_x * rel_unit_x + forward_y * rel_unit_y) / forward_norm))
    angle_deg = math.degrees(math.acos(cos_angle))
    if angle_deg > front_angle_deg:
        return None

    vehicle_vel_x, vehicle_vel_y = _xy_from_vector(vehicle_velocity)
    walker_vel_x, walker_vel_y = _xy_from_vector(walker_velocity)
    closing_speed = (vehicle_vel_x - walker_vel_x) * rel_unit_x + (vehicle_vel_y - walker_vel_y) * rel_unit_y
    ttc_s = math.inf
    if closing_speed > 1e-3:
        ttc_s = distance / closing_speed

    if distance <= brake_distance_m or ttc_s <= ttc_threshold_s:
        return distance, angle_deg, ttc_s
    return None


class PedestrianSafetySupervisor:
    """Emergency-brake safety layer for pedestrians in front of active vehicles."""

    def __init__(self, world_manager: Any, cfg: Any = None):
        self._world = world_manager
        self._config = PedestrianSafetyConfig.from_config(cfg)
        self._active_hazards: Dict[int, PedestrianHazard] = {}
        self._disabled_autopilot_vehicle_ids = set()
        if self._config.enabled:
            SAFETY_LOGGER.info(
                "Pedestrian safety supervisor enabled max_distance=%.2f front_angle=%.1f "
                "ttc_threshold=%.2f brake_distance=%.2f restore_autopilot=%s",
                self._config.max_distance_m,
                self._config.front_angle_deg,
                self._config.ttc_threshold_s,
                self._config.brake_distance_m,
                self._config.restore_autopilot,
            )

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def reset(self) -> None:
        self._active_hazards.clear()
        self._disabled_autopilot_vehicle_ids.clear()

    def step(self, step_index: int) -> None:
        if not self._config.enabled:
            return

        vehicles = self._safe_get_actors("vehicle")
        walkers = self._safe_get_actors("walker.pedestrian")
        current_hazards: Dict[int, PedestrianHazard] = {}

        for vehicle in vehicles:
            if not self._is_alive(vehicle):
                continue
            hazard = self._nearest_hazard(vehicle, walkers)
            if hazard is None:
                continue
            current_hazards[hazard.vehicle_id] = hazard
            autopilot_disabled = self._disable_autopilot_if_needed(vehicle)
            self._apply_emergency_brake(vehicle)
            previous = self._active_hazards.get(hazard.vehicle_id)
            if previous is None or previous.walker_id != hazard.walker_id:
                SAFETY_LOGGER.info(
                    "Pedestrian hazard trigger step=%d vehicle_id=%s walker_id=%s "
                    "distance=%.2f ttc=%s autopilot_disabled=%s",
                    step_index,
                    hazard.vehicle_id,
                    hazard.walker_id,
                    hazard.distance_m,
                    self._format_ttc(hazard.ttc_s),
                    autopilot_disabled,
                )

        cleared_vehicle_ids = set(self._active_hazards) - set(current_hazards)
        for vehicle_id in sorted(cleared_vehicle_ids):
            restored = self._restore_autopilot_if_needed(vehicle_id)
            SAFETY_LOGGER.info(
                "Pedestrian hazard clear step=%d vehicle_id=%s restored_autopilot=%s",
                step_index,
                vehicle_id,
                restored,
            )

        self._active_hazards = current_hazards

        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(step_index, int(runtime_cfg["step_debug_interval"]), logger=SAFETY_LOGGER):
            SAFETY_LOGGER.debug(
                "Pedestrian safety step=%d vehicles=%d walkers=%d active_hazards=%d disabled_autopilot=%d",
                step_index,
                len(vehicles),
                len(walkers),
                len(current_hazards),
                len(self._disabled_autopilot_vehicle_ids),
            )

    def _safe_get_actors(self, actor_type: str):
        try:
            return self._world.carla_actors(actor_type)
        except Exception as exc:  # noqa: BLE001
            SAFETY_LOGGER.debug("Failed to retrieve CARLA actors type=%s: %s", actor_type, exc)
            return []

    def _nearest_hazard(self, vehicle: carla.Vehicle, walkers) -> Optional[PedestrianHazard]:
        try:
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_forward = vehicle_transform.get_forward_vector()
            vehicle_velocity = vehicle.get_velocity()
        except Exception as exc:  # noqa: BLE001
            SAFETY_LOGGER.debug("Failed to read vehicle state vehicle_id=%s: %s", getattr(vehicle, "id", None), exc)
            return None

        nearest: Optional[PedestrianHazard] = None
        for walker in walkers:
            if not self._is_alive(walker):
                continue
            try:
                result = evaluate_pedestrian_hazard(
                    vehicle_location,
                    vehicle_forward,
                    vehicle_velocity,
                    walker.get_location(),
                    walker.get_velocity(),
                    max_distance_m=self._config.max_distance_m,
                    front_angle_deg=self._config.front_angle_deg,
                    ttc_threshold_s=self._config.ttc_threshold_s,
                    brake_distance_m=self._config.brake_distance_m,
                )
            except Exception as exc:  # noqa: BLE001
                SAFETY_LOGGER.debug(
                    "Failed hazard check vehicle_id=%s walker_id=%s: %s",
                    vehicle.id,
                    getattr(walker, "id", None),
                    exc,
                )
                continue
            if result is None:
                continue
            distance_m, angle_deg, ttc_s = result
            hazard = PedestrianHazard(
                vehicle_id=int(vehicle.id),
                walker_id=int(walker.id),
                distance_m=distance_m,
                angle_deg=angle_deg,
                ttc_s=ttc_s,
            )
            if nearest is None or hazard.distance_m < nearest.distance_m:
                nearest = hazard
        return nearest

    def _disable_autopilot_if_needed(self, vehicle: carla.Vehicle) -> bool:
        vehicle_id = int(vehicle.id)
        try:
            is_autopilot = self._world.is_autopilot_actor(vehicle_id)
        except AttributeError:
            is_autopilot = False
        if not is_autopilot:
            return False
        if vehicle_id in self._disabled_autopilot_vehicle_ids:
            return False
        try:
            self._world.set_actor_autopilot(vehicle, False)
        except Exception as exc:  # noqa: BLE001
            SAFETY_LOGGER.warning("Failed to disable autopilot vehicle_id=%s: %s", vehicle_id, exc)
            return False
        self._disabled_autopilot_vehicle_ids.add(vehicle_id)
        return True

    def _restore_autopilot_if_needed(self, vehicle_id: int) -> bool:
        if not self._config.restore_autopilot or vehicle_id not in self._disabled_autopilot_vehicle_ids:
            return False
        actor = None
        try:
            actor = self._world.carla_world.get_actor(vehicle_id)
        except Exception as exc:  # noqa: BLE001
            SAFETY_LOGGER.debug("Failed to get actor for autopilot restore vehicle_id=%s: %s", vehicle_id, exc)
        restored = False
        if actor is not None and self._is_alive(actor):
            try:
                self._world.set_actor_autopilot(actor, True)
                restored = True
            except Exception as exc:  # noqa: BLE001
                SAFETY_LOGGER.warning("Failed to restore autopilot vehicle_id=%s: %s", vehicle_id, exc)
        self._disabled_autopilot_vehicle_ids.discard(vehicle_id)
        return restored

    def _apply_emergency_brake(self, vehicle: carla.Vehicle) -> None:
        if carla is None:
            return
        steer = 0.0
        try:
            steer = float(vehicle.get_control().steer)
        except Exception:  # noqa: BLE001
            pass
        try:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0))
        except Exception as exc:  # noqa: BLE001
            SAFETY_LOGGER.warning("Failed to apply pedestrian emergency brake vehicle_id=%s: %s", getattr(vehicle, "id", None), exc)

    @staticmethod
    def _is_alive(actor: Any) -> bool:
        return actor is not None and bool(getattr(actor, "is_alive", True))

    @staticmethod
    def _format_ttc(ttc_s: float) -> str:
        if math.isinf(ttc_s):
            return "inf"
        return f"{ttc_s:.2f}"
