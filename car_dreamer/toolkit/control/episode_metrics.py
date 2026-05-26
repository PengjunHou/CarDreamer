from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class EpisodeMetrics:
    """Per-episode metric collector for Route B evaluation.

    Usage:
        metrics.reset()                                  # at episode start
        metrics.record(t, ego, control, c_out, npcs, info)  # each step after control
        metrics.on_emergency_stop()                      # via monkey-patched BasicAgent
        metrics.add_bandwidth(bytes_in_round)            # each comm round
        result = metrics.finalize()                      # at episode end
    """

    def __init__(
        self,
        dt: float,
        t_max_steps: int = 800,
        npc_conflict_max_dist_m: float = 30.0,
        npc_conflict_fov_deg: float = 180.0,
        ego_radius_m: float = 1.0,
        npc_radius_m: float = 1.0,
        ttc_eps: float = 0.1,
    ) -> None:
        self.dt = float(dt)
        self.t_max_steps = int(t_max_steps)
        self.npc_conflict_max_dist_m = float(npc_conflict_max_dist_m)
        self.npc_conflict_fov_deg = float(npc_conflict_fov_deg)
        self.ego_radius_m = float(ego_radius_m)
        self.npc_radius_m = float(npc_radius_m)
        self.ttc_eps = float(ttc_eps)

        self.reset()

    def reset(self) -> None:
        self._speeds: List[float] = []
        self._confidences: List[float] = []
        self._brakes: List[float] = []
        self._throttles: List[float] = []
        self._ttc_min: float = float("inf")
        self._estop_count: int = 0
        self._bandwidth_bytes_total: float = 0.0
        self._bandwidth_round_count: int = 0
        self._t_start: Optional[int] = None
        self._t_done: Optional[int] = None
        self._collided: bool = False
        self._last_step: int = -1

    def add_bandwidth(self, bytes_in_round: float) -> None:
        self._bandwidth_bytes_total += float(bytes_in_round)
        self._bandwidth_round_count += 1

    def on_emergency_stop(self) -> None:
        self._estop_count += 1

    def record(
        self,
        t: int,
        ego: Any,
        control: Any,
        confidence: float,
        npcs: List[Any],
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        info = info or {}
        if self._t_start is None:
            self._t_start = int(t)
        self._last_step = int(t)

        try:
            vel = ego.get_velocity()
            speed = math.hypot(float(vel.x), float(vel.y))
        except Exception:
            speed = 0.0
        self._speeds.append(speed)
        self._confidences.append(float(confidence))

        try:
            self._brakes.append(float(getattr(control, "brake", 0.0)))
            self._throttles.append(float(getattr(control, "throttle", 0.0)))
        except (TypeError, ValueError):
            self._brakes.append(0.0)
            self._throttles.append(0.0)

        try:
            r_dest = float(info.get("r_destination", 0.0))
        except (TypeError, ValueError):
            r_dest = 0.0
        if r_dest > 0.0 and self._t_done is None:
            self._t_done = int(t)

        try:
            r_coll = float(info.get("r_collision", 0.0))
        except (TypeError, ValueError):
            r_coll = 0.0
        if r_coll < 0.0:
            self._collided = True

        ttc_t = self._compute_min_ttc(ego, npcs)
        if ttc_t is not None and ttc_t < self._ttc_min:
            self._ttc_min = float(ttc_t)

    def mark_completed(self, t: Optional[int] = None) -> None:
        if self._t_done is None:
            self._t_done = int(t) if t is not None else self._last_step

    def mark_collided(self) -> None:
        self._collided = True

    def _compute_min_ttc(self, ego: Any, npcs: List[Any]) -> Optional[float]:
        if not npcs:
            return None
        try:
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_yaw = math.radians(float(ego_tf.rotation.yaw))
            ego_vel = ego.get_velocity()
        except Exception:
            return None

        max_dist = self.npc_conflict_max_dist_m
        half_fov = self.npc_conflict_fov_deg / 2.0
        min_ttc: Optional[float] = None

        for v in npcs:
            try:
                v_loc = v.get_location()
                v_vel = v.get_velocity()
            except Exception:
                continue
            dx = float(v_loc.x) - float(ego_loc.x)
            dy = float(v_loc.y) - float(ego_loc.y)
            dist = math.hypot(dx, dy)
            if dist < 1e-6 or dist > max_dist:
                continue
            bearing = math.atan2(dy, dx) - ego_yaw
            bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
            if abs(math.degrees(bearing)) > half_fov:
                continue
            d_eff = max(dist - self.ego_radius_m - self.npc_radius_m, 1e-3)
            n_x = dx / dist
            n_y = dy / dist
            v_rel = (
                (float(ego_vel.x) - float(v_vel.x)) * n_x
                + (float(ego_vel.y) - float(v_vel.y)) * n_y
            )
            v_rel = max(v_rel, self.ttc_eps)
            ttc = d_eff / v_rel
            if min_ttc is None or ttc < min_ttc:
                min_ttc = ttc
        return min_ttc

    def finalize(self) -> Dict[str, float]:
        T = len(self._speeds)
        if T == 0:
            return self._empty_result()

        dt = self.dt
        speeds = self._speeds

        mean_speed = sum(speeds) / T
        var_speed = sum((s - mean_speed) ** 2 for s in speeds) / T
        std_speed = math.sqrt(var_speed) if var_speed > 0.0 else 0.0

        accs: List[float] = []
        for i in range(1, T):
            accs.append((speeds[i] - speeds[i - 1]) / dt)
        n_acc = len(accs)
        mean_abs_acc = sum(abs(a) for a in accs) / n_acc if n_acc > 0 else 0.0
        max_abs_acc = max((abs(a) for a in accs), default=0.0)

        jerks: List[float] = []
        for i in range(1, n_acc):
            jerks.append((accs[i] - accs[i - 1]) / dt)
        n_jerk = len(jerks)
        mean_abs_jerk = sum(abs(j) for j in jerks) / n_jerk if n_jerk > 0 else 0.0
        max_abs_jerk = max((abs(j) for j in jerks), default=0.0)

        if self._t_done is not None and self._t_start is not None:
            completion_steps = max(self._t_done - self._t_start, 0)
            completion_time = completion_steps * dt
            completed = True
        else:
            completion_steps = T
            completion_time = T * dt
            completed = False

        if self._bandwidth_round_count > 0:
            avg_bytes_per_round = (
                self._bandwidth_bytes_total / self._bandwidth_round_count
            )
        else:
            avg_bytes_per_round = 0.0
        avg_bytes_per_env_step = self._bandwidth_bytes_total / T if T > 0 else 0.0
        avg_bytes_per_sec = avg_bytes_per_env_step / dt if dt > 0.0 else 0.0

        mean_confidence = (
            sum(self._confidences) / len(self._confidences)
            if self._confidences
            else 0.0
        )

        min_ttc = float(self._ttc_min) if math.isfinite(self._ttc_min) else -1.0

        return {
            "mean_speed_mps": float(mean_speed),
            "mean_speed_kmh": float(mean_speed * 3.6),
            "speed_std_mps": float(std_speed),
            "completion_time_s": float(completion_time),
            "completion_steps": float(completion_steps),
            "completed": 1.0 if completed else 0.0,
            "collided": 1.0 if self._collided else 0.0,
            "mean_abs_acc_mps2": float(mean_abs_acc),
            "max_abs_acc_mps2": float(max_abs_acc),
            "mean_abs_jerk_mps3": float(mean_abs_jerk),
            "max_abs_jerk_mps3": float(max_abs_jerk),
            "emergency_stops": float(self._estop_count),
            "min_ttc_s": float(min_ttc),
            "avg_bandwidth_bytes_per_comm_round": float(avg_bytes_per_round),
            "avg_bandwidth_bytes_per_env_step": float(avg_bytes_per_env_step),
            "avg_bandwidth_bytes_per_sec": float(avg_bytes_per_sec),
            "total_bandwidth_bytes": float(self._bandwidth_bytes_total),
            "mean_confidence": float(mean_confidence),
            "num_steps": float(T),
        }

    def _empty_result(self) -> Dict[str, float]:
        return {
            "mean_speed_mps": 0.0,
            "mean_speed_kmh": 0.0,
            "speed_std_mps": 0.0,
            "completion_time_s": 0.0,
            "completion_steps": 0.0,
            "completed": 0.0,
            "collided": 0.0,
            "mean_abs_acc_mps2": 0.0,
            "max_abs_acc_mps2": 0.0,
            "mean_abs_jerk_mps3": 0.0,
            "max_abs_jerk_mps3": 0.0,
            "emergency_stops": 0.0,
            "min_ttc_s": -1.0,
            "avg_bandwidth_bytes_per_comm_round": 0.0,
            "avg_bandwidth_bytes_per_env_step": 0.0,
            "avg_bandwidth_bytes_per_sec": 0.0,
            "total_bandwidth_bytes": 0.0,
            "mean_confidence": 0.0,
            "num_steps": 0.0,
        }
