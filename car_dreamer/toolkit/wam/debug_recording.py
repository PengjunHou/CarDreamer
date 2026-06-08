from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .runtime import MotionPredictionRecord, NotableObjectRecord, ObjectState


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class ActorSnapshot:
    actor_id: int
    actor_type: str
    object_class: str
    position: Point3D
    velocity: Point3D
    yaw: float
    bbox: Tuple[Point2D, ...] = ()


@dataclass(frozen=True)
class PendingWAMStep:
    step: int
    time_s: float
    ego: ActorSnapshot
    wam: Mapping[str, object]
    notable_objects: Tuple[Mapping[str, object], ...]


def future_sample_offsets(horizon_s: float, sample_count: int) -> Tuple[float, ...]:
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    step_s = float(horizon_s) / float(sample_count)
    return tuple(round(step_s * idx, 6) for idx in range(1, int(sample_count) + 1))


def future_sample_step_offsets(fixed_dt: float, horizon_s: float, sample_count: int) -> Tuple[int, ...]:
    if fixed_dt <= 0:
        raise ValueError("fixed_dt must be positive")
    offsets = []
    previous = 0
    for offset_s in future_sample_offsets(horizon_s, sample_count):
        step_offset = max(1, int(round(offset_s / float(fixed_dt))))
        if step_offset <= previous:
            step_offset = previous + 1
        offsets.append(step_offset)
        previous = step_offset
    return tuple(offsets)


def predicted_future_waypoints(
    obj: ObjectState,
    *,
    future_offsets_s: Sequence[float],
    uncertainty: float,
) -> List[Dict[str, object]]:
    waypoints = []
    for dt_s in future_offsets_s:
        t = float(dt_s)
        waypoints.append(
            {
                "dt": float(t),
                "position": [
                    float(obj.x + obj.vx * t),
                    float(obj.y + obj.vy * t),
                    float(obj.z),
                ],
                "uncertainty": float(uncertainty),
            }
        )
    return waypoints


def actor_class_from_type(type_id: str) -> str:
    if type_id.startswith("walker.") or "pedestrian" in type_id:
        return "pedestrian"
    if type_id.startswith("vehicle."):
        return "vehicle"
    return "actor"


def carla_actor_bbox_footprint(actor) -> Tuple[Point2D, ...]:
    transform = actor.get_transform()
    bbox = actor.bounding_box
    yaw = math.radians(float(transform.rotation.yaw))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    origin_x = float(transform.location.x)
    origin_y = float(transform.location.y)
    center_x = float(getattr(bbox.location, "x", 0.0))
    center_y = float(getattr(bbox.location, "y", 0.0))
    extent_x = float(bbox.extent.x)
    extent_y = float(bbox.extent.y)

    corners = []
    for local_x, local_y in (
        (center_x - extent_x, center_y - extent_y),
        (center_x + extent_x, center_y - extent_y),
        (center_x + extent_x, center_y + extent_y),
        (center_x - extent_x, center_y + extent_y),
    ):
        world_x = origin_x + local_x * cos_yaw - local_y * sin_yaw
        world_y = origin_y + local_x * sin_yaw + local_y * cos_yaw
        corners.append((float(world_x), float(world_y)))
    return tuple(corners)


def snapshot_from_carla_actor(actor) -> ActorSnapshot:
    transform = actor.get_transform()
    location = transform.location
    velocity = actor.get_velocity()
    bbox = ()
    try:
        bbox = carla_actor_bbox_footprint(actor)
    except Exception:
        bbox = ()
    type_id = str(getattr(actor, "type_id", "actor"))
    return ActorSnapshot(
        actor_id=int(actor.id),
        actor_type=type_id,
        object_class=actor_class_from_type(type_id),
        position=(float(location.x), float(location.y), float(location.z)),
        velocity=(float(velocity.x), float(velocity.y), float(velocity.z)),
        yaw=float(transform.rotation.yaw),
        bbox=bbox,
    )


def snapshot_from_object_state(obj: ObjectState) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id=int(obj.actor_id),
        actor_type=str(obj.actor_type),
        object_class=str(obj.object_class),
        position=(float(obj.x), float(obj.y), float(obj.z)),
        velocity=(float(obj.vx), float(obj.vy), 0.0),
        yaw=float(obj.yaw),
        bbox=tuple(obj.bbox),
    )


def snapshot_to_json(snapshot: ActorSnapshot) -> Dict[str, object]:
    return {
        "id": int(snapshot.actor_id),
        "type": str(snapshot.object_class),
        "actor_type": str(snapshot.actor_type),
        "position": [float(v) for v in snapshot.position],
        "velocity": [float(v) for v in snapshot.velocity],
        "yaw": float(snapshot.yaw),
        "bbox": [[float(x), float(y)] for x, y in snapshot.bbox],
    }


def _prediction_uncertainty(
    actor_id: int,
    predictions: Mapping[int, MotionPredictionRecord],
    default: float,
) -> float:
    pred = predictions.get(int(actor_id))
    if pred is None:
        return float(default)
    return float(pred.uncertainty_score)


def _notable_to_pending_json(
    record: NotableObjectRecord,
    *,
    predictions: Mapping[int, MotionPredictionRecord],
    future_offsets_s: Sequence[float],
) -> Dict[str, object]:
    obj = record.object_state
    uncertainty = _prediction_uncertainty(int(obj.actor_id), predictions, math.nan)
    snapshot = snapshot_from_object_state(obj)
    return {
        "id": int(obj.actor_id),
        "type": str(obj.object_class),
        "actor_type": str(obj.actor_type),
        "position": [float(obj.x), float(obj.y), float(obj.z)],
        "velocity": [float(obj.vx), float(obj.vy), 0.0],
        "yaw": float(obj.yaw),
        "bbox": [[float(x), float(y)] for x, y in snapshot.bbox],
        "visible_to_ego": bool(record.visible),
        "invisible_to_ego": bool(record.invisible),
        "occluding": bool(record.occluding),
        "route_distance": float(record.route_distance),
        "visible_to_collaborators": [int(v) for v in obj.visible_to_collaborators],
        "uncertainty_score": float(uncertainty),
        "predicted": {
            "future_waypoints": predicted_future_waypoints(
                obj,
                future_offsets_s=future_offsets_s,
                uncertainty=uncertainty,
            )
        },
    }


class WAMNotableDebugRecorder:
    def __init__(
        self,
        *,
        fixed_dt: float,
        horizon_s: float = 3.0,
        future_samples: int = 6,
    ) -> None:
        self.fixed_dt = float(fixed_dt)
        self.horizon_s = float(horizon_s)
        self.future_samples = int(future_samples)
        self.future_offsets_s = future_sample_offsets(self.horizon_s, self.future_samples)
        self.future_step_offsets = future_sample_step_offsets(
            self.fixed_dt,
            self.horizon_s,
            self.future_samples,
        )
        self.horizon_steps = max(self.future_step_offsets)
        self._history: Dict[int, Dict[int, ActorSnapshot]] = {}
        self._pending: Deque[PendingWAMStep] = deque()

    def observe(
        self,
        *,
        step: int,
        time_s: float,
        ego: ActorSnapshot,
        actors: Iterable[ActorSnapshot],
        notable_records: Sequence[NotableObjectRecord],
        predictions: Mapping[int, MotionPredictionRecord],
        wam: Mapping[str, object],
        include_record: bool,
    ) -> List[Dict[str, object]]:
        step = int(step)
        actor_map = {int(actor.actor_id): actor for actor in actors}
        actor_map[int(ego.actor_id)] = ego
        self._history[step] = actor_map

        if include_record:
            notable_objects = tuple(
                _notable_to_pending_json(
                    record,
                    predictions=predictions,
                    future_offsets_s=self.future_offsets_s,
                )
                for record in notable_records
            )
            self._pending.append(
                PendingWAMStep(
                    step=step,
                    time_s=float(time_s),
                    ego=ego,
                    wam=dict(wam),
                    notable_objects=notable_objects,
                )
            )

        ready = self.flush_ready(current_step=step)
        self._drop_old_history()
        return ready

    def flush_ready(self, *, current_step: int) -> List[Dict[str, object]]:
        ready = []
        while self._pending and int(current_step) - self._pending[0].step >= self.horizon_steps:
            ready.append(self._build_completed_record(self._pending.popleft()))
        return ready

    def flush_all(self) -> List[Dict[str, object]]:
        records = []
        while self._pending:
            records.append(self._build_completed_record(self._pending.popleft()))
        self._drop_old_history()
        return records

    def _build_completed_record(self, pending: PendingWAMStep) -> Dict[str, object]:
        notable_objects = []
        for item in pending.notable_objects:
            completed = dict(item)
            actor_id = int(completed["id"])
            completed["ground_truth"] = {
                "future_waypoints": self._ground_truth_waypoints(
                    base_step=pending.step,
                    actor_id=actor_id,
                )
            }
            notable_objects.append(completed)

        return {
            "step": int(pending.step),
            "time_s": float(pending.time_s),
            "ego": snapshot_to_json(pending.ego),
            "wam": dict(pending.wam),
            "notable_objects": notable_objects,
        }

    def _ground_truth_waypoints(self, *, base_step: int, actor_id: int) -> List[Dict[str, object]]:
        waypoints = []
        for dt_s, step_offset in zip(self.future_offsets_s, self.future_step_offsets):
            snapshot = self._history.get(int(base_step) + int(step_offset), {}).get(int(actor_id))
            if snapshot is None:
                waypoints.append(
                    {
                        "dt": float(dt_s),
                        "position": None,
                        "available": False,
                    }
                )
            else:
                waypoints.append(
                    {
                        "dt": float(dt_s),
                        "position": [float(v) for v in snapshot.position],
                        "available": True,
                    }
                )
        return waypoints

    def _drop_old_history(self) -> None:
        if self._pending:
            min_needed = self._pending[0].step
        else:
            if not self._history:
                return
            min_needed = max(self._history)
        for step in list(self._history):
            if step < min_needed:
                del self._history[step]
