from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class ObjectState:
    actor_id: int
    actor_type: str
    object_class: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    yaw: float
    length: float
    width: float
    height: float
    bbox: Tuple[Point2D, ...] = ()
    distance_to_route: float = math.inf
    visible_to_ego: bool = False
    visible_to_collaborators: Tuple[int, ...] = ()


@dataclass(frozen=True)
class NotableObjectRecord:
    object_state: ObjectState
    notable: bool
    visible: bool
    invisible: bool
    occluding: bool
    route_distance: float


@dataclass(frozen=True)
class MotionPredictionRecord:
    actor_id: int
    future_xy: Tuple[Point2D, ...]
    covariance_diag: Tuple[Point2D, ...]
    uncertainty_score: float


@dataclass(frozen=True)
class CoopRequest:
    ego_id: int
    step: int
    high_uncertainty_object_ids: Tuple[int, ...]
    uncertainty_threshold: float
    reason: str


@dataclass(frozen=True)
class WAMPolicy:
    selected_vehicle_ids: Tuple[int, ...]
    modality_by_vehicle: Dict[int, str]
    bandwidth_by_vehicle: Dict[int, float]
    frequency_steps: int
    reason: str


def _as_xy(point: Sequence[float]) -> Point2D:
    return float(point[0]), float(point[1])


def _point_segment_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    qx = sx + t * dx
    qy = sy + t * dy
    return math.hypot(px - qx, py - qy)


def distance_to_route(point: Point2D, route_points: Sequence[Sequence[float]]) -> float:
    points = [_as_xy(item) for item in route_points]
    if not points:
        return math.inf
    if len(points) == 1:
        return math.hypot(point[0] - points[0][0], point[1] - points[0][1])
    return min(_point_segment_distance(point, points[i], points[i + 1]) for i in range(len(points) - 1))


def select_notable_objects(
    objects: Iterable[ObjectState],
    route_points: Sequence[Sequence[float]],
    *,
    notable_distance_m: float,
    max_notable_objects: int,
) -> List[NotableObjectRecord]:
    records: List[NotableObjectRecord] = []
    for obj in objects:
        route_distance = distance_to_route((obj.x, obj.y), route_points)
        if route_distance >= float(notable_distance_m):
            continue
        visible = bool(obj.visible_to_ego)
        invisible = (not visible) and bool(obj.visible_to_collaborators)
        records.append(
            NotableObjectRecord(
                object_state=obj,
                notable=True,
                visible=visible,
                invisible=invisible,
                occluding=False,
                route_distance=float(route_distance),
            )
        )
    records.sort(key=lambda item: (item.route_distance, item.object_state.actor_id))
    if max_notable_objects <= 0:
        return []
    return records[: int(max_notable_objects)]


def predict_notable_motion(
    notable_objects: Sequence[NotableObjectRecord],
    *,
    dt: float,
    horizon_steps: int,
    visible_uncertainty: float,
    invisible_uncertainty: float,
) -> Dict[int, MotionPredictionRecord]:
    predictions: Dict[int, MotionPredictionRecord] = {}
    steps = max(int(horizon_steps), 1)
    for record in notable_objects:
        obj = record.object_state
        uncertainty = float(visible_uncertainty if record.visible else invisible_uncertainty)
        future_xy = []
        covariance = []
        for idx in range(1, steps + 1):
            t = float(idx) * float(dt)
            future_xy.append((float(obj.x + obj.vx * t), float(obj.y + obj.vy * t)))
            covariance.append((uncertainty, uncertainty))
        predictions[int(obj.actor_id)] = MotionPredictionRecord(
            actor_id=int(obj.actor_id),
            future_xy=tuple(future_xy),
            covariance_diag=tuple(covariance),
            uncertainty_score=uncertainty,
        )
    return predictions


def build_coop_request(
    *,
    ego_id: int,
    step: int,
    predictions: Mapping[int, MotionPredictionRecord],
    uncertainty_threshold: float,
) -> Optional[CoopRequest]:
    high_ids = tuple(
        sorted(
            int(actor_id)
            for actor_id, pred in predictions.items()
            if float(pred.uncertainty_score) > float(uncertainty_threshold)
        )
    )
    if not high_ids:
        return None
    return CoopRequest(
        ego_id=int(ego_id),
        step=int(step),
        high_uncertainty_object_ids=high_ids,
        uncertainty_threshold=float(uncertainty_threshold),
        reason="notable_object_uncertainty_above_threshold",
    )


def build_placeholder_policy(
    *,
    request: Optional[CoopRequest],
    candidate_vehicle_ids: Iterable[int],
    uplink_bps: float,
    frequency_steps: int,
    default_modality: str,
) -> WAMPolicy:
    if request is None:
        return WAMPolicy(
            selected_vehicle_ids=(),
            modality_by_vehicle={},
            bandwidth_by_vehicle={},
            frequency_steps=int(frequency_steps),
            reason="no_coop_request",
        )
    selected = tuple(sorted(int(vehicle_id) for vehicle_id in candidate_vehicle_ids))
    if not selected:
        return WAMPolicy(
            selected_vehicle_ids=(),
            modality_by_vehicle={},
            bandwidth_by_vehicle={},
            frequency_steps=int(frequency_steps),
            reason="request_triggered_but_no_candidate_collaborators",
        )
    share = float(uplink_bps) / float(len(selected))
    return WAMPolicy(
        selected_vehicle_ids=selected,
        modality_by_vehicle={vehicle_id: str(default_modality) for vehicle_id in selected},
        bandwidth_by_vehicle={vehicle_id: share for vehicle_id in selected},
        frequency_steps=int(frequency_steps),
        reason="placeholder_all_candidates",
    )
