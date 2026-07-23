"""Route-corridor coverage quality and missing-risk utilities (coverage uncertainty).

Ported verbatim from Coopeception-V4 car_dreamer/toolkit/wam/coverage.py, with the single
`from .bev import BevSpec` replaced by an inlined minimal BevSpec so the module is torch-free
(numpy only). Coverage uses only BevSpec.size and BevSpec.pixels_per_meter.

The raster geometry: ego is centered, heading-up, and the grid covers [-range_m, range_m].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BevSpec:
    """Minimal BEV raster geometry (inlined from wam.bev to stay torch-free).

    ``range_m`` is the half-extent (meters from ego to each edge).
    """

    size: int = 64
    range_m: float = 50.0

    @property
    def pixels_per_meter(self) -> float:
        return float(self.size) / (2.0 * float(self.range_m))


Point2D = Tuple[float, float]
EgoPose = Tuple[float, float, float]
ObserverPose = Tuple[int, float, float, float]

COVERAGE_CHANNEL_NAMES: Tuple[str, ...] = (
    "route_mask",
    "route_risk",
    "ego_coverage_quality",
    "collaborator_coverage_quality",
    "total_coverage_quality",
    "poor_coverage_risk",
)
COVERAGE_NUM_CHANNELS = len(COVERAGE_CHANNEL_NAMES)


@dataclass(frozen=True)
class CoverageConfig:
    past_route_distance_m: float = 10.0
    future_route_distance_m: float = 40.0
    corridor_width_m: float = 8.0
    coverage_distance_scale_m: float = 20.0
    route_risk_distance_scale_m: float = 20.0
    u_prior: float = 1.0
    freshness_gamma: float = 5.0  # decay rate for message-age coverage discount (exp(-gamma * age_s))


def _world_to_ego(x: float, y: float, ego_pose: EgoPose) -> Point2D:
    ex, ey, yaw_deg = ego_pose
    yaw = math.radians(float(yaw_deg))
    cos_a, sin_a = math.cos(-yaw), math.sin(-yaw)
    dx, dy = float(x) - float(ex), float(y) - float(ey)
    return cos_a * dx - sin_a * dy, sin_a * dx + cos_a * dy


def _ego_to_world(x: float, y: float, ego_pose: EgoPose) -> Point2D:
    ex, ey, yaw_deg = ego_pose
    yaw = math.radians(float(yaw_deg))
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    return ex + x * cos_a - y * sin_a, ey + x * sin_a + y * cos_a


def _cell_centers_world(ego_pose: EgoPose, spec: BevSpec) -> np.ndarray:
    """Vectorized ego-frame cell grid -> world coordinates (rotation + translation of a meshgrid)."""
    center = float(spec.size) / 2.0
    ppm = float(spec.pixels_per_meter)
    rows, cols = np.meshgrid(np.arange(spec.size), np.arange(spec.size), indexing="ij")
    ego_x = (center - (rows + 0.5)) / ppm
    ego_y = (center - (cols + 0.5)) / ppm
    ex, ey, yaw_deg = ego_pose
    yaw = math.radians(float(yaw_deg))
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    wx = float(ex) + ego_x * cos_a - ego_y * sin_a
    wy = float(ey) + ego_x * sin_a + ego_y * cos_a
    return np.stack((wx, wy), axis=-1).astype(np.float32)


def _cells_distance_to_polyline(centers: np.ndarray, polyline: Sequence[Point2D]) -> np.ndarray:
    """Vectorized per-cell min distance to a polyline. ``centers`` is ``[H,W,2]``; returns ``[H,W]``."""
    hw = centers.shape[:2]
    if not polyline:
        return np.full(hw, np.inf, dtype=np.float32)
    poly = np.asarray(polyline, dtype=np.float64)
    pts = centers.reshape(-1, 2).astype(np.float64)  # [N,2]
    if len(poly) == 1:
        d = np.hypot(pts[:, 0] - poly[0, 0], pts[:, 1] - poly[0, 1])
        return d.reshape(hw).astype(np.float32)
    a = poly[:-1]           # [S,2]
    ab = poly[1:] - a       # [S,2]
    denom = np.maximum((ab ** 2).sum(1), 1e-9)          # [S]
    ap = pts[:, None, :] - a[None, :, :]                 # [N,S,2]
    t = np.clip((ap * ab[None]).sum(2) / denom[None], 0.0, 1.0)  # [N,S]
    proj = a[None] + t[..., None] * ab[None]             # [N,S,2]
    d = np.sqrt(((pts[:, None, :] - proj) ** 2).sum(2))  # [N,S]
    return d.min(1).reshape(hw).astype(np.float32)


def _point_segment_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    qx, qy = sx + t * dx, sy + t * dy
    return math.hypot(px - qx, py - qy)


def _distance_to_polyline(point: Point2D, polyline: Sequence[Point2D]) -> float:
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return math.hypot(point[0] - polyline[0][0], point[1] - polyline[0][1])
    return min(_point_segment_distance(point, a, b) for a, b in zip(polyline[:-1], polyline[1:]))


def _clip_forward_route(ego_xy: Point2D, route_xy: Sequence[Point2D], distance_m: float) -> Sequence[Point2D]:
    out = [ego_xy]
    remaining = max(float(distance_m), 0.0)
    prev = ego_xy
    for raw in route_xy:
        cur = (float(raw[0]), float(raw[1]))
        seg = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if seg <= 1e-6:
            continue
        if seg <= remaining:
            out.append(cur)
            remaining -= seg
            prev = cur
            continue
        ratio = remaining / seg
        out.append((prev[0] + (cur[0] - prev[0]) * ratio, prev[1] + (cur[1] - prev[1]) * ratio))
        break
    return out


def _clip_past_route(ego_xy: Point2D, past_route_xy: Sequence[Point2D], distance_m: float) -> Sequence[Point2D]:
    if distance_m <= 0:
        return []
    points = [(float(x), float(y)) for x, y in past_route_xy]
    out = []
    remaining = float(distance_m)
    prev = ego_xy
    for cur in reversed(points):
        seg = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if seg <= 1e-6:
            continue
        if seg <= remaining:
            out.append(cur)
            remaining -= seg
            prev = cur
            continue
        ratio = remaining / seg
        out.append((prev[0] + (cur[0] - prev[0]) * ratio, prev[1] + (cur[1] - prev[1]) * ratio))
        break
    out.reverse()
    return out


def route_corridor_polyline(
    ego_pose: EgoPose,
    route_xy: Sequence[Point2D],
    past_route_xy: Sequence[Point2D] = (),
    *,
    past_route_distance_m: float,
    future_route_distance_m: float,
) -> Sequence[Point2D]:
    ego_xy = (float(ego_pose[0]), float(ego_pose[1]))
    past = _clip_past_route(ego_xy, past_route_xy, past_route_distance_m)
    future = _clip_forward_route(ego_xy, route_xy, future_route_distance_m)
    return tuple(past + list(future))


def _segments_intersect(a1: Point2D, a2: Point2D, b1: Point2D, b2: Point2D) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)


def _line_of_sight_clear(
    start: Point2D,
    end: Point2D,
    actor_polygons: Mapping[int, Sequence[Point2D]],
    *,
    ignore_ids: Sequence[int] = (),
) -> bool:
    ignore = {int(v) for v in ignore_ids}
    for actor_id, poly in actor_polygons.items():
        if int(actor_id) in ignore or len(poly) < 2:
            continue
        for a, b in zip(poly, list(poly[1:]) + [poly[0]]):
            if _segments_intersect(start, end, a, b):
                return False
    return True


def _point_in_fov(point: Point2D, observer_pose: ObserverPose, fov: float, sight_range: float) -> bool:
    _, ox, oy, yaw_deg = observer_pose
    dx, dy = float(point[0]) - ox, float(point[1]) - oy
    dist = math.hypot(dx, dy)
    if dist <= 1e-6 or dist > float(sight_range):
        return False
    direction = (math.cos(math.radians(float(yaw_deg))), math.sin(math.radians(float(yaw_deg))))
    dot = max(min((direction[0] * dx + direction[1] * dy) / dist, 1.0), -1.0)
    angle = math.degrees(math.acos(dot))
    return abs(angle) <= float(fov) / 2.0


def _observer_quality(
    centers: np.ndarray,
    route_mask: np.ndarray,
    observer_pose: ObserverPose,
    actor_polygons: Mapping[int, Sequence[Point2D]],
    *,
    fov: float,
    sight_range: float,
    distance_scale_m: float,
    freshness: float = 1.0,
) -> np.ndarray:
    """Per-cell coverage quality of one observer, scaled by ``freshness`` in [0, 1].

    ``freshness`` discounts a stale observation (e.g. an old V2V snapshot): a region that was seen a
    while ago is less trustworthy now, so it contributes less coverage. ``freshness=1`` is a current
    (ego / zero-latency) observation.
    """
    out = np.zeros(route_mask.shape, dtype=np.float32)
    obs_id, ox, oy, _ = observer_pose
    scale = max(float(distance_scale_m), 1e-6)
    fresh = float(max(min(freshness, 1.0), 0.0))
    if fresh <= 0.0:
        return out
    rows, cols = np.nonzero(route_mask > 0.5)
    for r, c in zip(rows, cols):
        point = (float(centers[r, c, 0]), float(centers[r, c, 1]))
        if not _point_in_fov(point, observer_pose, fov, sight_range):
            continue
        if not _line_of_sight_clear((float(ox), float(oy)), point, actor_polygons, ignore_ids=(int(obs_id),)):
            continue
        dist = math.hypot(point[0] - float(ox), point[1] - float(oy))
        out[r, c] = fresh / (1.0 + dist / scale)
    return out


def build_coverage_raster(
    *,
    ego_pose: EgoPose,
    route_xy: Sequence[Point2D],
    past_route_xy: Sequence[Point2D] = (),
    ego_observer: ObserverPose,
    collaborator_observers: Sequence[ObserverPose] = (),
    actor_polygons: Optional[Mapping[int, Sequence[Point2D]]] = None,
    ego_fov: float,
    ego_sight_range: float,
    collaborator_fov: float,
    collaborator_sight_range: float,
    config: CoverageConfig = CoverageConfig(),
    spec: BevSpec = BevSpec(),
    collaborator_freshness: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """``collaborator_freshness`` (parallel to ``collaborator_observers``) discounts each collaborator's
    coverage by message freshness in [0, 1]; omitted / ``None`` means 1.0 (fresh, e.g. zero-latency)."""
    centers = _cell_centers_world(ego_pose, spec)
    corridor = route_corridor_polyline(
        ego_pose,
        route_xy,
        past_route_xy,
        past_route_distance_m=config.past_route_distance_m,
        future_route_distance_m=config.future_route_distance_m,
    )

    ego_xy = (float(ego_pose[0]), float(ego_pose[1]))
    half_width = float(config.corridor_width_m) / 2.0
    route_scale = max(float(config.route_risk_distance_scale_m), 1e-6)
    dist_to_corridor = _cells_distance_to_polyline(centers, corridor)  # [H,W]
    route_mask = (dist_to_corridor <= half_width).astype(np.float32)
    dist_to_ego = np.hypot(centers[..., 0] - ego_xy[0], centers[..., 1] - ego_xy[1])
    route_risk = (route_mask / (1.0 + dist_to_ego / route_scale)).astype(np.float32)

    polygons = actor_polygons or {}
    ego_quality = _observer_quality(
        centers,
        route_mask,
        ego_observer,
        polygons,
        fov=float(ego_fov),
        sight_range=float(ego_sight_range),
        distance_scale_m=float(config.coverage_distance_scale_m),
    )
    collab_quality = np.zeros_like(route_mask)
    for i, obs in enumerate(collaborator_observers):
        freshness = 1.0
        if collaborator_freshness is not None and i < len(collaborator_freshness):
            freshness = float(collaborator_freshness[i])
        q = _observer_quality(
            centers,
            route_mask,
            obs,
            polygons,
            fov=float(collaborator_fov),
            sight_range=float(collaborator_sight_range),
            distance_scale_m=float(config.coverage_distance_scale_m),
            freshness=freshness,
        )
        collab_quality = np.maximum(collab_quality, q)

    total_quality = np.maximum(ego_quality, collab_quality)
    poor_risk = route_mask * route_risk * (1.0 - total_quality)
    raster = np.stack((route_mask, route_risk, ego_quality, collab_quality, total_quality, poor_risk), axis=0)
    metrics = coverage_metrics(raster, u_prior=float(config.u_prior))
    return raster.astype(np.float32), metrics


def coverage_metrics(raster: np.ndarray, *, u_prior: float = 1.0, eps: float = 1e-6) -> Dict[str, float]:
    if raster is None or np.asarray(raster).size == 0:
        return {
            "coverage_uncertainty": 0.0,
            "route_coverage_quality_mean": 0.0,
            "poor_coverage_risk_mean": 0.0,
            "route_coverage_ratio": 0.0,
        }
    arr = np.asarray(raster, dtype=np.float32)
    route_mask, route_risk, total_quality, poor_risk = arr[0], arr[1], arr[4], arr[5]
    weight = route_mask * route_risk
    denom = float(weight.sum())
    if denom <= eps:
        return {
            "coverage_uncertainty": 0.0,
            "route_coverage_quality_mean": 0.0,
            "poor_coverage_risk_mean": 0.0,
            "route_coverage_ratio": 0.0,
        }
    poor_mean = float(poor_risk.sum() / denom)
    covered = (route_mask > 0.5) & (total_quality > 0.0)
    route_cells = max(int((route_mask > 0.5).sum()), 1)
    return {
        "coverage_uncertainty": float(u_prior) * poor_mean,
        "route_coverage_quality_mean": float((weight * total_quality).sum() / denom),
        "poor_coverage_risk_mean": poor_mean,
        "route_coverage_ratio": float(covered.sum() / route_cells),
    }
