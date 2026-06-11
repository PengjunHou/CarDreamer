from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from car_dreamer.toolkit.observer.handlers.renderer.constants import Color
from car_dreamer.toolkit.wam.bev import BEV_CHANNEL_NAMES, BEV_NUM_CHANNELS


Point2D = Tuple[float, float]


EGO_COLOR = Color.RED
NOTABLE_OBJECT_COLOR = Color.GREEN
GT_TRAJECTORY_COLOR = Color.BLUE
PRED_TRAJECTORY_COLOR = Color.SKY_BLUE_0

_BEV_RGB_COLORS = np.asarray(
    [
        Color.SKY_BLUE_0,     # vehicle
        Color.ORANGE_0,       # pedestrian
        Color.PLUM_0,         # bicycle
        Color.ALUMINIUM_1,    # other
        Color.SCARLET_RED_0,  # ego
        Color.BUTTER_1,       # route
        Color.ALUMINIUM_5,    # drivable
    ],
    dtype=np.float32,
)


def _map_surface(map_renderer) -> np.ndarray:
    surface = getattr(map_renderer, "_surface", None)
    if surface is None:
        raise ValueError("map_renderer does not expose a _surface")
    canvas = np.zeros_like(surface)
    mask = np.any(surface > 0, axis=2)
    canvas[mask] = surface[mask]
    return canvas


def _world_to_pixel(map_renderer, x: float, y: float) -> Tuple[int, int]:
    scale = float(getattr(map_renderer, "_scale", 1.0))
    pixels_per_meter = float(getattr(map_renderer, "_pixels_per_meter"))
    offset_x, offset_y = getattr(map_renderer, "_world_offset_in_meter")
    px = scale * pixels_per_meter * (float(x) - float(offset_x))
    py = scale * pixels_per_meter * (float(y) - float(offset_y))
    return int(round(px)), int(round(py))


def _draw_polyline(
    canvas: np.ndarray,
    points: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    *,
    thickness: int,
    dashed: bool = False,
) -> None:
    if len(points) < 2:
        return
    if not dashed:
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
        return
    for idx in range(len(points) - 1):
        start = np.array(points[idx], dtype=float)
        end = np.array(points[idx + 1], dtype=float)
        dist = float(np.linalg.norm(end - start))
        if dist <= 1e-6:
            continue
        direction = (end - start) / dist
        cursor = 0.0
        dash = 10.0
        gap = 7.0
        while cursor < dist:
            seg_start = start + direction * cursor
            seg_end = start + direction * min(cursor + dash, dist)
            cv2.line(
                canvas,
                tuple(np.round(seg_start).astype(int)),
                tuple(np.round(seg_end).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cursor += dash + gap


def _draw_bbox_or_point(
    canvas: np.ndarray,
    map_renderer,
    item: Mapping[str, object],
    color: Tuple[int, int, int],
    *,
    fill: bool,
) -> Tuple[int, int]:
    bbox = item.get("bbox") or []
    if bbox:
        pts = np.array([_world_to_pixel(map_renderer, float(point[0]), float(point[1])) for point in bbox], dtype=np.int32)
        if len(pts) > 3:
            pts = cv2.convexHull(pts)
        pts_flat = pts.reshape(-1, 2)
        if fill:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, dst=canvas)
        else:
            cv2.fillPoly(canvas, [pts], color)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        return tuple(np.mean(pts_flat, axis=0).astype(int))

    position = item.get("position", [0.0, 0.0, 0.0])
    center = _world_to_pixel(map_renderer, float(position[0]), float(position[1]))
    cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
    return center


def _positions_from_waypoints(
    waypoints: Sequence[Mapping[str, object]],
) -> Sequence[Sequence[float]]:
    positions = []
    for waypoint in waypoints:
        position = waypoint.get("position")
        if position is not None:
            positions.append(position)
    return positions


def _draw_waypoint_dots(
    canvas: np.ndarray,
    points: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    *,
    radius: int,
) -> None:
    for point in points:
        cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)


def render_bev_raster_rgb(
    raster: np.ndarray,
    *,
    background: Tuple[int, int, int] = Color.BLACK,
) -> np.ndarray:
    """Render a WAM semantic BEV raster ``[C,H,W]`` as an RGB uint8 image.

    Channels are alpha-composited in the canonical BEV order. Later semantic layers such as ego and
    route remain visible over drivable/background cells.
    """
    arr = np.asarray(raster)
    if arr.ndim != 3:
        raise ValueError(f"expected BEV raster [C,H,W], got shape={arr.shape}")
    if int(arr.shape[0]) != BEV_NUM_CHANNELS:
        raise ValueError(f"expected {BEV_NUM_CHANNELS} BEV channels {BEV_CHANNEL_NAMES}, got {arr.shape[0]}")
    h, w = int(arr.shape[1]), int(arr.shape[2])
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = np.asarray(background, dtype=np.float32)
    occ = arr.astype(bool)
    for ch, color in enumerate(_BEV_RGB_COLORS):
        img[occ[ch]] = color
    return np.clip(img, 0, 255).astype(np.uint8)


def render_vehicle_centric_wam_bev(
    *,
    map_renderer,
    vehicle: Mapping[str, object],
    visible_objects: Sequence[Mapping[str, object]],
    output_path: Path,
    route_xy: Sequence[Point2D] = (),
    image_size_px: int = 512,
    bev_range_m: float = 64.0,
    ego_offset_m: float = 12.0,
    vehicle_color: Tuple[int, int, int] = Color.BLUE,
    object_color: Tuple[int, int, int] = Color.GREEN,
    route_color: Tuple[int, int, int] = Color.SKY_BLUE_0,
) -> None:
    """Render one vehicle-centric BEV frame with no text overlays.

    The caller supplies the already visibility-filtered objects for this vehicle; objects not in
    ``visible_objects`` are not drawn.
    """
    position = vehicle.get("position", (0.0, 0.0, 0.0))
    center_px = _world_to_pixel(map_renderer, float(position[0]), float(position[1]))
    source_pixels_per_meter = float(getattr(map_renderer, "_pixels_per_meter"))

    canvas = _map_surface(map_renderer)
    _draw_bbox_or_point(canvas, map_renderer, vehicle, vehicle_color, fill=False)
    for obj in visible_objects:
        _draw_bbox_or_point(canvas, map_renderer, obj, object_color, fill=False)
    if len(route_xy) >= 2:
        route_points = [_world_to_pixel(map_renderer, float(point[0]), float(point[1])) for point in route_xy]
        _draw_polyline(canvas, route_points, route_color, thickness=6, dashed=False)

    canvas = _ego_centric_warp(
        canvas,
        ego_center_px=center_px,
        ego_yaw_deg=float(vehicle.get("yaw", 0.0)),
        image_size_px=int(image_size_px),
        bev_range_m=float(bev_range_m),
        source_pixels_per_meter=source_pixels_per_meter,
        ego_offset_m=float(ego_offset_m),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _ego_centric_warp(
    canvas: np.ndarray,
    *,
    ego_center_px: Tuple[int, int],
    ego_yaw_deg: float,
    image_size_px: int,
    bev_range_m: float,
    source_pixels_per_meter: float,
    ego_offset_m: float,
) -> np.ndarray:
    output_pixels_per_meter = float(image_size_px) / float(bev_range_m)
    scale = output_pixels_per_meter / float(source_pixels_per_meter)
    pixels_ahead_vehicle = (float(bev_range_m) / 2.0 - float(ego_offset_m)) * output_pixels_per_meter
    matrix = cv2.getRotationMatrix2D(ego_center_px, float(ego_yaw_deg) + 90.0, scale)
    matrix[0][2] -= ego_center_px[0] - float(image_size_px) / 2.0
    matrix[1][2] -= ego_center_px[1] - float(image_size_px) / 2.0 - pixels_ahead_vehicle
    return cv2.warpAffine(
        canvas,
        matrix,
        (int(image_size_px), int(image_size_px)),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=Color.BLACK,
    )


def render_wam_bev_record(
    *,
    map_renderer,
    record: Mapping[str, object],
    output_path: Path,
    bev_range_m: float = 64.0,
    image_size_px: Optional[int] = None,
    ego_offset_m: float = 12.0,
) -> None:
    ego = record["ego"]
    ego_position = ego["position"]
    center_px = _world_to_pixel(map_renderer, float(ego_position[0]), float(ego_position[1]))
    size_px = int(image_size_px or 512)
    source_pixels_per_meter = float(getattr(map_renderer, "_pixels_per_meter"))

    canvas = _map_surface(map_renderer)

    _draw_bbox_or_point(
        canvas,
        map_renderer,
        ego,
        EGO_COLOR,
        fill=True,
    )

    for obj in record.get("notable_objects", []):
        _draw_bbox_or_point(
            canvas,
            map_renderer,
            obj,
            NOTABLE_OBJECT_COLOR,
            fill=False,
        )

        gt_wpts = obj.get("ground_truth", {}).get("future_waypoints", [])
        pred_wpts = obj.get("predicted", {}).get("future_waypoints", [])
        gt_points = [_world_to_pixel(map_renderer, float(point[0]), float(point[1])) for point in _positions_from_waypoints(gt_wpts)]
        pred_points = [
            _world_to_pixel(map_renderer, float(point[0]), float(point[1])) for point in _positions_from_waypoints(pred_wpts)
        ]
        obj_position = obj.get("position", [0.0, 0.0, 0.0])
        obj_start = _world_to_pixel(map_renderer, float(obj_position[0]), float(obj_position[1]))
        if gt_points:
            _draw_polyline(canvas, [obj_start, *gt_points], GT_TRAJECTORY_COLOR, thickness=3, dashed=False)
            _draw_waypoint_dots(canvas, gt_points, GT_TRAJECTORY_COLOR, radius=3)
        if pred_points:
            _draw_polyline(canvas, [obj_start, *pred_points], PRED_TRAJECTORY_COLOR, thickness=2, dashed=True)
            _draw_waypoint_dots(canvas, pred_points, PRED_TRAJECTORY_COLOR, radius=2)

    canvas = _ego_centric_warp(
        canvas,
        ego_center_px=center_px,
        ego_yaw_deg=float(ego.get("yaw", 0.0)),
        image_size_px=size_px,
        bev_range_m=float(bev_range_m),
        source_pixels_per_meter=source_pixels_per_meter,
        ego_offset_m=float(ego_offset_m),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
