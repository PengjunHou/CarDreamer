"""Per-vehicle, visibility-aware BEV semantic map (WAM Design §5.3 / §14 / §15.4).

A vehicle's BEV ``B^sem [C,H,W]`` is an **ego-centric, heading-up** semantic occupancy raster that contains
**only what that vehicle can see**: the caller passes the visible object subset (FOV + occlusion already
decided upstream via ``ObjectState.visible_to_ego`` / ``visible_to_collaborators``), so invisible objects are
excluded by construction.

Channels (fixed order):
    vehicle, pedestrian, bicycle, other   -- per-class occupancy of the visible objects
    ego                                   -- the vehicle's own footprint (centered)
    route                                 -- the vehicle's planned route polyline
    drivable                              -- drivable area (best-effort; empty until map polygons are wired)

This module is pure (numpy rasterizer + a small torch decoder), CARLA-free, and unit-tested:
    * :func:`rasterize_bev` -- world polygons -> ego frame -> pixels -> convex fill, ``np.uint8 [C,H,W]``.
    * :class:`WAMBevDecoder` (§14) + :func:`bev_reconstruction_loss` (§15.4) + :func:`bev_iou` -- the decoder
      half of the BEV autoencoder (the encoder ``E_bev`` lives in
      :class:`car_dreamer.toolkit.wam.graph_model.WAMHeteroGraphEmbedding`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import CLASS_TO_ID, OBJECT_CLASSES
from .runtime import ObjectState

Point2D = Tuple[float, float]
EgoPose = Tuple[float, float, float]  # (x, y, yaw_degrees), same convention as graph._EgoFrame

# Fixed channel layout: object classes + ego + route + drivable.
BEV_CHANNEL_NAMES: Tuple[str, ...] = OBJECT_CLASSES + ("ego", "route", "drivable")
BEV_NUM_CHANNELS: int = len(BEV_CHANNEL_NAMES)
_CH_EGO = len(OBJECT_CLASSES)
_CH_ROUTE = _CH_EGO + 1
_CH_DRIVABLE = _CH_EGO + 2


@dataclass(frozen=True)
class BevSpec:
    """Geometry of the BEV raster. ``range_m`` is the half-extent (meters from ego to each edge)."""

    size: int = 64
    range_m: float = 50.0
    route_width_m: float = 3.0
    ego_length: float = 4.5
    ego_width: float = 2.0

    @property
    def channels(self) -> int:
        return BEV_NUM_CHANNELS

    @property
    def pixels_per_meter(self) -> float:
        return float(self.size) / (2.0 * float(self.range_m))


# =====================================================================
# Geometry helpers (world -> ego -> pixel), matching graph._EgoFrame
# =====================================================================


def _world_to_ego(x: float, y: float, ego_pose: EgoPose) -> Point2D:
    ex, ey, yaw_deg = ego_pose
    yaw = math.radians(float(yaw_deg))
    cos_a, sin_a = math.cos(-yaw), math.sin(-yaw)
    dx, dy = float(x) - float(ex), float(y) - float(ey)
    return cos_a * dx - sin_a * dy, sin_a * dx + cos_a * dy


def _ego_to_pixel(ex: float, ey: float, spec: BevSpec) -> Point2D:
    """Ego-frame meters -> pixel ``(col, row)``. Forward (+x) is up; left (+y) is left; ego at center."""
    c0 = spec.size / 2.0
    ppm = spec.pixels_per_meter
    col = c0 - ey * ppm
    row = c0 - ex * ppm
    return col, row


def _object_world_footprint(obj: ObjectState) -> Sequence[Point2D]:
    """Object world footprint: use ``bbox`` if present, else derive from pose + (length, width)."""
    if obj.bbox and len(obj.bbox) >= 3:
        return [(float(px), float(py)) for px, py in obj.bbox]
    yaw = math.radians(float(obj.yaw))
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    hl, hw = float(obj.length) / 2.0, float(obj.width) / 2.0
    corners = []
    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        wx = float(obj.x) + lx * cos_y - ly * sin_y
        wy = float(obj.y) + lx * sin_y + ly * cos_y
        corners.append((wx, wy))
    return corners


def _ego_world_footprint(ego_pose: EgoPose, length: float, width: float) -> Sequence[Point2D]:
    ex, ey, yaw_deg = ego_pose
    yaw = math.radians(float(yaw_deg))
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    hl, hw = float(length) / 2.0, float(width) / 2.0
    corners = []
    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        corners.append((ex + lx * cos_y - ly * sin_y, ey + lx * sin_y + ly * cos_y))
    return corners


def _fill_convex(channel: np.ndarray, poly_px: Sequence[Point2D]) -> None:
    """Fill a convex polygon (pixel ``(col,row)`` corners) into ``channel [H,W]`` (sets 1.0 inside)."""
    size = channel.shape[0]
    xs = [p[0] for p in poly_px]
    ys = [p[1] for p in poly_px]
    x0 = max(int(math.floor(min(xs))), 0)
    x1 = min(int(math.ceil(max(xs))), size - 1)
    y0 = max(int(math.floor(min(ys))), 0)
    y1 = min(int(math.ceil(max(ys))), size - 1)
    if x1 < x0 or y1 < y0:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    pos = np.zeros(gx.shape, dtype=bool)
    neg = np.zeros(gx.shape, dtype=bool)
    n = len(poly_px)
    for i in range(n):
        ax, ay = poly_px[i]
        bx, by = poly_px[(i + 1) % n]
        cross = (bx - ax) * (gy - ay) - (by - ay) * (gx - ax)
        pos |= cross > 1e-9
        neg |= cross < -1e-9
    inside = ~(pos & neg)  # consistent sign across all edges -> inside the convex polygon
    if bool(inside.any()):
        channel[y0 : y1 + 1, x0 : x1 + 1][inside] = 1.0
    else:
        # sub-pixel polygon: no pixel center falls inside -> mark the cell nearest its centroid so small /
        # distant objects (and the ego footprint at coarse resolution) still register at least one cell.
        cc = int(round(sum(xs) / len(xs) - 0.5))
        rr = int(round(sum(ys) / len(ys) - 0.5))
        if 0 <= cc < size and 0 <= rr < size:
            channel[rr, cc] = 1.0


def _segment_quad(p0: Point2D, p1: Point2D, half_width: float) -> Optional[Sequence[Point2D]]:
    """A thin quad around segment ``p0->p1`` (ego-frame meters) for polyline rasterization."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    nx, ny = -dy / length * half_width, dx / length * half_width  # perpendicular offset
    return [
        (p0[0] + nx, p0[1] + ny),
        (p1[0] + nx, p1[1] + ny),
        (p1[0] - nx, p1[1] - ny),
        (p0[0] - nx, p0[1] - ny),
    ]


# =====================================================================
# Rasterizer
# =====================================================================


def rasterize_bev(
    ego_pose: EgoPose,
    objects: Sequence[ObjectState],
    *,
    route_xy: Sequence[Point2D] = (),
    drivable_polygons: Optional[Sequence[Sequence[Point2D]]] = None,
    spec: BevSpec = BevSpec(),
) -> np.ndarray:
    """Rasterize one vehicle's visibility-aware BEV ``B^sem`` -> ``np.uint8 [C,H,W]``.

    ``objects`` must already be the **visible** subset for this vehicle (invisible ones are simply not
    passed). Each object fills its class channel; ego/route/drivable fill their dedicated channels.
    """
    c, s = spec.channels, spec.size
    raster = np.zeros((c, s, s), dtype=np.float32)

    def fill(ch_idx: int, world_poly: Sequence[Point2D]) -> None:
        poly_px = [_ego_to_pixel(*_world_to_ego(wx, wy, ego_pose), spec) for wx, wy in world_poly]
        _fill_convex(raster[ch_idx], poly_px)

    # object class channels (visible objects only)
    for obj in objects:
        ch = CLASS_TO_ID.get(str(obj.object_class), CLASS_TO_ID["other"])
        fill(ch, _object_world_footprint(obj))

    # ego footprint
    fill(_CH_EGO, _ego_world_footprint(ego_pose, spec.ego_length, spec.ego_width))

    # route polyline (thickened segments)
    route = [(float(px), float(py)) for px, py in route_xy]
    if len(route) >= 2:
        hw = spec.route_width_m / 2.0
        for a, b in zip(route[:-1], route[1:]):
            ea = _ego_to_pixel(*_world_to_ego(a[0], a[1], ego_pose), spec)
            eb = _ego_to_pixel(*_world_to_ego(b[0], b[1], ego_pose), spec)
            # build the quad in pixel space directly (half-width scaled to pixels)
            quad = _segment_quad(ea, eb, hw * spec.pixels_per_meter)
            if quad is not None:
                _fill_convex(raster[_CH_ROUTE], quad)

    # drivable area (best-effort: only if polygons supplied)
    for poly in drivable_polygons or ():
        fill(_CH_DRIVABLE, poly)

    return (raster > 0.5).astype(np.uint8)


# =====================================================================
# BEV decoder D_bev (§14) + reconstruction loss (§15.4) + IoU
# =====================================================================


class WAMBevDecoder(nn.Module):
    """§14 BEV decoder ``D_bev: z^bev -> B̂^sem`` (per-cell occupancy logits ``[N,C,H,W]``)."""

    def __init__(self, latent_dim: int, channels: int = BEV_NUM_CHANNELS, size: int = 64, base_ch: int = 64):
        super().__init__()
        self.channels = int(channels)
        self.size = int(size)
        # start from an 8x8 spatial map and upsample to ``size`` (size must be a multiple of 8).
        self.start = 8
        if self.size % self.start != 0:
            raise ValueError(f"bev size ({self.size}) must be a multiple of {self.start}")
        self.base_ch = int(base_ch)
        self.fc = nn.Linear(int(latent_dim), self.base_ch * self.start * self.start)
        n_up = int(round(math.log2(self.size // self.start)))
        ups = []
        ch = self.base_ch
        for _ in range(n_up):
            ups += [
                nn.ConvTranspose2d(ch, max(ch // 2, 16), kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            ch = max(ch // 2, 16)
        self.ups = nn.Sequential(*ups)
        self.head = nn.Conv2d(ch, self.channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return z.new_zeros((z.shape[0], self.channels, self.size, self.size))
        x = self.fc(z).view(z.shape[0], self.base_ch, self.start, self.start)
        x = self.ups(x)
        return self.head(x)  # logits [N, C, size, size]


def bev_reconstruction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """§15.4 BEV reconstruction: per-channel BCE-with-logits occupancy. ``[N,C,H,W]`` (target in {0,1})."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
    if mask is not None:
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(loss)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def bev_iou(prob: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Mean occupancy IoU over channels (``prob``/``target``: ``[...,C,H,W]``)."""
    if prob.numel() == 0:
        return 0.0
    pred = (prob >= float(threshold)).float()
    tgt = (target >= 0.5).float()
    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter
    return float((inter + eps) / (union + eps))
