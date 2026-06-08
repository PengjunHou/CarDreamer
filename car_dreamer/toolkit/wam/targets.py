"""Ground-truth future-trajectory targets for the Gaussian Trajectory Head (§15.2).

The trajectory head predicts each current object's future positions in the **ego frame at time t**
(the same frame the object node states are built in, see
:func:`car_dreamer.toolkit.wam.graph.build_wam_hetero_graph`). To supervise it we need, for each
object node present at ``t``, that actor's actual world position at the sampled future steps,
transformed into the ego frame at ``t``.

This module provides:
  * :func:`build_trajectory_targets` -- a pure, simulator-free alignment + ego-frame transform.
  * :class:`TrajectoryTargetBuffer` -- a rolling buffer that pairs a graph captured at step ``t`` with
    its GT futures once the horizon has elapsed, mirroring the delayed-history pattern of
    :class:`car_dreamer.toolkit.wam.debug_recording.WAMNotableDebugRecorder`.

It is torch-free (returns numpy arrays); callers wrap the targets in tensors for the NLL.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .debug_recording import ActorSnapshot, future_sample_step_offsets

Point2D = Tuple[float, float]
EgoPose = Tuple[float, float, float]  # (x, y, yaw_degrees)


def _to_xy(value) -> Optional[Point2D]:
    if value is None:
        return None
    if isinstance(value, ActorSnapshot):
        return float(value.position[0]), float(value.position[1])
    return float(value[0]), float(value[1])


def _world_to_ego_xy(x: float, y: float, ego_pose: EgoPose, ego_frame: bool) -> Point2D:
    """Match the ego-frame convention of ``graph._EgoFrame.xy``."""
    ex, ey, yaw_deg = ego_pose
    if not ego_frame:
        return float(x), float(y)
    yaw = math.radians(float(yaw_deg))
    cos_a, sin_a = math.cos(-yaw), math.sin(-yaw)
    dx, dy = float(x) - float(ex), float(y) - float(ey)
    return cos_a * dx - sin_a * dy, sin_a * dx + cos_a * dy


def build_trajectory_targets(
    object_node_ids: Sequence[int],
    ego_pose: EgoPose,
    future_positions: Sequence[Mapping[int, object]],
    *,
    ego_frame: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``target_xy [Q, H, 2]`` (ego frame at ``t``) + ``valid_mask [Q, H]``.

    :param object_node_ids: actor ids of the current graph's object nodes (length ``Q``).
    :param ego_pose: ego ``(x, y, yaw_deg)`` at time ``t``.
    :param future_positions: length-``H`` sequence; ``future_positions[k]`` maps actor id -> world
        ``(x, y)`` (or :class:`ActorSnapshot`) at the ``k``-th sampled future step. Missing ids are
        masked out.
    """
    q = len(object_node_ids)
    h = len(future_positions)
    target = np.zeros((q, h, 2), dtype=np.float32)
    mask = np.zeros((q, h), dtype=np.float32)
    for qi, actor_id in enumerate(object_node_ids):
        for k in range(h):
            xy = _to_xy(future_positions[k].get(int(actor_id)))
            if xy is None:
                continue
            tx, ty = _world_to_ego_xy(xy[0], xy[1], ego_pose, ego_frame)
            target[qi, k] = (tx, ty)
            mask[qi, k] = 1.0
    return target, mask


class TrajectoryTargetBuffer:
    """Pairs graphs captured at step ``t`` with GT futures once ``horizon_steps`` have elapsed.

    Usage in a recording / replay loop::

        buf = TrajectoryTargetBuffer(fixed_dt=dt, horizon_s=3.0, samples=6)
        # each step:
        buf.observe(step, {actor_id: ActorSnapshot_or_xy, ...})
        buf.register(step, object_node_ids, ego_pose)   # when a graph is built at this step
        for step_t, target_xy, valid_mask in buf.flush_ready(step):
            ...  # train the trajectory head on (graph_at[step_t], target_xy, valid_mask)
    """

    def __init__(self, *, fixed_dt: float, horizon_s: float = 3.0, samples: int = 6, ego_frame: bool = True):
        self.fixed_dt = float(fixed_dt)
        self.horizon_s = float(horizon_s)
        self.samples = int(samples)
        self.ego_frame = bool(ego_frame)
        self.step_offsets = future_sample_step_offsets(self.fixed_dt, self.horizon_s, self.samples)
        self.horizon_steps = max(self.step_offsets)
        self._history: Dict[int, Dict[int, Point2D]] = {}
        self._pending: Deque[Tuple[int, List[int], EgoPose]] = deque()

    def observe(self, step: int, snapshots: Mapping[int, object]) -> None:
        positions: Dict[int, Point2D] = {}
        for actor_id, value in snapshots.items():
            xy = _to_xy(value)
            if xy is not None:
                positions[int(actor_id)] = xy
        self._history[int(step)] = positions

    def register(self, step: int, object_node_ids: Sequence[int], ego_pose: EgoPose) -> None:
        self._pending.append((int(step), [int(i) for i in object_node_ids], tuple(ego_pose)))

    def _futures_for(self, step: int) -> List[Dict[int, Point2D]]:
        return [dict(self._history.get(step + int(offset), {})) for offset in self.step_offsets]

    def _emit(self, step: int, ids: List[int], ego_pose: EgoPose) -> Tuple[int, np.ndarray, np.ndarray]:
        target, mask = build_trajectory_targets(
            ids, ego_pose, self._futures_for(step), ego_frame=self.ego_frame
        )
        return step, target, mask

    def flush_ready(self, current_step: int) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        ready: List[Tuple[int, np.ndarray, np.ndarray]] = []
        while self._pending and int(current_step) - self._pending[0][0] >= self.horizon_steps:
            step, ids, ego_pose = self._pending.popleft()
            ready.append(self._emit(step, ids, ego_pose))
            self._drop_old_history()
        return ready

    def flush_all(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Emit all pending graphs using whatever futures are available (rest masked out)."""
        ready: List[Tuple[int, np.ndarray, np.ndarray]] = []
        while self._pending:
            step, ids, ego_pose = self._pending.popleft()
            ready.append(self._emit(step, ids, ego_pose))
        self._drop_old_history()
        return ready

    def _drop_old_history(self) -> None:
        if self._pending:
            min_needed = self._pending[0][0]
        elif self._history:
            min_needed = max(self._history)
        else:
            return
        for step in list(self._history):
            if step < min_needed:
                del self._history[step]
