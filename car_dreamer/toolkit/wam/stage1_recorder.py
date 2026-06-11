"""Record Stage-1 training windows from the live env (WAM Design §16.1 data side).

:class:`WAMStage1DataRecorder` mirrors :class:`car_dreamer.toolkit.wam.targets.TrajectoryTargetBuffer`:
it maintains a sliding window of the last ``K+1`` policy-conditioned graphs and, once the trajectory
horizon has elapsed, pairs the window captured at step ``t`` with the GT future positions of the **last
graph's valid object nodes**, writing a self-contained ``.pt`` sample (consumed by
:class:`car_dreamer.toolkit.wam.stage1.WAMStage1Dataset`).

Perception labels (notable/visible/invisible) ride inside the graphs' object nodes, so no separate label
tensor is recorded. The CARLA-specific extraction (graph / actor world positions / ego pose) lives in the
caller (``scripts/record_wam_stage1_data.py``); this recorder is CARLA-free pure assembly, so it is
unit-testable.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .debug_recording import ActorSnapshot, future_sample_step_offsets
from .graph import OBJECT
from .stage1 import make_stage1_sample
from .targets import build_trajectory_targets

EgoPose = Tuple[float, float, float]
Point2D = Tuple[float, float]


def _to_xy(value) -> Optional[Point2D]:
    if value is None:
        return None
    if isinstance(value, ActorSnapshot):
        return float(value.position[0]), float(value.position[1])
    return float(value[0]), float(value[1])


def valid_object_ids(graph) -> List[int]:
    """The last graph's valid object node ids (node_id≥0 & node_mask>0.5, in node order).

    Matches the filter :meth:`WAMPerceptionModel.forward` uses to form its query objects, so a recorded
    target row order aligns with the model's ``object_node_ids`` output.
    """
    node_id = graph[OBJECT].node_id
    valid = node_id >= 0
    mask = getattr(graph[OBJECT], "node_mask", None)
    if mask is not None:
        valid = valid & (mask > 0.5)
    return [int(i) for i in node_id[valid].tolist()]


class WAMStage1DataRecorder:
    """Buffer per-step graph windows + GT futures and emit ``.pt`` Stage-1 window samples."""

    def __init__(
        self,
        out_dir: Union[str, Path],
        *,
        fixed_dt: float,
        horizon_s: float = 3.0,
        samples: int = 6,
        history_window: int = 4,
        ego_frame: bool = True,
        prefix: str = "sample",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_dt = float(fixed_dt)
        self.step_offsets = future_sample_step_offsets(self.fixed_dt, horizon_s, samples)
        self.horizon_steps = max(self.step_offsets)
        self.history_window = int(history_window)
        self.ego_frame = bool(ego_frame)
        self.prefix = str(prefix)
        self._window: Deque = deque(maxlen=self.history_window + 1)
        self._history: Dict[int, Dict[int, Point2D]] = {}
        self._pending: Deque[Tuple[int, Dict[str, object]]] = deque()
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    def observe(self, step: int, snapshots: Mapping[int, object]) -> None:
        positions: Dict[int, Point2D] = {}
        for actor_id, value in snapshots.items():
            xy = _to_xy(value)
            if xy is not None:
                positions[int(actor_id)] = xy
        self._history[int(step)] = positions

    def register(self, step: int, *, graph, ego_pose: EgoPose) -> None:
        """Slide ``graph`` into the window and buffer a sample (window snapshot + last-graph object ids)."""
        self._window.append(graph)
        self._pending.append(
            (
                int(step),
                {
                    "window": list(self._window),
                    "object_node_ids": valid_object_ids(graph),
                    "ego_pose": tuple(ego_pose),
                },
            )
        )

    def _futures_for(self, step: int) -> List[Dict[int, Point2D]]:
        return [dict(self._history.get(step + int(offset), {})) for offset in self.step_offsets]

    def _emit(self, step: int, payload: Dict[str, object]) -> Path:
        target_xy, valid = build_trajectory_targets(
            payload["object_node_ids"],
            payload["ego_pose"],
            self._futures_for(step),
            ego_frame=self.ego_frame,
        )
        sample = make_stage1_sample(
            payload["window"],
            torch.from_numpy(target_xy),
            torch.from_numpy(valid),
            payload["object_node_ids"],
        )
        path = self.out_dir / f"{self.prefix}_{self._written:06d}.pt"
        torch.save(sample, path)
        self._written += 1
        return path

    def flush_ready(self, current_step: int) -> List[Path]:
        written: List[Path] = []
        while self._pending and int(current_step) - self._pending[0][0] >= self.horizon_steps:
            step, payload = self._pending.popleft()
            written.append(self._emit(step, payload))
            self._drop_old_history()
        return written

    def flush_all(self) -> List[Path]:
        """Emit all pending samples using whatever futures are available (rest masked out)."""
        written: List[Path] = []
        while self._pending:
            step, payload = self._pending.popleft()
            written.append(self._emit(step, payload))
        self._drop_old_history()
        return written

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
