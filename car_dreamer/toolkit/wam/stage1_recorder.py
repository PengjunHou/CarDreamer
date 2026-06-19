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
from typing import Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .debug_recording import ActorSnapshot, future_sample_step_offsets
from .graph import OBJECT
from .stage1 import make_stage1_sample
from .targets import build_trajectory_targets

EgoPose = Tuple[float, float, float]
Point2D = Tuple[float, float]
GraphBuilder = Callable[[Mapping[str, object], Sequence[object], int], object]


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
        sample_period_s: Optional[float] = None,
        graph_builder: Optional[GraphBuilder] = None,
        receive_window_steps: Optional[int] = None,
        allow_cross_policy_messages: bool = False,
        ego_frame: bool = True,
        prefix: str = "sample",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_dt = float(fixed_dt)
        raw_sample_period_s = self.fixed_dt if sample_period_s is None else float(sample_period_s)
        if raw_sample_period_s <= 0:
            raise ValueError("sample_period_s must be positive")
        self.sample_period_s = raw_sample_period_s
        self.sample_period_steps = max(1, int(round(self.sample_period_s / self.fixed_dt)))
        self.step_offsets = future_sample_step_offsets(self.fixed_dt, horizon_s, samples)
        self.horizon_steps = max(self.step_offsets)
        self.history_window = int(history_window)
        self.graph_builder = graph_builder
        self.receive_window_steps = None if receive_window_steps is None else int(receive_window_steps)
        self.allow_cross_policy_messages = bool(allow_cross_policy_messages)
        self.ego_frame = bool(ego_frame)
        self.prefix = str(prefix)
        self._window: Deque[Tuple[int, object]] = deque(maxlen=self.history_window + 1)
        self._slot_window: Deque[Tuple[int, Mapping[str, object]]] = deque(maxlen=self.history_window + 1)
        self._history: Dict[int, Dict[int, Point2D]] = {}
        self._messages: Dict[object, object] = {}
        self._active_policy_by_step: Dict[int, Optional[int]] = {}
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

    def observe_messages(
        self,
        step: int,
        messages: Sequence[object],
        *,
        active_policy_id: Optional[int] = None,
    ) -> None:
        self._active_policy_by_step[int(step)] = None if active_policy_id is None else int(active_policy_id)
        for message in messages:
            msg_id = getattr(message, "msg_id", None)
            if msg_id is None:
                msg_id = (
                    int(getattr(message, "policy_id", -1)),
                    int(getattr(message, "sender_id", -1)),
                    int(getattr(message, "t_sense", -1)),
                    int(getattr(message, "t_recv", -1)),
                )
            self._messages[msg_id] = message

    def should_register_step(self, step: int) -> bool:
        return int(step) % int(self.sample_period_steps) == 0

    def register(self, step: int, *, graph, ego_pose: EgoPose) -> None:
        """Slide ``graph`` into the window and buffer a sample (window snapshot + last-graph object ids)."""
        self._window.append((int(step), graph))
        window_steps = [int(item[0]) for item in self._window]
        self._pending.append(
            (
                int(step),
                {
                    "window": [item[1] for item in self._window],
                    "window_steps": window_steps,
                    "object_node_ids": valid_object_ids(graph),
                    "ego_pose": tuple(ego_pose),
                },
            )
        )

    def register_slot(self, step: int, *, state: Mapping[str, object]) -> None:
        """Slide a graph source state into the window; graphs are rebuilt when the sample is emitted."""
        self._slot_window.append((int(step), dict(state)))
        window_steps = [int(item[0]) for item in self._slot_window]
        self._pending.append(
            (
                int(step),
                {
                    "source_window": [dict(item[1]) for item in self._slot_window],
                    "window_steps": window_steps,
                    "ego_pose": tuple(state["ego_pose"]),
                },
            )
        )

    def _futures_for(self, step: int) -> List[Dict[int, Point2D]]:
        return [dict(self._history.get(step + int(offset), {})) for offset in self.step_offsets]

    def _messages_for_slot(self, slot_step: int, prediction_step: int) -> List[object]:
        oldest = int(prediction_step) - int(self.receive_window_steps or 0)
        active_policy_id = self._active_policy_by_step.get(int(prediction_step))
        out = []
        for message in self._messages.values():
            t_sense = int(getattr(message, "t_sense"))
            if t_sense != int(slot_step):
                continue
            if int(getattr(message, "t_recv")) > int(prediction_step):
                continue
            if self.receive_window_steps is not None and not (oldest <= t_sense <= int(prediction_step)):
                continue
            if (
                not self.allow_cross_policy_messages
                and active_policy_id is not None
                and int(getattr(message, "policy_id")) != int(active_policy_id)
            ):
                continue
            out.append(message)
        return out

    def _build_window_from_sources(self, prediction_step: int, payload: Mapping[str, object]):
        if self.graph_builder is None:
            raise RuntimeError("register_slot() requires WAMStage1DataRecorder(graph_builder=...)")
        graphs = []
        slot_message_counts = []
        slot_selected_vehicle_ids = []
        for slot_step, state in zip(payload["window_steps"], payload["source_window"]):
            messages = self._messages_for_slot(int(slot_step), int(prediction_step))
            graph = self.graph_builder(state, messages, int(prediction_step))
            graphs.append(graph)
            slot_message_counts.append(len(messages))
            slot_selected_vehicle_ids.append(sorted({int(getattr(message, "sender_id")) for message in messages}))
        return graphs, slot_message_counts, slot_selected_vehicle_ids

    def _emit(self, step: int, payload: Dict[str, object]) -> Path:
        if "source_window" in payload:
            window, slot_message_counts, slot_selected_vehicle_ids = self._build_window_from_sources(step, payload)
            object_node_ids = valid_object_ids(window[-1]) if window else []
        else:
            window = payload["window"]
            slot_message_counts = [0 for _ in payload.get("window_steps", ())]
            slot_selected_vehicle_ids = [[] for _ in payload.get("window_steps", ())]
            object_node_ids = payload["object_node_ids"]
        target_xy, valid = build_trajectory_targets(
            object_node_ids,
            payload["ego_pose"],
            self._futures_for(step),
            ego_frame=self.ego_frame,
        )
        sample = make_stage1_sample(
            window,
            torch.from_numpy(target_xy),
            torch.from_numpy(valid),
            object_node_ids,
            metadata={
                "step": int(step),
                "window_steps": [int(v) for v in payload.get("window_steps", ())],
                "slot_message_counts": [int(v) for v in slot_message_counts],
                "slot_selected_vehicle_ids": [[int(v) for v in ids] for ids in slot_selected_vehicle_ids],
                "fixed_dt": float(self.fixed_dt),
                "sample_period_s": float(self.sample_period_s),
                "sample_period_steps": int(self.sample_period_steps),
            },
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
        if self.receive_window_steps is not None:
            oldest_message_step = int(min_needed) - int(self.receive_window_steps)
            for key, message in list(self._messages.items()):
                if int(getattr(message, "t_sense", oldest_message_step)) < oldest_message_step:
                    del self._messages[key]
            for step in list(self._active_policy_by_step):
                if int(step) < oldest_message_step:
                    del self._active_policy_by_step[step]
