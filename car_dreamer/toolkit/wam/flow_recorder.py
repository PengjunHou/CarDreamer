"""Record BS-centric Diffusion UWM training samples from the live env (WAM Design Update §16.2 data side).

:class:`WAMFlowDataRecorder` pairs the request-vehicle perception graph captured at step ``t``
(``sim._wam_graph``) with the **future collaboration-policy chunk** ``π_{q,t:t+H-1}`` and the **future
request-vehicle BEV semantic rasters** ``B^sem_{q,t+1:t+H}`` once the horizon has elapsed, then writes a
self-contained ``.pt`` sample (consumed by :class:`car_dreamer.toolkit.wam.stage2.WAMFlowDataset`).

The recorder buffers per step: the active policy (:meth:`observe_policy`) and the request vehicle's
visibility-aware BEV raster (:meth:`observe_bev`, rasterized from the objects that vehicle can see); plus
the sample skeleton (:meth:`register`). At flush, the policy chunk slot ``h`` is the policy at ``t+h``
(encoded against the sample's candidate ordering); the BEV target is ``B^sem`` at ``t+1..t+H``; the BEV
history is ``B^sem`` at ``t-K..t``. Rasters are stored as ``uint8`` occupancy to keep ``.pt`` small; the
Stage-2 trainer encodes them with the shared ``E_bev``.

The CARLA-specific extraction lives in ``scripts/record_wam_flow_data.py``; this recorder is CARLA-free
pure assembly (rasterization included), so it is unit-testable.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .bev import BevSpec, rasterize_bev
from .flow_matching import encode_policy_chunk
from .runtime import ObjectState, WAMPolicy
from .stage2 import make_flow_sample

EgoPose = Tuple[float, float, float]
Point2D = Tuple[float, float]


class WAMFlowDataRecorder:
    """Buffer per-step policies + BEV rasters + request-graph skeletons and emit ``.pt`` samples."""

    def __init__(
        self,
        out_dir: Union[str, Path],
        *,
        fixed_dt: float = 0.1,
        horizon_s: float = 3.0,
        samples: int = 6,
        max_members: int = 8,
        num_formats: int = 2,
        history_window: int = 0,
        request_index: int = 0,
        bev_spec: Optional[BevSpec] = None,
        enable_bev: bool = True,
        prefix: str = "sample",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_dt = float(fixed_dt)
        self.horizon = int(samples)  # H: policy-chunk length / BEV future length
        self.enable_bev = bool(enable_bev)
        self.bev_spec = bev_spec if bev_spec is not None else BevSpec()
        # delayed flush: policy chunk needs t..t+H-1; future BEV needs t+1..t+H -> delay H.
        self.horizon_steps = self.horizon if self.enable_bev else max(self.horizon - 1, 0)
        self.max_members = int(max_members)
        self.num_formats = int(num_formats)
        self.history_window = int(history_window)
        self.request_index = int(request_index)
        self.prefix = str(prefix)
        self._policy_history: Dict[int, WAMPolicy] = {}
        self._bev_by_step: Dict[int, np.ndarray] = {}
        self._pending: Deque[Tuple[int, Dict[str, object]]] = deque()
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    def observe_policy(self, step: int, policy: Optional[WAMPolicy]) -> None:
        """Record the policy active at ``step`` (used to assemble future chunks). Called every step."""
        if policy is not None:
            self._policy_history[int(step)] = policy

    def observe_bev(
        self,
        step: int,
        ego_pose: EgoPose,
        visible_objects: Sequence[ObjectState],
        route_xy: Sequence[Point2D] = (),
    ) -> None:
        """Rasterize + record the request vehicle's visibility-aware ``B^sem`` at ``step``. Called every step."""
        if not self.enable_bev:
            return
        self._bev_by_step[int(step)] = rasterize_bev(
            ego_pose, list(visible_objects), route_xy=route_xy, spec=self.bev_spec
        )

    def register(
        self,
        step: int,
        *,
        graph,
        candidate_ids: Sequence[int],
        notable_object_ids: Sequence[int] = (),
        vehicle_graphs: Optional[Sequence] = None,
    ) -> None:
        """Register the sample skeleton at ``step`` (request graph + candidate ordering + notable ids)."""
        self._pending.append(
            (
                int(step),
                {
                    "graph": graph,
                    "vehicle_graphs": list(vehicle_graphs) if vehicle_graphs is not None else None,
                    "candidate_ids": [int(i) for i in candidate_ids],
                    "notable_object_ids": [int(i) for i in notable_object_ids],
                },
            )
        )

    def _policy_chunk(self, step: int, candidate_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        policies: List[WAMPolicy] = []
        present: List[float] = []
        empty = WAMPolicy((), {}, {}, 0, "missing")
        for h in range(self.horizon):
            pol = self._policy_history.get(step + h)
            policies.append(pol if pol is not None else empty)
            present.append(1.0 if pol is not None else 0.0)
        chunk = encode_policy_chunk(
            policies, candidate_ids, max_members=self.max_members, num_formats=self.num_formats
        )
        return chunk, torch.tensor(present, dtype=torch.float32)

    def _bev_history(self, step: int) -> Optional[torch.Tensor]:
        """``[K+1, C, H, W]`` uint8 rasters for steps ``t-K..t`` (missing -> zeros)."""
        if not self.enable_bev:
            return None
        c, s = self.bev_spec.channels, self.bev_spec.size
        k = self.history_window
        out = np.zeros((k + 1, c, s, s), dtype=np.uint8)
        for i, st in enumerate(range(step - k, step + 1)):
            raster = self._bev_by_step.get(st)
            if raster is not None:
                out[i] = raster
        return torch.from_numpy(out)

    def _bev_future(self, step: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``([H, C, H, W] uint8, [H] mask)`` for steps ``t+1..t+H`` (missing -> zeros + mask 0)."""
        if not self.enable_bev:
            return None, None
        c, s = self.bev_spec.channels, self.bev_spec.size
        out = np.zeros((self.horizon, c, s, s), dtype=np.uint8)
        mask = np.zeros((self.horizon,), dtype=np.float32)
        for h in range(self.horizon):
            raster = self._bev_by_step.get(step + 1 + h)
            if raster is not None:
                out[h] = raster
                mask[h] = 1.0
        return torch.from_numpy(out), torch.from_numpy(mask)

    def _emit(self, step: int, payload: Dict[str, object]) -> Path:
        candidate_ids = payload["candidate_ids"]
        policy_chunk, policy_step_mask = self._policy_chunk(step, candidate_ids)
        member_mask = torch.zeros(self.max_members)
        member_mask[: min(len(candidate_ids), self.max_members)] = 1.0
        bev_future, bev_step_mask = self._bev_future(step)
        sample = make_flow_sample(
            payload["graph"],
            policy_chunk,
            member_mask,
            policy_step_mask,
            vehicle_graphs=payload["vehicle_graphs"],
            request_index=self.request_index,
            notable_object_ids=payload["notable_object_ids"],
            bev_history=self._bev_history(step),
            bev_future=bev_future,
            bev_step_mask=bev_step_mask,
            bev_spec=self.bev_spec,
            history_window=self.history_window,
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
        """Emit all pending samples using whatever future policies / BEV rasters are available."""
        written: List[Path] = []
        while self._pending:
            step, payload = self._pending.popleft()
            written.append(self._emit(step, payload))
        self._drop_old_history()
        return written

    def _drop_old_history(self) -> None:
        if self._pending:
            min_needed = self._pending[0][0] - self.history_window
        elif self._policy_history or self._bev_by_step:
            min_needed = max(list(self._policy_history) + list(self._bev_by_step)) - self.history_window
        else:
            return
        for step in list(self._policy_history):
            if step < min_needed:
                del self._policy_history[step]
        for step in list(self._bev_by_step):
            if step < min_needed:
                del self._bev_by_step[step]
