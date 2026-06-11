"""Record BS-centric Diffusion UWM training samples from the live env (WAM Design Update §16.2 data side).

:class:`WAMFlowDataRecorder` pairs the request-vehicle perception graph captured at step ``t``
(``sim._wam_graph``) with the **future collaboration-policy chunk** ``π_{q,t:t+H-1}`` once the horizon has
elapsed, then writes a self-contained ``.pt`` sample (consumed by
:class:`car_dreamer.toolkit.wam.stage2.WAMFlowDataset`).

The recorder buffers the per-step policy via :meth:`observe_policy` (called every step) and the per-step
sample skeleton via :meth:`register` (called for steps that become samples). At flush, the chunk slot
``h`` is the policy at ``t+h`` (encoded against the sample's candidate ordering), masked where missing.

The CARLA-specific extraction (graph / policy / candidate ids / notable ids / ego pose) lives in the
caller (``scripts/record_wam_flow_data.py``); this recorder is CARLA-free pure assembly, so its sample
construction is unit-testable.

Simplifications (v1): ``vehicle_graphs = [request_graph]`` (per-vehicle local graphs for all coverage
vehicles are deferred); the BEV history/future are zero placeholders.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .flow_matching import encode_policy_chunk
from .runtime import WAMPolicy
from .stage2 import make_flow_sample


class WAMFlowDataRecorder:
    """Buffer per-step policies + request-graph skeletons and emit ``.pt`` flow-matching samples."""

    def __init__(
        self,
        out_dir: Union[str, Path],
        *,
        fixed_dt: float = 0.1,
        horizon_s: float = 3.0,
        samples: int = 6,
        max_members: int = 8,
        num_formats: int = 2,
        bev_latent_dim: int = 256,
        history_window: int = 0,
        request_index: int = 0,
        prefix: str = "sample",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_dt = float(fixed_dt)
        self.horizon = int(samples)  # H: policy-chunk length
        # delayed flush: need policies up to t+H-1 before a sample's chunk is complete.
        self.horizon_steps = max(self.horizon - 1, 0)
        self.max_members = int(max_members)
        self.num_formats = int(num_formats)
        self.bev_latent_dim = int(bev_latent_dim)
        self.history_window = int(history_window)
        self.request_index = int(request_index)
        self.prefix = str(prefix)
        self._policy_history: Dict[int, WAMPolicy] = {}
        self._pending: Deque[Tuple[int, Dict[str, object]]] = deque()
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    def observe_policy(self, step: int, policy: Optional[WAMPolicy]) -> None:
        """Record the policy active at ``step`` (used to assemble future chunks). Called every step."""
        if policy is not None:
            self._policy_history[int(step)] = policy

    def register(
        self,
        step: int,
        *,
        graph,
        candidate_ids: Sequence[int],
        notable_object_ids: Sequence[int] = (),
        vehicle_graphs: Optional[Sequence] = None,
        bev_history: Optional[torch.Tensor] = None,
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
                    "bev_history": bev_history,
                },
            )
        )

    def _policy_chunk(self, step: int, candidate_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the policy chunk ``[H,M,P]`` at ``t..t+H-1`` + per-step presence mask ``[H]``."""
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

    def _emit(self, step: int, payload: Dict[str, object]) -> Path:
        candidate_ids = payload["candidate_ids"]
        policy_chunk, policy_step_mask = self._policy_chunk(step, candidate_ids)
        member_mask = torch.zeros(self.max_members)
        member_mask[: min(len(candidate_ids), self.max_members)] = 1.0
        sample = make_flow_sample(
            payload["graph"],
            policy_chunk,
            member_mask,
            policy_step_mask,
            vehicle_graphs=payload["vehicle_graphs"],
            request_index=self.request_index,
            notable_object_ids=payload["notable_object_ids"],
            bev_history=payload["bev_history"],
            bev_latent_dim=self.bev_latent_dim,
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
        """Emit all pending samples using whatever future policies are available (rest masked out)."""
        written: List[Path] = []
        while self._pending:
            step, payload = self._pending.popleft()
            written.append(self._emit(step, payload))
        self._drop_old_history()
        return written

    def _drop_old_history(self) -> None:
        if self._pending:
            min_needed = self._pending[0][0]
        elif self._policy_history:
            min_needed = max(self._policy_history)
        else:
            return
        for step in list(self._policy_history):
            if step < min_needed:
                del self._policy_history[step]
