"""Policy-augmented Stage-1 data helpers.

This module is CARLA-free glue for recording and evaluating Stage-1 samples under a
fixed family of collaboration policies. Live scripts provide actor state and optional
latency estimates; the helpers here enumerate policies, construct policy-conditioned
graphs, attach plain metadata, and write backward-compatible ``.pt`` samples.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .bev import BevSpec, rasterize_bev
from .coverage import coverage_metrics
from .debug_recording import future_sample_step_offsets
from .graph import (
    OBJECT_STATE_DIM,
    GraphBuildSpec,
    ObservationNodeInput,
    VehicleNodeInput,
    build_wam_hetero_graph,
)
from .heads import policy_uncertainty, trajectory_ade_fde
from .runtime import ObjectState, WAMPolicy
from .stage1 import _align_target, make_stage1_sample
from .stage1_recorder import _to_xy, valid_object_ids
from .targets import build_trajectory_targets

EgoPose = Tuple[float, float, float]
Point2D = Tuple[float, float]

STAGE1_POLICY_TYPES: Tuple[str, ...] = (
    "ego_only",
    "single_candidate_objlist",
    "single_candidate_bev",
    "all_candidates_objlist",
    "all_candidates_bev",
)


def policy_to_dict(policy: WAMPolicy) -> Dict[str, object]:
    """Serialize ``WAMPolicy`` into JSON/torch-save friendly primitives."""
    return {
        "selected_vehicle_ids": [int(v) for v in policy.selected_vehicle_ids],
        "modality_by_vehicle": {int(k): str(v) for k, v in policy.modality_by_vehicle.items()},
        "bandwidth_by_vehicle": {int(k): float(v) for k, v in policy.bandwidth_by_vehicle.items()},
        "frequency_steps": int(policy.frequency_steps),
        "reason": str(policy.reason),
    }


def policy_key(policy_type: str, policy: WAMPolicy) -> str:
    selected = ",".join(str(v) for v in policy.selected_vehicle_ids) or "none"
    modalities = ",".join(f"{int(k)}:{v}" for k, v in sorted(policy.modality_by_vehicle.items())) or "none"
    return f"{policy_type}|{selected}|{modalities}"


def enumerate_stage1_policies(
    candidate_vehicle_ids: Iterable[int],
    *,
    bandwidth_ratio: float,
    frequency_steps: int,
) -> List[Tuple[str, WAMPolicy]]:
    """Rule-enumerate the Stage-1 policy family for one live step."""
    candidates = tuple(sorted({int(v) for v in candidate_vehicle_ids}))
    freq = int(frequency_steps)
    out: List[Tuple[str, WAMPolicy]] = [
        (
            "ego_only",
            WAMPolicy(
                selected_vehicle_ids=(),
                modality_by_vehicle={},
                bandwidth_by_vehicle={},
                frequency_steps=freq,
                reason="stage1_ego_only",
            ),
        )
    ]

    def make_policy(policy_type: str, selected: Sequence[int], modality: str) -> WAMPolicy:
        selected_tuple = tuple(int(v) for v in selected)
        ratio = min(max(float(bandwidth_ratio), 0.0), 1.0) if selected_tuple else 0.0
        return WAMPolicy(
            selected_vehicle_ids=selected_tuple,
            modality_by_vehicle={vid: str(modality) for vid in selected_tuple},
            bandwidth_by_vehicle={vid: ratio for vid in selected_tuple},
            frequency_steps=freq,
            reason=f"stage1_{policy_type}",
        )

    for vid in candidates:
        out.append(("single_candidate_objlist", make_policy("single_candidate_objlist", (vid,), "objlist")))
    for vid in candidates:
        out.append(("single_candidate_bev", make_policy("single_candidate_bev", (vid,), "bev")))
    if candidates:
        out.append(("all_candidates_objlist", make_policy("all_candidates_objlist", candidates, "objlist")))
        out.append(("all_candidates_bev", make_policy("all_candidates_bev", candidates, "bev")))
    return out


def visible_object_ids_by_vehicle(
    ego_id: int,
    candidate_vehicle_ids: Iterable[int],
    objects: Sequence[ObjectState],
) -> Dict[int, List[int]]:
    """Return object ids visible to ego and each candidate vehicle."""
    out: Dict[int, List[int]] = {
        int(ego_id): [int(s.actor_id) for s in objects if bool(s.visible_to_ego)]
    }
    for vid in sorted({int(v) for v in candidate_vehicle_ids}):
        out[int(vid)] = [int(s.actor_id) for s in objects if int(vid) in s.visible_to_collaborators]
    return out


def objlist_payload_bytes(n_objects: int, *, overhead_bytes: int = 64) -> float:
    return float(max(int(n_objects), 0) * OBJECT_STATE_DIM * 4 + int(overhead_bytes))


def bev_payload_bytes(
    spec: BevSpec,
    *,
    mode: str = "feature",
    feature_dim: int = 256,
    feature_dtype_bytes: int = 4,
) -> float:
    if str(mode).lower() == "feature":
        return float(max(int(feature_dim), 0) * max(int(feature_dtype_bytes), 1))
    return float(int(spec.channels) * int(spec.size) ** 2)


def build_stage1_policy_graph(
    *,
    ego: VehicleNodeInput,
    collaborators: Sequence[VehicleNodeInput],
    objects: Sequence[ObjectState],
    policy: WAMPolicy,
    spec: GraphBuildSpec,
    notable_ids: Iterable[int] = (),
    latency_by_vehicle: Optional[Mapping[int, float]] = None,
    bev_spec: BevSpec = BevSpec(),
    bev_payload_mode: str = "feature",
    bev_feature_dim: int = 256,
    bev_feature_dtype_bytes: int = 4,
    gamma_freshness: float = 5.0,
    overhead_bytes: int = 64,
) -> object:
    """Build a policy-conditioned graph using the current step's visibility state."""
    latency_by_vehicle = {int(k): float(v) for k, v in (latency_by_vehicle or {}).items()}
    objects = list(objects)
    selected = [int(v) for v in policy.selected_vehicle_ids]
    collaborator_by_id = {int(v.actor_id): v for v in collaborators}

    ego_visible = [s for s in objects if bool(s.visible_to_ego)]
    ego_pose = (float(ego.x), float(ego.y), float(ego.yaw))
    observations: List[ObservationNodeInput] = [
        ObservationNodeInput(
            vehicle_id=int(ego.actor_id),
            modality="objlist",
            observed_object_ids=tuple(int(s.actor_id) for s in ego_visible),
            payload_bytes=objlist_payload_bytes(len(ego_visible), overhead_bytes=overhead_bytes),
            latency_s=0.0,
            freshness=1.0,
            quality=1.0,
            sample_age_s=0.0,
        ),
        ObservationNodeInput(
            vehicle_id=int(ego.actor_id),
            modality="bev",
            observed_object_ids=tuple(int(s.actor_id) for s in ego_visible),
            payload_bytes=bev_payload_bytes(
                bev_spec,
                mode=bev_payload_mode,
                feature_dim=bev_feature_dim,
                feature_dtype_bytes=bev_feature_dtype_bytes,
            ),
            latency_s=0.0,
            freshness=1.0,
            quality=1.0,
            sample_age_s=0.0,
            bev_raster=rasterize_bev(ego_pose, ego_visible, route_xy=ego.route_xy, spec=bev_spec),
        ),
    ]

    included_collaborators = [collaborator_by_id[vid] for vid in selected if vid in collaborator_by_id]
    for vid in selected:
        collab = collaborator_by_id.get(vid)
        if collab is None:
            continue
        modality = str(policy.modality_by_vehicle.get(vid, "objlist"))
        latency_s = float(latency_by_vehicle.get(vid, 0.0))
        freshness = math.exp(-float(gamma_freshness) * latency_s)
        if modality == "bev":
            vis_objs = [s for s in objects if int(vid) in s.visible_to_collaborators]
            veh_pose = (float(collab.x), float(collab.y), float(collab.yaw))
            observed_ids = tuple(int(s.actor_id) for s in vis_objs)
            payload = bev_payload_bytes(
                bev_spec,
                mode=bev_payload_mode,
                feature_dim=bev_feature_dim,
                feature_dtype_bytes=bev_feature_dtype_bytes,
            )
            bev_raster = rasterize_bev(veh_pose, vis_objs, route_xy=(), spec=bev_spec)
        else:
            observed_ids = tuple(int(s.actor_id) for s in objects if int(vid) in s.visible_to_collaborators)
            payload = objlist_payload_bytes(len(observed_ids), overhead_bytes=overhead_bytes)
            bev_raster = None
        observations.append(
            ObservationNodeInput(
                vehicle_id=int(vid),
                modality=modality,
                observed_object_ids=observed_ids,
                payload_bytes=payload,
                latency_s=latency_s,
                freshness=freshness,
                quality=1.0,
                sample_age_s=0.0,
                bev_raster=bev_raster,
            )
        )

    return build_wam_hetero_graph(
        ego=ego,
        collaborators=included_collaborators,
        objects=objects,
        observations=observations,
        policy=policy,
        spec=spec,
        notable_ids={int(i) for i in notable_ids},
        latency_by_vehicle=dict(latency_by_vehicle),
    )


def make_stage1_policy_metadata(
    *,
    step: int,
    episode_id: int,
    policy_type: str,
    policy: WAMPolicy,
    candidate_vehicle_ids: Iterable[int],
    notable_object_ids: Iterable[int],
    visible_ids_by_vehicle: Mapping[int, Sequence[int]],
    ego_pose: EgoPose,
    fixed_dt: float,
    selected_by_policy: bool = True,
) -> Dict[str, object]:
    return {
        "step": int(step),
        "episode_id": int(episode_id),
        "policy_type": str(policy_type),
        "policy": policy_to_dict(policy),
        "candidate_vehicle_ids": [int(v) for v in sorted({int(v) for v in candidate_vehicle_ids})],
        "notable_object_ids": [int(v) for v in notable_object_ids],
        "visible_object_ids_by_vehicle": {
            int(k): [int(v) for v in values] for k, values in visible_ids_by_vehicle.items()
        },
        "ego_pose": [float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])],
        "fixed_dt": float(fixed_dt),
        "selected_by_policy": bool(selected_by_policy),
    }


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class WAMStage1PolicyDataRecorder:
    """Record policy-augmented Stage-1 graph windows plus shared future GT."""

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
        manifest: Optional[Mapping[str, object]] = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_dt = float(fixed_dt)
        self.step_offsets = future_sample_step_offsets(self.fixed_dt, horizon_s, samples)
        self.horizon_steps = max(self.step_offsets)
        self.history_window = int(history_window)
        self.ego_frame = bool(ego_frame)
        self.prefix = str(prefix)
        self._windows: Dict[str, Deque] = {}
        self._history: Dict[int, Dict[int, Point2D]] = {}
        self._pending: Deque[Tuple[int, Dict[str, object]]] = deque()
        self._written = 0
        self._manifest: Dict[str, object] = dict(manifest or {})
        self._manifest.setdefault("policy_types", list(STAGE1_POLICY_TYPES))
        self._manifest.setdefault("history_window", self.history_window)
        self._manifest.setdefault("trajectory_horizon_steps", int(samples))
        self._manifest.setdefault("fixed_dt", self.fixed_dt)
        self.write_manifest()

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

    def reset_windows(self) -> None:
        self._windows.clear()

    def reset_episode(self) -> None:
        self._windows.clear()
        self._pending.clear()
        self._history.clear()

    def register(
        self,
        step: int,
        *,
        key: str,
        graph,
        ego_pose: EgoPose,
        metadata: Mapping[str, object],
        coverage=None,
    ) -> None:
        window = self._windows.setdefault(str(key), deque(maxlen=self.history_window + 1))
        window.append((graph, None if coverage is None else torch.as_tensor(coverage, dtype=torch.float32)))
        payload = {
            "window": [item[0] for item in window],
            "coverage_history": [item[1] for item in window],
            "object_node_ids": valid_object_ids(graph),
            "ego_pose": tuple(ego_pose),
            "metadata": dict(metadata),
        }
        self._pending.append((int(step), payload))

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
            metadata=payload.get("metadata"),
        )
        coverage_history = payload.get("coverage_history")
        if coverage_history and all(item is not None for item in coverage_history):
            sample["coverage_history"] = torch.stack(list(coverage_history), dim=0)
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
        if written:
            self.write_manifest()
        return written

    def flush_all(self) -> List[Path]:
        written: List[Path] = []
        while self._pending:
            step, payload = self._pending.popleft()
            written.append(self._emit(step, payload))
        self._drop_old_history()
        if written:
            self.write_manifest()
        return written

    def write_manifest(self) -> Path:
        manifest = dict(self._manifest)
        manifest["sample_count"] = int(self._written)
        manifest["files"] = [p.name for p in sorted(self.out_dir.glob(f"{self.prefix}_*.pt"))]
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

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


@torch.no_grad()
def evaluate_stage1_uncertainty_rows(
    model,
    samples: Sequence[Dict[str, object]],
    *,
    device: Union[str, torch.device] = "cpu",
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Run a Stage-1 model over samples and return CSV-ready uncertainty rows."""
    device = torch.device(device)
    model.to(device)
    was_training = bool(model.training)
    model.eval()
    rows: List[Dict[str, object]] = []
    for idx, sample in enumerate(samples):
        if limit is not None and idx >= int(limit):
            break
        window = [g.to(device) for g in sample["window"]]
        out = model(window)
        if int(out["object_node_ids"].numel()) == 0:
            uncertainty = 0.0
            ade_fde = {"ade": 0.0, "fde": 0.0}
        else:
            tgt, val = _align_target(
                out["object_node_ids"],
                sample["object_node_ids"],
                sample["target_xy"].to(device),
                sample["valid"].to(device),
            )
            uncertainty = float(policy_uncertainty(out["notable_prob"], out["traj_log_var"], valid_mask=val))
            notable = out["labels"].get("notable")
            ade_fde = trajectory_ade_fde(out["traj_mu"], tgt, valid_mask=val, notable_weight=notable)
        coverage = {"coverage_uncertainty": 0.0, "route_coverage_quality_mean": 0.0, "poor_coverage_risk_mean": 0.0}
        if "coverage_history" in sample:
            coverage_tensor = sample["coverage_history"]
            if hasattr(coverage_tensor, "detach"):
                coverage_arr = coverage_tensor[-1].detach().cpu().numpy()
            else:
                coverage_arr = coverage_tensor[-1]
            coverage = coverage_metrics(coverage_arr)
        total_uncertainty = float(uncertainty) + float(coverage.get("coverage_uncertainty", 0.0))
        metadata = dict(sample.get("metadata", {}))
        policy = dict(metadata.get("policy", {}))
        rows.append(
            {
                "step": int(metadata.get("step", -1)),
                "episode_id": int(metadata.get("episode_id", -1)),
                "policy_type": str(metadata.get("policy_type", "")),
                "selected_vehicle_ids": list(policy.get("selected_vehicle_ids", [])),
                "modality_by_vehicle": dict(policy.get("modality_by_vehicle", {})),
                "notable_object_ids": list(metadata.get("notable_object_ids", [])),
                "uncertainty": float(uncertainty),
                "motion_uncertainty": float(uncertainty),
                "coverage_uncertainty": float(coverage.get("coverage_uncertainty", 0.0)),
                "total_uncertainty": float(total_uncertainty),
                "route_coverage_quality_mean": float(coverage.get("route_coverage_quality_mean", 0.0)),
                "poor_coverage_risk_mean": float(coverage.get("poor_coverage_risk_mean", 0.0)),
                "ade": float(ade_fde["ade"]),
                "fde": float(ade_fde["fde"]),
            }
        )
    if was_training:
        model.train()
    return rows
