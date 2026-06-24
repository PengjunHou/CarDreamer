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
from typing import Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ..communication import CommConfig, CommPolicy, CommunicationProcess, SenseSnapshot
from .bev import BevSpec, rasterize_bev
from .coverage import coverage_metrics
from .debug_recording import future_sample_step_offsets
from .graph import (
    OBJECT,
    OBJECT_STATE_DIM,
    VEHICLE,
    GraphBuildSpec,
    ObservationNodeInput,
    VehicleNodeInput,
    build_wam_hetero_graph,
)
from .heads import per_object_trace, policy_uncertainty, trajectory_ade_fde
from .runtime import ObjectState, WAMPolicy
from .stage1 import _align_labels, _align_target, make_stage1_sample
from .stage1_recorder import _to_xy, perception_labels_at_t, union_object_ids, valid_object_ids
from .targets import build_trajectory_targets

EgoPose = Tuple[float, float, float]
Point2D = Tuple[float, float]
LinkRateFn = Callable[[int, float, float], float]

COMM_REPLAY_METADATA_FIELDS: Tuple[str, ...] = (
    "replay_mode",
    "comm_window_slots",
    "comm_window_v2v_slots",
    "comm_window_v2v_slot_rate",
    "comm_window_has_v2v_graph",
    "comm_final_has_v2v_graph",
    "comm_final_ego_visible_objects",
    "comm_final_collab_only_objects",
    "comm_final_total_objects",
    "comm_final_collab_object_ratio",
    "comm_generated_messages",
    "comm_received_messages_by_prediction_step",
)

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
        live_states: Optional[Sequence[object]] = None,
        notable_ids: Sequence[int] = (),
    ) -> None:
        window = self._windows.setdefault(str(key), deque(maxlen=self.history_window + 1))
        window.append((graph, None if coverage is None else torch.as_tensor(coverage, dtype=torch.float32)))
        window_graphs = [item[0] for item in window]
        # Predict/supervise the union over this policy's window (matches WAMPerceptionModel.forward), and
        # attach t-time GT labels when the caller provides the prediction-step live states (else None ->
        # the trainer falls back to per-graph node labels).
        object_node_ids = union_object_ids(window_graphs)
        payload = {
            "window": window_graphs,
            "coverage_history": [item[1] for item in window],
            "object_node_ids": object_node_ids,
            "perception_labels": perception_labels_at_t(object_node_ids, live_states, notable_ids),
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
            perception_labels=payload.get("perception_labels"),
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


def _policy_modalities(value) -> Tuple[str, ...]:
    if value is None:
        return ("objlist",)
    if isinstance(value, str):
        return (str(value),)
    try:
        return tuple(str(v) for v in value)
    except TypeError:
        return (str(value),)


def _node_distance(a: VehicleNodeInput, b: VehicleNodeInput) -> float:
    return float(math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))


def _graph_has_v2v_vehicle(graph) -> bool:
    if graph is None or VEHICLE not in graph.node_types:
        return False
    veh = graph[VEHICLE]
    node_id = getattr(veh, "node_id", None)
    if node_id is None:
        return False
    valid = node_id >= 0
    node_mask = getattr(veh, "node_mask", None)
    if node_mask is not None:
        valid = valid & (node_mask > 0.5)
    return int(valid.sum().item()) > 1


def _graph_object_visibility_sets(graph) -> Dict[str, set]:
    if graph is None or OBJECT not in graph.node_types:
        return {"all": set(), "ego_visible": set(), "collab_only": set()}
    obj = graph[OBJECT]
    node_id = getattr(obj, "node_id", None)
    if node_id is None:
        return {"all": set(), "ego_visible": set(), "collab_only": set()}
    valid = node_id >= 0
    node_mask = getattr(obj, "node_mask", None)
    if node_mask is not None:
        valid = valid & (node_mask > 0.5)
    visible = getattr(obj, "visible", torch.zeros_like(node_id, dtype=torch.float32)) > 0.5
    invisible = getattr(obj, "invisible", torch.zeros_like(node_id, dtype=torch.float32)) > 0.5
    return {
        "all": set(int(v) for v in node_id[valid].detach().cpu().tolist()),
        "ego_visible": set(int(v) for v in node_id[valid & visible].detach().cpu().tolist()),
        "collab_only": set(int(v) for v in node_id[valid & invisible].detach().cpu().tolist()),
    }


def _empty_comm_replay_stats() -> Dict[str, float]:
    return {
        "comm_window_slots": 0.0,
        "comm_window_v2v_slots": 0.0,
        "comm_window_v2v_slot_rate": 0.0,
        "comm_window_has_v2v_graph": 0.0,
        "comm_final_has_v2v_graph": 0.0,
        "comm_final_ego_visible_objects": 0.0,
        "comm_final_collab_only_objects": 0.0,
        "comm_final_total_objects": 0.0,
        "comm_final_collab_object_ratio": 0.0,
        "comm_generated_messages": 0.0,
        "comm_received_messages_by_prediction_step": 0.0,
    }


class _CommunicationReplayRuntime:
    def __init__(
        self,
        *,
        key: str,
        policy_type: str,
        policy: WAMPolicy,
        comm_policy: CommPolicy,
        process: CommunicationProcess,
    ) -> None:
        self.key = str(key)
        self.policy_type = str(policy_type)
        self.policy = policy
        self.comm_policy = comm_policy
        self.process = process
        self.messages: Dict[object, object] = {}
        self.generated_messages = 0

    def remember(self, messages: Sequence[object]) -> None:
        for message in messages:
            msg_id = getattr(message, "msg_id", None)
            if msg_id is None:
                msg_id = (
                    int(getattr(message, "policy_id", -1)),
                    int(getattr(message, "sender_id", -1)),
                    int(getattr(message, "t_sense", -1)),
                    int(getattr(message, "t_recv", -1)),
                )
            self.messages[msg_id] = message


class WAMStage1CommunicationPolicyDataRecorder:
    """Record fixed-policy Stage-1 samples while replaying V2V communication delays offline."""

    def __init__(
        self,
        out_dir: Union[str, Path],
        *,
        fixed_dt: float,
        comm_config: CommConfig,
        link_rate_bps: LinkRateFn,
        graph_builder: Callable[[Mapping[str, object], Sequence[object], int], object],
        coverage_builder: Optional[Callable[[Mapping[str, object], Sequence[object], int], object]] = None,
        horizon_s: float = 3.0,
        samples: int = 6,
        history_window: int = 4,
        ego_frame: bool = True,
        prefix: str = "sample",
        manifest: Optional[Mapping[str, object]] = None,
        bandwidth_ratio: float = 1.0,
        bev_spec: BevSpec = BevSpec(),
        bev_payload_mode: str = "feature",
        bev_feature_dim: int = 256,
        bev_feature_dtype_bytes: int = 4,
        overhead_bytes: int = 64,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_dt = float(fixed_dt)
        self.comm_config = comm_config
        self.link_rate_bps = link_rate_bps
        self.graph_builder = graph_builder
        self.coverage_builder = coverage_builder
        self.step_offsets = future_sample_step_offsets(self.fixed_dt, horizon_s, samples)
        self.horizon_steps = max(self.step_offsets)
        self.history_window = int(history_window)
        self.ego_frame = bool(ego_frame)
        self.prefix = str(prefix)
        self.bandwidth_ratio = float(bandwidth_ratio)
        self.bev_spec = bev_spec
        self.bev_payload_mode = str(bev_payload_mode)
        self.bev_feature_dim = int(bev_feature_dim)
        self.bev_feature_dtype_bytes = int(bev_feature_dtype_bytes)
        self.overhead_bytes = int(overhead_bytes)
        self._slot_window: Deque[Tuple[int, Mapping[str, object]]] = deque(maxlen=self.history_window + 1)
        self._history: Dict[int, Dict[int, Point2D]] = {}
        self._pending: Deque[Tuple[int, Dict[str, object]]] = deque()
        self._runtimes: Dict[str, _CommunicationReplayRuntime] = {}
        self._policy_order: List[str] = []
        self._written = 0
        self._manifest: Dict[str, object] = dict(manifest or {})
        self._manifest.setdefault("policy_types", list(STAGE1_POLICY_TYPES))
        self._manifest.setdefault("history_window", self.history_window)
        self._manifest.setdefault("trajectory_horizon_steps", int(samples))
        self._manifest.setdefault("fixed_dt", self.fixed_dt)
        self._manifest.setdefault("policy_replay_mode", "communication")
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

    def reset_episode(self) -> None:
        self._slot_window.clear()
        self._pending.clear()
        self._history.clear()
        self._runtimes.clear()
        self._policy_order.clear()

    def _ensure_runtimes(self, state: Mapping[str, object], *, start_step: int) -> None:
        if self._runtimes:
            return
        ego = state["ego"]
        collaborators = tuple(state.get("collaborators", ()))
        candidate_ids = [int(v.actor_id) for v in collaborators]
        policies = enumerate_stage1_policies(
            candidate_ids,
            bandwidth_ratio=float(self.bandwidth_ratio),
            frequency_steps=int(self.comm_config.sensor_period_steps),
        )
        duration_steps = max(int(self.comm_config.policy_duration_steps), 1_000_000_000)
        for idx, (policy_type, policy) in enumerate(policies):
            key = policy_key(policy_type, policy)
            comm_policy = CommPolicy(
                policy_id=int(idx),
                request_vehicle_id=int(ego.actor_id),
                start_step=int(start_step),
                duration_steps=int(duration_steps),
                selected_collaborators=tuple(int(v) for v in policy.selected_vehicle_ids),
                modalities_by_vehicle={
                    int(vid): _policy_modalities(policy.modality_by_vehicle.get(int(vid), "objlist"))
                    for vid in policy.selected_vehicle_ids
                },
                bandwidth_by_vehicle={int(k): float(v) for k, v in policy.bandwidth_by_vehicle.items()},
                reason=f"fixed_replay_{policy_type}",
            )
            proc = CommunicationProcess(self.comm_config, int(ego.actor_id))
            proc.set_policy(comm_policy, int(start_step))
            self._runtimes[key] = _CommunicationReplayRuntime(
                key=key,
                policy_type=policy_type,
                policy=policy,
                comm_policy=comm_policy,
                process=proc,
            )
            self._policy_order.append(key)

    def _snapshot_for_sender(
        self,
        *,
        sender_id: int,
        state: Mapping[str, object],
        comm_policy: CommPolicy,
    ) -> Optional[SenseSnapshot]:
        ego = state["ego"]
        collaborators = {int(v.actor_id): v for v in state.get("collaborators", ())}
        node = collaborators.get(int(sender_id))
        if node is None:
            return None
        objects = tuple(state.get("live_states", ()))
        observed = tuple(s for s in objects if int(sender_id) in s.visible_to_collaborators)
        modalities = tuple(comm_policy.modalities_by_vehicle.get(int(sender_id), ("objlist",)))
        data = {
            "sender_id": int(sender_id),
            "pose": {
                "x": float(node.x),
                "y": float(node.y),
                "z": float(node.z),
                "yaw": float(node.yaw),
            },
            "vel": {"vx": float(node.vx), "vy": float(node.vy)},
            "object_states": observed,
        }
        payload_size = int(self.overhead_bytes)
        for modality in modalities:
            if str(modality) == "bev":
                veh_pose = (float(node.x), float(node.y), float(node.yaw))
                data["bev"] = rasterize_bev(veh_pose, observed, route_xy=(), spec=self.bev_spec)
                payload_size += int(
                    bev_payload_bytes(
                        self.bev_spec,
                        mode=self.bev_payload_mode,
                        feature_dim=self.bev_feature_dim,
                        feature_dtype_bytes=self.bev_feature_dtype_bytes,
                    )
                )
            else:
                observed_ids = tuple(int(s.actor_id) for s in observed)
                data["objlist"] = {"observed_object_ids": observed_ids}
                payload_size += int(max(len(observed_ids), 0) * OBJECT_STATE_DIM * 4)
        return SenseSnapshot(
            sender_id=int(sender_id),
            distance_m=_node_distance(node, ego),
            payload_size=int(payload_size),
            modalities=tuple(str(v) for v in modalities),
            data=data,
        )

    def _advance_communication(self, step: int, state: Mapping[str, object]) -> None:
        for runtime in self._runtimes.values():
            proc = runtime.process
            delivered = proc.deliver(int(step))
            runtime.remember(delivered)
            if not proc.is_sensor_tick(int(step)):
                continue
            policy = runtime.comm_policy
            snapshots: Dict[int, SenseSnapshot] = {}
            for sender_id in policy.selected_collaborators:
                snapshot = self._snapshot_for_sender(sender_id=int(sender_id), state=state, comm_policy=policy)
                if snapshot is not None:
                    snapshots[int(sender_id)] = snapshot
            emitted = proc.generate(int(step), snapshots, self.link_rate_bps)
            runtime.generated_messages += len(emitted)
            runtime.remember(emitted)

    def register_step(
        self,
        step: int,
        *,
        state: Mapping[str, object],
        episode_id: int,
        fixed_dt: float,
        is_sample_step: bool,
    ) -> int:
        self._ensure_runtimes(state, start_step=0)
        self._advance_communication(int(step), state)
        if not bool(is_sample_step):
            return 0

        self._slot_window.append((int(step), dict(state)))
        window_steps = [int(item[0]) for item in self._slot_window]
        source_window = [dict(item[1]) for item in self._slot_window]
        ego = state["ego"]
        collaborators = tuple(state.get("collaborators", ()))
        objects = tuple(state.get("live_states", ()))
        candidate_ids = [int(v.actor_id) for v in collaborators]
        visible_ids = visible_object_ids_by_vehicle(int(ego.actor_id), candidate_ids, objects)

        count = 0
        for key in self._policy_order:
            runtime = self._runtimes[key]
            metadata = make_stage1_policy_metadata(
                step=int(step),
                episode_id=int(episode_id),
                policy_type=runtime.policy_type,
                policy=runtime.policy,
                candidate_vehicle_ids=candidate_ids,
                notable_object_ids=state.get("notable_ids", ()),
                visible_ids_by_vehicle=visible_ids,
                ego_pose=tuple(state["ego_pose"]),
                fixed_dt=float(fixed_dt),
            )
            metadata["replay_mode"] = "communication"
            metadata["comm_policy_id"] = int(runtime.comm_policy.policy_id)
            self._pending.append(
                (
                    int(step),
                    {
                        "key": key,
                        "source_window": source_window,
                        "window_steps": window_steps,
                        "ego_pose": tuple(state["ego_pose"]),
                        "metadata": metadata,
                    },
                )
            )
            count += 1
        return count

    def _futures_for(self, step: int) -> List[Dict[int, Point2D]]:
        return [dict(self._history.get(step + int(offset), {})) for offset in self.step_offsets]

    def _messages_for_slot(self, runtime: _CommunicationReplayRuntime, slot_step: int, prediction_step: int) -> List[object]:
        oldest = int(prediction_step) - int(self.comm_config.prediction_window_steps)
        out = []
        for message in runtime.messages.values():
            t_sense = int(getattr(message, "t_sense"))
            if t_sense != int(slot_step):
                continue
            if int(getattr(message, "t_recv")) > int(prediction_step):
                continue
            if not (oldest <= t_sense <= int(prediction_step)):
                continue
            if (
                not bool(self.comm_config.allow_cross_policy_messages)
                and int(getattr(message, "policy_id")) != int(runtime.comm_policy.policy_id)
            ):
                continue
            out.append(message)
        return out

    def _available_messages_for_prediction(
        self,
        runtime: _CommunicationReplayRuntime,
        prediction_step: int,
    ) -> List[object]:
        oldest = int(prediction_step) - int(self.comm_config.prediction_window_steps)
        out = []
        for message in runtime.messages.values():
            t_sense = int(getattr(message, "t_sense"))
            if int(getattr(message, "t_recv")) > int(prediction_step):
                continue
            if not (oldest <= t_sense <= int(prediction_step)):
                continue
            if (
                not bool(self.comm_config.allow_cross_policy_messages)
                and int(getattr(message, "policy_id")) != int(runtime.comm_policy.policy_id)
            ):
                continue
            out.append(message)
        return out

    def _received_messages_by_prediction_step(
        self,
        runtime: _CommunicationReplayRuntime,
        prediction_step: int,
    ) -> int:
        oldest = int(prediction_step) - int(self.comm_config.prediction_window_steps)
        count = 0
        for message in runtime.messages.values():
            t_sense = int(getattr(message, "t_sense"))
            if int(getattr(message, "t_recv")) > int(prediction_step):
                continue
            if not (oldest <= t_sense <= int(prediction_step)):
                continue
            if (
                not bool(self.comm_config.allow_cross_policy_messages)
                and int(getattr(message, "policy_id")) != int(runtime.comm_policy.policy_id)
            ):
                continue
            count += 1
        return count

    def _build_window(self, runtime: _CommunicationReplayRuntime, prediction_step: int, payload: Mapping[str, object]):
        graphs = []
        coverage_history = []
        for slot_step, state in zip(payload["window_steps"], payload["source_window"]):
            messages = self._messages_for_slot(runtime, int(slot_step), int(prediction_step))
            graphs.append(self.graph_builder(state, messages, int(prediction_step)))
            if self.coverage_builder is not None:
                # Graph slots intentionally require t_sense == slot_step. Coverage mirrors
                # online _build_wam_coverage(), which uses the current receive queue: any
                # message already received by prediction_step and still inside Tw can expand
                # current coverage, even if it was sensed at an earlier slot.
                coverage_messages = (
                    self._available_messages_for_prediction(runtime, int(prediction_step))
                    if int(slot_step) == int(prediction_step)
                    else messages
                )
                coverage = self.coverage_builder(state, coverage_messages, int(prediction_step))
                if coverage is not None:
                    coverage_history.append(torch.as_tensor(coverage, dtype=torch.float32))
        coverage_tensor = None
        if coverage_history and len(coverage_history) == len(graphs):
            coverage_tensor = torch.stack(coverage_history, dim=0)
        return graphs, coverage_tensor

    def _window_stats(
        self,
        runtime: _CommunicationReplayRuntime,
        prediction_step: int,
        graphs: Sequence[object],
    ) -> Dict[str, float]:
        stats = _empty_comm_replay_stats()
        if not graphs:
            return stats
        slot_count = len(graphs)
        v2v_slots = sum(1 for graph in graphs if _graph_has_v2v_vehicle(graph))
        final_sets = _graph_object_visibility_sets(graphs[-1])
        final_total = len(final_sets["all"])
        stats.update(
            {
                "comm_window_slots": float(slot_count),
                "comm_window_v2v_slots": float(v2v_slots),
                "comm_window_v2v_slot_rate": float(v2v_slots) / float(slot_count),
                "comm_window_has_v2v_graph": 1.0 if v2v_slots else 0.0,
                "comm_final_has_v2v_graph": 1.0 if _graph_has_v2v_vehicle(graphs[-1]) else 0.0,
                "comm_final_ego_visible_objects": float(len(final_sets["ego_visible"])),
                "comm_final_collab_only_objects": float(len(final_sets["collab_only"])),
                "comm_final_total_objects": float(final_total),
                "comm_final_collab_object_ratio": (
                    float(len(final_sets["collab_only"])) / float(final_total) if final_total else 0.0
                ),
                "comm_generated_messages": float(
                    sum(1 for msg in runtime.messages.values() if int(getattr(msg, "t_sense")) <= int(prediction_step))
                ),
                "comm_received_messages_by_prediction_step": float(
                    self._received_messages_by_prediction_step(runtime, int(prediction_step))
                ),
            }
        )
        return stats

    def _emit(self, step: int, payload: Dict[str, object]) -> Path:
        runtime = self._runtimes[str(payload["key"])]
        window, coverage_history = self._build_window(runtime, int(step), payload)
        # Predict/supervise the union over the window (matches WAMPerceptionModel.forward). Under real
        # V2V latency the last frame is often ego-only, so collaborator-only notable objects only appear
        # in earlier (already-received) slots -- the union keeps them instead of dropping them.
        object_node_ids = union_object_ids(window) if window else []
        target_xy, valid = build_trajectory_targets(
            object_node_ids,
            payload["ego_pose"],
            self._futures_for(step),
            ego_frame=self.ego_frame,
        )
        # t-time GT perception labels from the prediction-step (last) slot state.
        source_window = payload.get("source_window") or ()
        last_state = source_window[-1] if source_window else None
        perception_labels = None
        if last_state is not None:
            perception_labels = perception_labels_at_t(
                object_node_ids,
                last_state.get("live_states", ()),
                last_state.get("notable_ids", ()),
            )
        metadata = dict(payload.get("metadata", {}))
        metadata.update(self._window_stats(runtime, int(step), window))
        sample = make_stage1_sample(
            window,
            torch.from_numpy(target_xy),
            torch.from_numpy(valid),
            object_node_ids,
            perception_labels=perception_labels,
            metadata=metadata,
        )
        if coverage_history is not None:
            sample["coverage_history"] = coverage_history
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
        oldest_message_step = int(min_needed) - int(self.comm_config.prediction_window_steps) - 1
        for runtime in self._runtimes.values():
            for key, message in list(runtime.messages.items()):
                if int(getattr(message, "t_sense", oldest_message_step)) < oldest_message_step:
                    del runtime.messages[key]


def _median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


@torch.no_grad()
def evaluate_stage1_uncertainty_rows(
    model,
    samples: Sequence[Dict[str, object]],
    *,
    device: Union[str, torch.device] = "cpu",
    limit: Optional[int] = None,
    sigma_scale: Optional[float] = None,
    alpha: float = 0.5,
) -> List[Dict[str, object]]:
    """Run a Stage-1 model over samples and return CSV-ready uncertainty rows.

    Also emits [0, 1]-normalized columns. ``motion_uncertainty_norm_notable`` is the mean over the
    GT-notable set (``metadata['notable_object_ids']``) of ``1 - exp(-TrΣ_o / τ)``, with un-observed
    notable objects counted as 1 (blind-spot penalty); a step with no notable objects -> 0 (it is NOT
    averaged over the union -- that union fallback applies only to legacy samples lacking the
    ``notable_object_ids`` field). ``total_uncertainty_norm_notable`` =
    ``alpha * motion_norm + (1-alpha) * coverage_uncertainty`` (convex, so in [0, 1]). ``sigma_scale``
    (τ, m²) sets the saturation scale; if ``None`` it is the median observed ``TrΣ_o`` across all samples.
    """
    device = torch.device(device)
    model.to(device)
    was_training = bool(model.training)
    model.eval()
    rows: List[Dict[str, object]] = []
    all_traces: List[float] = []
    for idx, sample in enumerate(samples):
        if limit is not None and idx >= int(limit):
            break
        window = [g.to(device) for g in sample["window"]]
        out = model(window)
        trace_by_id: Dict[int, float] = {}
        if int(out["object_node_ids"].numel()) > 0:
            trace_o = per_object_trace(out["traj_log_var"])  # [Q]; TrΣ_o, no GT needed
            for i, oid in enumerate(out["object_node_ids"].tolist()):
                trace_by_id[int(oid)] = float(trace_o[i])
            all_traces.extend(trace_by_id.values())
        if int(out["object_node_ids"].numel()) == 0:
            uncertainty = 0.0
            uncertainty_notable = 0.0
            ade_fde = {"ade": 0.0, "fde": 0.0, "ade_notable": 0.0, "fde_notable": 0.0}
        else:
            tgt, val = _align_target(
                out["object_node_ids"],
                sample["object_node_ids"],
                sample["target_xy"].to(device),
                sample["valid"].to(device),
            )
            uncertainty = float(policy_uncertainty(out["notable_prob"], out["traj_log_var"], valid_mask=val))
            rec_labels = sample.get("perception_labels")
            if rec_labels is not None:
                notable = _align_labels(out["object_node_ids"], sample["object_node_ids"], rec_labels, device).get("notable")
            else:
                notable = out["labels"].get("notable")
            ade_fde = trajectory_ade_fde(out["traj_mu"], tgt, valid_mask=val, notable_weight=notable)
            # Notable-only uncertainty: weight Tr(Σ) by the (hard) GT notable label instead of the
            # model's soft notable_prob, so the comparison is restricted to the task-relevant objects.
            if notable is not None:
                uncertainty_notable = float(policy_uncertainty(notable.to(device), out["traj_log_var"], valid_mask=val))
            else:
                uncertainty_notable = float(uncertainty)
        coverage = {"coverage_uncertainty": 0.0, "route_coverage_quality_mean": 0.0, "poor_coverage_risk_mean": 0.0}
        if "coverage_history" in sample:
            coverage_tensor = sample["coverage_history"]
            if hasattr(coverage_tensor, "detach"):
                coverage_arr = coverage_tensor[-1].detach().cpu().numpy()
            else:
                coverage_arr = coverage_tensor[-1]
            coverage = coverage_metrics(coverage_arr)
        coverage_uncertainty = float(coverage.get("coverage_uncertainty", 0.0))
        total_uncertainty = float(uncertainty) + coverage_uncertainty
        total_uncertainty_notable = float(uncertainty_notable) + coverage_uncertainty
        metadata = dict(sample.get("metadata", {}))
        policy = dict(metadata.get("policy", {}))
        row = {
            "step": int(metadata.get("step", -1)),
            "episode_id": int(metadata.get("episode_id", -1)),
            "policy_type": str(metadata.get("policy_type", "")),
            "selected_vehicle_ids": list(policy.get("selected_vehicle_ids", [])),
            "modality_by_vehicle": dict(policy.get("modality_by_vehicle", {})),
            "notable_object_ids": list(metadata.get("notable_object_ids", [])),
            "uncertainty": float(uncertainty),
            "motion_uncertainty": float(uncertainty),
            "coverage_uncertainty": coverage_uncertainty,
            "total_uncertainty": float(total_uncertainty),
            # Notable-only variants: restricted to GT-notable objects (the task set), so collaborator
            # clutter in the window union does not dilute/inflate the comparison.
            "motion_uncertainty_notable": float(uncertainty_notable),
            "total_uncertainty_notable": float(total_uncertainty_notable),
            "route_coverage_quality_mean": float(coverage.get("route_coverage_quality_mean", 0.0)),
            "poor_coverage_risk_mean": float(coverage.get("poor_coverage_risk_mean", 0.0)),
            "ade": float(ade_fde["ade"]),
            "fde": float(ade_fde["fde"]),
            "ade_notable": float(ade_fde.get("ade_notable", 0.0)),
            "fde_notable": float(ade_fde.get("fde_notable", 0.0)),
        }
        for field in COMM_REPLAY_METADATA_FIELDS:
            if field in metadata:
                row[field] = metadata[field]
        row["_trace_by_id"] = trace_by_id
        # ``_has_ref`` distinguishes "notable set is known (this is policy/eval data)" from "field absent
        # (legacy data)". A known-but-empty notable set means no notable objects this step -> motion 0,
        # NOT a fallback to the union.
        row["_has_ref"] = "notable_object_ids" in metadata
        row["_ref_ids"] = [int(v) for v in metadata.get("notable_object_ids", [])]
        rows.append(row)

    # [0, 1] normalization: saturate per-object TrΣ, fixed GT-notable set with blind-spot=1, then a
    # convex combination with the (already [0, 1]) coverage term. tau defaults to the median TrΣ.
    if sigma_scale is not None:
        tau = max(float(sigma_scale), 1e-6)
    else:
        tau = max(float(_median(all_traces) or 1.0), 1e-6)
    a = float(min(max(alpha, 0.0), 1.0))
    for row in rows:
        trace_by_id = row.pop("_trace_by_id", {})
        ref_ids = row.pop("_ref_ids", [])
        has_ref = bool(row.pop("_has_ref", False))
        if has_ref:  # known GT-notable set: average over it (blind-spot=1); empty -> no notable -> 0
            u_vals = [(1.0 - math.exp(-trace_by_id[o] / tau)) if o in trace_by_id else 1.0 for o in ref_ids]
        elif trace_by_id:  # notable set unknown (legacy data) -> fall back to the observed union
            u_vals = [1.0 - math.exp(-t / tau) for t in trace_by_id.values()]
        else:
            u_vals = []
        motion_norm = float(sum(u_vals) / len(u_vals)) if u_vals else 0.0
        cov01 = min(max(float(row.get("coverage_uncertainty", 0.0)), 0.0), 1.0)
        row["motion_uncertainty_norm_notable"] = motion_norm
        row["total_uncertainty_norm_notable"] = a * motion_norm + (1.0 - a) * cov01
        row["sigma_scale"] = float(tau)

    if was_training:
        model.train()
    return rows
