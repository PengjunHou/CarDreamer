from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class RegionBox:
    center: Tuple[float, float]
    size: Tuple[float, float]
    yaw: float = 0.0


@dataclass
class EgoState:
    pose_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    yaw: float
    observable_region: RegionBox


@dataclass
class QueryRecord:
    query_id: str
    query_embedding_input: List[float]
    required_region: RegionBox
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateVehicleState:
    vehicle_id: int
    delta_pos: Tuple[float, float]
    delta_vel: Tuple[float, float]
    delta_yaw: float
    shared_summary_raw: List[float]
    shared_summary_semantic: List[float]
    intent_summary: List[float]
    complementarity: float
    accessibility: float
    component_valid_mask: Dict[str, bool]
    observable_region: RegionBox
    communication_stats: Dict[str, float] = field(default_factory=dict)
    query_task_relevance: Dict[str, float] = field(default_factory=dict)
    sender_collab: Dict[str, float] = field(default_factory=dict)
    sender_gain: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Policy action variables (u_t per vehicle, as defined in Section IV.B)
    alpha: float = 0.0       # collaboration selection indicator (0 or 1)
    nu: float = 0.0          # sharing frequency: 0.2=low, 0.5=mid, 1.0=high
    bandwidth: float = 0.0   # allocated bandwidth (normalized, 0..1)
    payload_type: str = "tokens"
    payload_encoder_id: str = "tokens_v1"


@dataclass
class CanonicalStepRecord:
    scene_id: str
    episode_id: str
    scene_type: str
    step: int
    dt: float
    ego_state: EgoState
    candidate_vehicles: List[CandidateVehicleState]
    queries: List[QueryRecord]
    ego_sc: Dict[str, float]
    communication_stats: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    policy_id: str = ""  # which fixed policy produced this step (e.g. "P1".."P8")


@dataclass
class CanonicalEpisodeRecord:
    scene_id: str
    episode_id: str
    scene_type: str
    dt: float
    steps: List[CanonicalStepRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)
    policy_id: str = ""  # policy used for the entire episode


def episode_to_dict(episode: CanonicalEpisodeRecord) -> Dict[str, Any]:
    return asdict(episode)


def episode_from_dict(payload: Dict[str, Any]) -> CanonicalEpisodeRecord:
    steps = []
    for raw_step in payload.get("steps", []):
        ego = raw_step["ego_state"]
        ego_state = EgoState(
            pose_xy=tuple(ego["pose_xy"]),
            velocity_xy=tuple(ego["velocity_xy"]),
            yaw=float(ego["yaw"]),
            observable_region=RegionBox(
                center=tuple(ego["observable_region"]["center"]),
                size=tuple(ego["observable_region"]["size"]),
                yaw=float(ego["observable_region"].get("yaw", 0.0)),
            ),
        )
        queries = [
            QueryRecord(
                query_id=str(item["query_id"]),
                query_embedding_input=[float(x) for x in item["query_embedding_input"]],
                required_region=RegionBox(
                    center=tuple(item["required_region"]["center"]),
                    size=tuple(item["required_region"]["size"]),
                    yaw=float(item["required_region"].get("yaw", 0.0)),
                ),
                metadata=dict(item.get("metadata", {})),
            )
            for item in raw_step.get("queries", [])
        ]
        vehicles = [
            CandidateVehicleState(
                vehicle_id=int(item["vehicle_id"]),
                delta_pos=tuple(item["delta_pos"]),
                delta_vel=tuple(item["delta_vel"]),
                delta_yaw=float(item["delta_yaw"]),
                shared_summary_raw=[float(x) for x in item.get("shared_summary_raw", [])],
                shared_summary_semantic=[float(x) for x in item.get("shared_summary_semantic", [])],
                intent_summary=[float(x) for x in item.get("intent_summary", [])],
                complementarity=float(item.get("complementarity", 0.0)),
                accessibility=float(item.get("accessibility", 0.0)),
                component_valid_mask={str(k): bool(v) for k, v in item.get("component_valid_mask", {}).items()},
                observable_region=RegionBox(
                    center=tuple(item["observable_region"]["center"]),
                    size=tuple(item["observable_region"]["size"]),
                    yaw=float(item["observable_region"].get("yaw", 0.0)),
                ),
                communication_stats={str(k): float(v) for k, v in item.get("communication_stats", {}).items()},
                query_task_relevance={str(k): float(v) for k, v in item.get("query_task_relevance", {}).items()},
                sender_collab={str(k): float(v) for k, v in item.get("sender_collab", {}).items()},
                sender_gain={str(k): float(v) for k, v in item.get("sender_gain", {}).items()},
                metadata=dict(item.get("metadata", {})),
                alpha=float(item.get("alpha", 0.0)),
                nu=float(item.get("nu", 0.0)),
                bandwidth=float(item.get("bandwidth", 0.0)),
                payload_type=str(item.get("payload_type", "tokens") or "tokens"),
                payload_encoder_id=str(item.get("payload_encoder_id", "tokens_v1") or "tokens_v1"),
            )
            for item in raw_step.get("candidate_vehicles", [])
        ]
        steps.append(
            CanonicalStepRecord(
                scene_id=str(raw_step["scene_id"]),
                episode_id=str(raw_step["episode_id"]),
                scene_type=str(raw_step["scene_type"]),
                step=int(raw_step["step"]),
                dt=float(raw_step["dt"]),
                ego_state=ego_state,
                candidate_vehicles=vehicles,
                queries=queries,
                ego_sc={str(k): float(v) for k, v in raw_step.get("ego_sc", {}).items()},
                communication_stats={str(k): float(v) for k, v in raw_step.get("communication_stats", {}).items()},
                metadata=dict(raw_step.get("metadata", {})),
                policy_id=str(raw_step.get("policy_id", "")),
            )
        )
    return CanonicalEpisodeRecord(
        scene_id=str(payload["scene_id"]),
        episode_id=str(payload["episode_id"]),
        scene_type=str(payload["scene_type"]),
        dt=float(payload["dt"]),
        steps=steps,
        metadata=dict(payload.get("metadata", {})),
        policy_id=str(payload.get("policy_id", "")),
    )


def validate_episode_record(episode: CanonicalEpisodeRecord) -> None:
    if not episode.steps:
        raise ValueError("Canonical episode must contain at least one step.")
    expected_queries = [query.query_id for query in episode.steps[0].queries]
    expected_dt = float(episode.dt)

    for index, step in enumerate(episode.steps):
        if int(step.step) != index:
            raise ValueError(f"Canonical steps must be contiguous from zero, got step={step.step} at index={index}.")
        if abs(float(step.dt) - expected_dt) > 1e-6:
            raise ValueError("All steps in an episode must share the same dt.")
        query_ids = [query.query_id for query in step.queries]
        if query_ids != expected_queries:
            raise ValueError("All steps in an episode must share the same ordered query list.")
        for vehicle in step.candidate_vehicles:
            if len(vehicle.delta_pos) != 2 or len(vehicle.delta_vel) != 2:
                raise ValueError("delta_pos and delta_vel must be 2D.")
            if not vehicle.shared_summary_raw:
                raise ValueError("shared_summary_raw must be non-empty.")
            if not vehicle.shared_summary_semantic:
                raise ValueError("shared_summary_semantic must be non-empty.")
            if not vehicle.intent_summary:
                raise ValueError("intent_summary must be non-empty.")
            if not str(vehicle.payload_type or "").strip():
                raise ValueError("payload_type must be non-empty.")
            if not str(vehicle.payload_encoder_id or "").strip():
                raise ValueError("payload_encoder_id must be non-empty.")
            missing_queries = set(expected_queries) - set(vehicle.query_task_relevance.keys())
            if missing_queries:
                raise ValueError(f"Vehicle {vehicle.vehicle_id} missing task relevance for queries: {sorted(missing_queries)}")
            missing_collab = set(expected_queries) - set(vehicle.sender_collab.keys())
            if missing_collab:
                raise ValueError(f"Vehicle {vehicle.vehicle_id} missing sender_collab for queries: {sorted(missing_collab)}")
            missing_gain = set(expected_queries) - set(vehicle.sender_gain.keys())
            if missing_gain:
                raise ValueError(f"Vehicle {vehicle.vehicle_id} missing sender_gain for queries: {sorted(missing_gain)}")
        missing_ego_sc = set(expected_queries) - set(step.ego_sc.keys())
        if missing_ego_sc:
            raise ValueError(f"Step {step.step} missing ego_sc for queries: {sorted(missing_ego_sc)}")
