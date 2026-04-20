from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..communication.payloads import payload_type_to_one_hot
from .schema import CandidateVehicleState, CanonicalStepRecord, EgoState, QueryRecord, RegionBox


def wrap_angle_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def get_component_valid_mask_layout() -> Tuple[str, ...]:
    return (
        "delta_pos",
        "delta_vel",
        "delta_yaw",
        "shared_latent",
        "shared_summary_raw",
        "shared_summary_semantic",
        "shared_confidence",
        "intent_summary",
        "complementarity",
        "accessibility",
        "action",  # alpha, nu, bandwidth, beta_one_hot from policy u_t
    )


def get_node_feature_layout(shared_latent_dim: int) -> Dict[str, slice]:
    """Return named slices for each block in the packed node feature vector."""
    cursor = 0
    layout: Dict[str, slice] = {}
    layout["x_raw"] = slice(cursor, cursor + 5)        # delta_pos(2) + delta_vel(2) + delta_yaw(1)
    cursor += 5
    layout["shared_latent"] = slice(cursor, cursor + shared_latent_dim)
    cursor += shared_latent_dim
    layout["x_shared"] = slice(layout["shared_latent"].start, layout["shared_latent"].stop)
    layout["x_derived"] = slice(cursor, cursor + 2)    # complementarity + accessibility
    cursor += 2
    layout["action"] = slice(cursor, cursor + 8)       # alpha + nu + bandwidth + beta_one_hot[5]
    cursor += 8
    return layout


def get_vehicle_exogenous_feature_keys() -> Tuple[str, ...]:
    return (
        "window_message_count",
        "selected_message_count",
        "latest_latency_s",
        "latest_payload_bytes",
        "current_distance_m",
        "shared_source_received_feat",
        "shared_source_raw",
    )


def get_step_exogenous_feature_keys() -> Tuple[str, ...]:
    return (
        "num_candidate_vehicles",
        "avg_latency_s",
    )


def build_observable_region(
    delta_pos: Sequence[float],
    delta_yaw: float,
    range_m: float = 18.0,
    width_m: float = 10.0,
    lookahead_m: float = 7.0,
) -> RegionBox:
    dx, dy = float(delta_pos[0]), float(delta_pos[1])
    forward = (math.cos(float(delta_yaw)), math.sin(float(delta_yaw)))
    center = (dx + lookahead_m * forward[0], dy + lookahead_m * forward[1])
    return RegionBox(center=center, size=(range_m, width_m), yaw=float(delta_yaw))


def pack_vehicle_node_state(vehicle: CandidateVehicleState) -> np.ndarray:
    """Pack all per-vehicle features into a flat float32 vector.

    Layout: [x_raw(5) | shared_latent | x_derived(2) | action(8)]

    The action block (alpha, nu, bandwidth, beta_one_hot) encodes the policy decision u_t
    applied to this vehicle, enabling policy-conditioned dynamics learning.
    """
    beta_one_hot = payload_type_to_one_hot(getattr(vehicle, "payload_type", "tokens"))
    blocks = [
        np.asarray(
            [
                float(vehicle.delta_pos[0]),
                float(vehicle.delta_pos[1]),
                float(vehicle.delta_vel[0]),
                float(vehicle.delta_vel[1]),
                float(vehicle.delta_yaw),
            ],
            dtype=np.float32,
        ),
        np.asarray(vehicle.shared_latent, dtype=np.float32).reshape(-1),
        np.asarray([float(vehicle.complementarity), float(vehicle.accessibility)], dtype=np.float32),
        np.concatenate(
            [
                np.asarray([float(vehicle.alpha), float(vehicle.nu), float(vehicle.bandwidth)], dtype=np.float32),
                beta_one_hot,
            ],
            axis=0,
        ),
    ]
    return np.concatenate(blocks, axis=0)


def pack_vehicle_state_features(vehicle: CandidateVehicleState) -> np.ndarray:
    """Pack the per-vehicle state x_i,t without action variables."""
    blocks = [
        np.asarray(
            [
                float(vehicle.delta_pos[0]),
                float(vehicle.delta_pos[1]),
                float(vehicle.delta_vel[0]),
                float(vehicle.delta_vel[1]),
                float(vehicle.delta_yaw),
            ],
            dtype=np.float32,
        ),
        np.asarray(vehicle.shared_latent, dtype=np.float32).reshape(-1),
        np.asarray([float(vehicle.complementarity), float(vehicle.accessibility)], dtype=np.float32),
    ]
    return np.concatenate(blocks, axis=0)


def pack_vehicle_action_features(vehicle: CandidateVehicleState) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray([float(vehicle.alpha), float(vehicle.nu), float(vehicle.bandwidth)], dtype=np.float32),
            payload_type_to_one_hot(getattr(vehicle, "payload_type", "tokens")),
        ],
        axis=0,
    )


def pack_vehicle_raw_state_target(vehicle: CandidateVehicleState) -> np.ndarray:
    return np.asarray(
        [
            float(vehicle.delta_pos[0]),
            float(vehicle.delta_pos[1]),
            float(vehicle.delta_vel[0]),
            float(vehicle.delta_vel[1]),
            float(vehicle.delta_yaw),
        ],
        dtype=np.float32,
    )


def pack_vehicle_shared_state_target(vehicle: CandidateVehicleState) -> np.ndarray:
    if not vehicle.shared_latent:
        raise ValueError(
            "Episode contains compact-summary-only shared state. "
            "Strict shared-latent mode requires per-vehicle shared_latent."
        )
    return np.asarray(vehicle.shared_latent, dtype=np.float32).reshape(-1)


def pack_vehicle_exogenous_features(vehicle: CandidateVehicleState) -> np.ndarray:
    stats = vehicle.communication_stats or {}
    return np.asarray(
        [float(stats.get(key, 0.0)) for key in get_vehicle_exogenous_feature_keys()],
        dtype=np.float32,
    )


def pack_ego_state_features(ego_state: EgoState) -> np.ndarray:
    return np.asarray(
        [
            float(ego_state.pose_xy[0]),
            float(ego_state.pose_xy[1]),
            float(ego_state.velocity_xy[0]),
            float(ego_state.velocity_xy[1]),
            float(ego_state.yaw),
        ],
        dtype=np.float32,
    )


def pack_step_exogenous_features(step: CanonicalStepRecord) -> np.ndarray:
    stats = step.communication_stats or {}
    return np.asarray(
        [float(stats.get(key, 0.0)) for key in get_step_exogenous_feature_keys()],
        dtype=np.float32,
    )


def pack_component_valid_mask(vehicle: CandidateVehicleState) -> np.ndarray:
    layout = get_component_valid_mask_layout()
    mask = []
    for key in layout:
        if key == "action":
            # action is always considered valid (0.0 = not selected is still a valid signal)
            mask.append(1.0)
        else:
            mask.append(1.0 if bool(vehicle.component_valid_mask.get(key, False)) else 0.0)
    return np.asarray(mask, dtype=np.float32)


def pack_query_features(query: QueryRecord) -> np.ndarray:
    region = query.required_region
    base = np.asarray(query.query_embedding_input, dtype=np.float32).reshape(-1)
    region_features = np.asarray(
        [
            float(region.center[0]),
            float(region.center[1]),
            float(region.size[0]),
            float(region.size[1]),
            math.cos(float(region.yaw)),
            math.sin(float(region.yaw)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, region_features], axis=0)


def compute_accessibility(distance_m: float, latency_s: float, lambda_d: float = 0.03, lambda_tau: float = 1.25) -> float:
    return float(math.exp(-float(lambda_d) * max(float(distance_m), 0.0) - float(lambda_tau) * max(float(latency_s), 0.0)))


def compute_complementarity(sender_region: RegionBox, ego_region: RegionBox, eps: float = 1e-6, resolution: int = 9) -> float:
    sender_points = sample_region_points(sender_region, resolution=resolution)
    if sender_points.size == 0:
        return 0.0
    sender_inside_ego = _points_in_region(sender_points, ego_region)
    additional_fraction = float((~sender_inside_ego).mean())
    return additional_fraction / (1.0 + float(eps))


def compute_task_relevance(sender_region: RegionBox, ego_region: RegionBox, required_region: RegionBox, eps: float = 1e-6, resolution: int = 9) -> float:
    sender_points = sample_region_points(sender_region, resolution=resolution)
    if sender_points.size == 0:
        return 0.0
    sender_additional = sender_points[~_points_in_region(sender_points, ego_region)]
    if sender_additional.size == 0:
        return 0.0
    required_points = sample_region_points(required_region, resolution=resolution)
    if required_points.size == 0:
        return 0.0
    hits = _points_in_region(sender_additional, required_region)
    return float(hits.mean()) / (1.0 + float(eps))


def sample_region_points(region: RegionBox, resolution: int = 9) -> np.ndarray:
    half_w = 0.5 * float(region.size[0])
    half_h = 0.5 * float(region.size[1])
    if half_w <= 0 or half_h <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    xs = np.linspace(-half_w, half_w, num=max(int(resolution), 2), dtype=np.float32)
    ys = np.linspace(-half_h, half_h, num=max(int(resolution), 2), dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    c = math.cos(float(region.yaw))
    s = math.sin(float(region.yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    pts = grid @ rot.T
    pts[:, 0] += float(region.center[0])
    pts[:, 1] += float(region.center[1])
    return pts


def build_pairwise_edge_attr(node_positions: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    attrs: List[np.ndarray] = []
    for src, dst in edge_index.T:
        dx = float(node_positions[dst, 0] - node_positions[src, 0])
        dy = float(node_positions[dst, 1] - node_positions[src, 1])
        dist = float(math.sqrt(dx * dx + dy * dy))
        attrs.append(np.asarray([dx, dy, dist], dtype=np.float32))
    return np.stack(attrs, axis=0)


def _points_in_region(points: np.ndarray, region: RegionBox) -> np.ndarray:
    shifted = np.asarray(points, dtype=np.float32).copy()
    shifted[:, 0] -= float(region.center[0])
    shifted[:, 1] -= float(region.center[1])
    c = math.cos(-float(region.yaw))
    s = math.sin(-float(region.yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    local = shifted @ rot.T
    half_w = 0.5 * float(region.size[0]) + 1e-6
    half_h = 0.5 * float(region.size[1]) + 1e-6
    return (np.abs(local[:, 0]) <= half_w) & (np.abs(local[:, 1]) <= half_h)
