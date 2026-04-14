from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .features import (
    build_observable_region,
    compute_accessibility,
    compute_complementarity,
    compute_task_relevance,
)
from .queries import make_query_records
from .schema import (
    CanonicalEpisodeRecord,
    CanonicalStepRecord,
    CandidateVehicleState,
    EgoState,
)


def generate_synthetic_canonical_episode(
    scene_type: str = "right_turn",
    scene_id: str = "synthetic_scene",
    episode_id: str = "synthetic_episode",
    num_steps: int = 20,
    num_vehicles: int = 3,
    dt: float = 0.1,
    seed: int = 0,
) -> CanonicalEpisodeRecord:
    rng = np.random.default_rng(seed)
    queries = make_query_records(scene_type)
    query_ids = [query.query_id for query in queries]
    ego_region = build_observable_region((0.0, 0.0), 0.0, range_m=16.0, width_m=9.0, lookahead_m=8.0)

    vehicle_state = []
    for index in range(num_vehicles):
        base_x = rng.uniform(6.0, 18.0)
        base_y = rng.uniform(-6.0, 6.0)
        vel_x = rng.uniform(-0.8, 0.8)
        vel_y = rng.uniform(-0.4, 0.4)
        yaw = rng.uniform(-0.6, 0.6)
        vehicle_state.append(
            {
                "vehicle_id": 1000 + index,
                "pos": np.asarray([base_x, base_y], dtype=np.float32),
                "vel": np.asarray([vel_x, vel_y], dtype=np.float32),
                "yaw": float(yaw),
                "yaw_rate": float(rng.uniform(-0.05, 0.05)),
            }
        )

    steps: List[CanonicalStepRecord] = []
    ego_pos = np.zeros((2,), dtype=np.float32)
    ego_vel = np.asarray([1.2, 0.0], dtype=np.float32)
    ego_yaw = 0.0

    for step_index in range(num_steps):
        vehicles: List[CandidateVehicleState] = []
        total_gain = {query_id: 0.0 for query_id in query_ids}

        for item in vehicle_state:
            pos = item["pos"] + step_index * dt * item["vel"]
            vel = item["vel"]
            yaw = item["yaw"] + step_index * dt * item["yaw_rate"]
            delta_pos = (float(pos[0]), float(pos[1]))
            delta_vel = (float(vel[0] - ego_vel[0]), float(vel[1] - ego_vel[1]))
            delta_yaw = float(yaw - ego_yaw)
            sender_region = build_observable_region(delta_pos, delta_yaw)
            distance_m = float(np.linalg.norm(pos))
            latency_s = float(0.02 + 0.004 * (distance_m / 10.0) + 0.001 * (item["vehicle_id"] % 3))
            complementarity = compute_complementarity(sender_region, ego_region)
            accessibility = compute_accessibility(distance_m, latency_s)
            q_conf = float(np.clip(0.85 - 0.02 * distance_m + 0.05 * rng.normal(), 0.05, 1.0))
            intent_summary = _synthetic_intent_one_hot(scene_type, delta_yaw, delta_vel)
            shared_summary_raw = [
                float(pos[0] / 20.0),
                float(pos[1] / 10.0),
                float(vel[0]),
                float(vel[1]),
                float(distance_m / 25.0),
                float(latency_s),
                float(complementarity),
                float(accessibility),
            ]
            shared_summary_semantic = [
                float(q_conf),
                float(0.5 + 0.5 * math.cos(delta_yaw)),
                float(0.5 + 0.5 * math.sin(delta_yaw)),
                float(np.clip(1.0 - distance_m / 25.0, 0.0, 1.0)),
                float(intent_summary[0]),
                float(intent_summary[1]),
                float(intent_summary[2]),
                float(intent_summary[3]),
            ]
            task_relevance: Dict[str, float] = {}
            sender_collab: Dict[str, float] = {}
            sender_gain: Dict[str, float] = {}

            for query in queries:
                n_val = compute_task_relevance(sender_region, ego_region, query.required_region)
                collab = float(complementarity * n_val * accessibility)
                gain = float(collab * (0.5 + 0.5 * q_conf))
                task_relevance[query.query_id] = n_val
                sender_collab[query.query_id] = collab
                sender_gain[query.query_id] = gain
                total_gain[query.query_id] += gain

            vehicles.append(
                CandidateVehicleState(
                    vehicle_id=int(item["vehicle_id"]),
                    delta_pos=delta_pos,
                    delta_vel=delta_vel,
                    delta_yaw=delta_yaw,
                    shared_summary_raw=shared_summary_raw,
                    shared_summary_semantic=shared_summary_semantic,
                    shared_confidence=q_conf,
                    intent_summary=intent_summary,
                    complementarity=complementarity,
                    accessibility=accessibility,
                    component_valid_mask={
                        "delta_pos": True,
                        "delta_vel": True,
                        "delta_yaw": True,
                        "shared_summary_raw": True,
                        "shared_summary_semantic": True,
                        "shared_confidence": True,
                        "intent_summary": True,
                        "complementarity": True,
                        "accessibility": True,
                    },
                    observable_region=sender_region,
                    communication_stats={"distance_m": distance_m, "latency_s": latency_s},
                    query_task_relevance=task_relevance,
                    sender_collab=sender_collab,
                    sender_gain=sender_gain,
                    metadata={"synthetic": True},
                )
            )

        ego_sc = {
            query_id: float(np.clip(0.15 + 0.65 * math.tanh(total_gain[query_id]), 0.0, 1.0))
            for query_id in query_ids
        }
        ego_state = EgoState(
            pose_xy=(float(ego_pos[0]), float(ego_pos[1])),
            velocity_xy=(float(ego_vel[0]), float(ego_vel[1])),
            yaw=float(ego_yaw),
            observable_region=ego_region,
        )
        steps.append(
            CanonicalStepRecord(
                scene_id=scene_id,
                episode_id=episode_id,
                scene_type=scene_type,
                step=step_index,
                dt=dt,
                ego_state=ego_state,
                candidate_vehicles=vehicles,
                queries=queries,
                ego_sc=ego_sc,
                communication_stats={"avg_latency_s": float(np.mean([v.communication_stats["latency_s"] for v in vehicles]))},
                metadata={"synthetic": True},
            )
        )
    return CanonicalEpisodeRecord(scene_id=scene_id, episode_id=episode_id, scene_type=scene_type, dt=dt, steps=steps, metadata={"synthetic": True})


def _synthetic_intent_one_hot(scene_type: str, delta_yaw: float, delta_vel: tuple[float, float]) -> List[float]:
    speed = math.sqrt(float(delta_vel[0]) ** 2 + float(delta_vel[1]) ** 2)
    lane_follow = 1.0
    turn_left = 0.0
    turn_right = 0.0
    stationary = 0.0
    if scene_type == "left_turn" or delta_yaw > 0.2:
        lane_follow, turn_left = 0.0, 1.0
    elif scene_type == "right_turn" or delta_yaw < -0.2:
        lane_follow, turn_right = 0.0, 1.0
    if speed < 0.2:
        lane_follow, stationary = 0.0, 1.0
    return [lane_follow, turn_left, turn_right, stationary]
