from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .features import (
    build_observable_region,
    compute_accessibility,
    compute_complementarity,
    compute_task_relevance,
    wrap_angle_rad,
)
from .queries import make_query_records
from .schema import (
    CanonicalEpisodeRecord,
    CanonicalStepRecord,
    CandidateVehicleState,
    EgoState,
    validate_episode_record,
)


DEFAULT_QUERY_ORDER: Tuple[str, ...] = (
    "clg_left_rear_vehicle",
    "clg_right_rear_vehicle",
    "clg_right_front_vehicle",
    "clg_left_front_vehicle",
    "clg_front_vehicle",
    "clg_rear_vehicle",
)


def adapt_vlm_records_to_canonical_episode(
    source: str | Path | Sequence[Mapping[str, Any]],
    *,
    scene_type: str = "right_turn",
    scene_id: str | None = None,
    episode_id: str | None = None,
    dt: float = 0.1,
    query_ids: Sequence[str] | None = None,
) -> CanonicalEpisodeRecord:
    records = _load_records(source)
    if not records:
        raise ValueError("VLM records source is empty.")

    inferred_episode_id, inferred_scene_id = _infer_ids(source)
    ordered_query_ids = list(query_ids or _infer_query_order(records))
    query_records = make_query_records(scene_type, ordered_query_ids)

    step_records: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        step = _safe_int(record.get("step"), default=-1)
        question_id = str(record.get("question_id", ""))
        if step < 0 or question_id not in ordered_query_ids:
            continue
        step_records[step][question_id] = dict(record)
    if not step_records:
        raise ValueError("No usable step/question records were found in the VLM log.")

    sensor_tracks = _build_sensor_tracks(step_records)
    sorted_steps = sorted(step_records.keys())
    episode_steps: List[CanonicalStepRecord] = []

    for step in sorted_steps:
        records_by_query = step_records[step]
        ego_pose = sensor_tracks["ego_pose"].get(step)
        if ego_pose is None:
            ego_pose = _fallback_pose(sensor_tracks["ego_pose"], step)
        if ego_pose is None:
            raise ValueError(f"Could not infer ego pose for step {step}.")

        ego_velocity = _estimate_velocity(sensor_tracks["ego_pose"], step, dt)
        ego_region = build_observable_region((0.0, 0.0), 0.0, range_m=24.0, width_m=12.0, lookahead_m=12.0)
        ego_state = EgoState(
            pose_xy=(float(ego_pose["x"]), float(ego_pose["y"])),
            velocity_xy=ego_velocity,
            yaw=float(ego_pose["yaw_rad"]),
            observable_region=ego_region,
        )

        step_sender_data = sensor_tracks["senders"].get(step, {})
        vehicles: List[CandidateVehicleState] = []
        for sender_id in sorted(step_sender_data.keys()):
            sender_pose = step_sender_data[sender_id]["pose"]
            sender_velocity = _estimate_velocity(sensor_tracks["sender_pose"][sender_id], step, dt)
            delta_pos = (
                float(sender_pose["x"] - ego_pose["x"]),
                float(sender_pose["y"] - ego_pose["y"]),
            )
            delta_vel = (
                float(sender_velocity[0] - ego_velocity[0]),
                float(sender_velocity[1] - ego_velocity[1]),
            )
            delta_yaw = wrap_angle_rad(float(sender_pose["yaw_rad"] - ego_pose["yaw_rad"]))
            sender_region = build_observable_region(delta_pos, delta_yaw)

            latency_s = _safe_mean(
                [obs.get("latency") for obs in step_sender_data[sender_id]["observations"]],
                default=0.0,
            )
            distance_m = math.sqrt(delta_pos[0] * delta_pos[0] + delta_pos[1] * delta_pos[1])
            complementarity = compute_complementarity(sender_region, ego_region)
            accessibility = compute_accessibility(distance_m, latency_s)

            raw_summary, semantic_summary = _summarize_sender_observations(
                step_sender_data[sender_id]["per_query"],
            )
            intent_summary = _infer_intent_summary(scene_type, delta_yaw, delta_vel)

            task_relevance = {
                query.query_id: compute_task_relevance(sender_region, ego_region, query.required_region)
                for query in query_records
            }
            sender_collab = {
                query_id: float(complementarity * task_relevance[query_id] * accessibility)
                for query_id in ordered_query_ids
            }
            sender_gain = {
                query_id: _extract_sender_gain(records_by_query.get(query_id, {}), sender_id)
                for query_id in ordered_query_ids
            }

            vehicles.append(
                CandidateVehicleState(
                    vehicle_id=int(sender_id),
                    delta_pos=delta_pos,
                    delta_vel=delta_vel,
                    delta_yaw=delta_yaw,
                    shared_summary_raw=raw_summary,
                    shared_summary_semantic=semantic_summary,
                    intent_summary=intent_summary,
                    complementarity=complementarity,
                    accessibility=accessibility,
                    component_valid_mask={
                        "delta_pos": True,
                        "delta_vel": True,
                        "delta_yaw": True,
                        "shared_summary_raw": bool(raw_summary),
                        "shared_summary_semantic": bool(semantic_summary),
                        "intent_summary": True,
                        "complementarity": True,
                        "accessibility": True,
                        "action": False,
                    },
                    observable_region=sender_region,
                    communication_stats={
                        "distance_m": float(distance_m),
                        "latency_s": float(latency_s),
                        "num_observations": float(len(step_sender_data[sender_id]["observations"])),
                    },
                    query_task_relevance=task_relevance,
                    sender_collab=sender_collab,
                    sender_gain=sender_gain,
                    metadata={
                        "source": "vlm_records_adapter",
                        "sender_ids_seen_this_step": sorted(int(k) for k in step_sender_data.keys()),
                    },
                )
            )

        ego_sc = {
            query_id: _extract_ego_sc(records_by_query.get(query_id, {}))
            for query_id in ordered_query_ids
        }
        episode_steps.append(
            CanonicalStepRecord(
                scene_id=scene_id or inferred_scene_id,
                episode_id=episode_id or inferred_episode_id,
                scene_type=scene_type,
                step=int(step),
                dt=float(dt),
                ego_state=ego_state,
                candidate_vehicles=vehicles,
                queries=query_records,
                ego_sc=ego_sc,
                communication_stats={
                    "num_candidate_vehicles": float(len(vehicles)),
                    "num_questions": float(len(ordered_query_ids)),
                },
                metadata={
                    "source": str(source) if isinstance(source, (str, Path)) else "in_memory_records",
                    "adapter": "adapt_vlm_records_to_canonical_episode",
                },
            )
        )

    episode = CanonicalEpisodeRecord(
        scene_id=scene_id or inferred_scene_id,
        episode_id=episode_id or inferred_episode_id,
        scene_type=scene_type,
        dt=float(dt),
        steps=episode_steps,
        metadata={
            "source": str(source) if isinstance(source, (str, Path)) else "in_memory_records",
            "adapter": "vlm_records",
        },
    )
    validate_episode_record(episode)
    return episode


def _load_records(source: str | Path | Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = list(source)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return [dict(item) for item in payload["records"]]
        if isinstance(payload.get("steps"), list):
            return [dict(item) for item in payload["steps"]]
    raise TypeError("Unsupported VLM records payload.")


def _infer_ids(source: str | Path | Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    if isinstance(source, (str, Path)):
        stem = Path(source).stem
        return stem, f"{stem}_scene"
    return "vlm_episode", "vlm_scene"


def _infer_query_order(records: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = {str(record.get("question_id", "")) for record in records if record.get("question_id")}
    ordered = [query_id for query_id in DEFAULT_QUERY_ORDER if query_id in seen]
    return ordered + sorted(seen - set(ordered))


def _build_sensor_tracks(
    step_records: Mapping[int, Mapping[str, Mapping[str, Any]]]
) -> Dict[str, Any]:
    ego_pose: Dict[int, Dict[str, float]] = {}
    senders: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    sender_pose: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(dict)

    for step, records_by_query in step_records.items():
        ego_observations: List[Dict[str, Any]] = []
        per_sender_obs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        per_sender_query: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        for query_id, record in records_by_query.items():
            for sensor in record.get("per_sensor_scores", []) or []:
                obs = dict(sensor)
                pose = obs.get("pose") or {}
                if "yaw_rad" not in pose:
                    pose["yaw_rad"] = math.radians(_safe_float(pose.get("yaw"), default=0.0))
                obs["pose"] = pose
                if bool(obs.get("is_ego", False)):
                    ego_observations.append(obs)
                    continue
                sender_id = _safe_int(obs.get("sender_id"), default=-1)
                if sender_id < 0:
                    continue
                per_sender_obs[sender_id].append(obs)
                per_sender_query[sender_id][str(query_id)] = obs

        if ego_observations:
            ego_pose[step] = _average_pose(ego_observations)

        for sender_id, observations in per_sender_obs.items():
            pose_summary = _average_pose(observations)
            senders[step][sender_id] = {
                "pose": pose_summary,
                "observations": observations,
                "per_query": per_sender_query[sender_id],
            }
            sender_pose[sender_id][step] = pose_summary

    return {
        "ego_pose": ego_pose,
        "senders": senders,
        "sender_pose": sender_pose,
    }


def _average_pose(observations: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    xs = [_safe_float((obs.get("pose") or {}).get("x")) for obs in observations]
    ys = [_safe_float((obs.get("pose") or {}).get("y")) for obs in observations]
    yaws = [_safe_float((obs.get("pose") or {}).get("yaw_rad"), default=math.radians(_safe_float((obs.get("pose") or {}).get("yaw")))) for obs in observations]
    mean_cos = _safe_mean([math.cos(yaw) for yaw in yaws], default=1.0)
    mean_sin = _safe_mean([math.sin(yaw) for yaw in yaws], default=0.0)
    return {
        "x": _safe_mean(xs, default=0.0),
        "y": _safe_mean(ys, default=0.0),
        "yaw_rad": math.atan2(mean_sin, mean_cos),
    }


def _fallback_pose(poses_by_step: Mapping[int, Mapping[str, float]], step: int) -> Dict[str, float] | None:
    if step in poses_by_step:
        return dict(poses_by_step[step])
    previous = [idx for idx in poses_by_step.keys() if idx < step]
    if previous:
        return dict(poses_by_step[max(previous)])
    following = [idx for idx in poses_by_step.keys() if idx > step]
    if following:
        return dict(poses_by_step[min(following)])
    return None


def _estimate_velocity(
    poses_by_step: Mapping[int, Mapping[str, float]],
    step: int,
    dt: float,
) -> Tuple[float, float]:
    current = _fallback_pose(poses_by_step, step)
    if current is None:
        return (0.0, 0.0)
    prev_pose = _fallback_pose({idx: pose for idx, pose in poses_by_step.items() if idx < step}, step - 1)
    next_pose = _fallback_pose({idx: pose for idx, pose in poses_by_step.items() if idx > step}, step + 1)
    if prev_pose is not None and next_pose is not None:
        denom = max(2.0 * float(dt), 1e-6)
        return (
            float(next_pose["x"] - prev_pose["x"]) / denom,
            float(next_pose["y"] - prev_pose["y"]) / denom,
        )
    if prev_pose is not None:
        denom = max(float(dt), 1e-6)
        return (
            float(current["x"] - prev_pose["x"]) / denom,
            float(current["y"] - prev_pose["y"]) / denom,
        )
    if next_pose is not None:
        denom = max(float(dt), 1e-6)
        return (
            float(next_pose["x"] - current["x"]) / denom,
            float(next_pose["y"] - current["y"]) / denom,
        )
    return (0.0, 0.0)


def _summarize_sender_observations(
    per_query_observations: Mapping[str, Mapping[str, Any]]
) -> Tuple[List[float], List[float]]:
    observations = list(per_query_observations.values())
    visibility_scores = [_safe_float(obs.get("visibility_score")) for obs in observations]
    answerability_scores = [_safe_float(obs.get("answerability_score")) for obs in observations]
    latencies = [_safe_float(obs.get("latency")) for obs in observations]
    information_terms = [_safe_float(obs.get("information_term")) for obs in observations]
    positive_scores = [_safe_float(obs.get("positive_score")) for obs in observations]
    negative_scores = [_safe_float(obs.get("negative_score")) for obs in observations]
    unknown_scores = [_safe_float(obs.get("unknown_score"), default=0.1) for obs in observations]
    beliefs = [_safe_float(obs.get("belief")) for obs in observations]
    evidences = [_safe_float(obs.get("evidence")) for obs in observations]
    confidences = [_safe_float(obs.get("confidence")) for obs in observations]

    raw_summary = [
        _safe_mean(visibility_scores, default=0.0),
        _safe_mean(answerability_scores, default=0.0),
        _safe_mean(latencies, default=0.0),
        _safe_mean(information_terms, default=0.0),
        _safe_mean([1.0 if str(obs.get("question_answerability", "")).lower() == "answerable" else 0.0 for obs in observations], default=0.0),
        _safe_mean([1.0 if str(obs.get("visibility_status", "")).lower() == "visible" else 0.0 for obs in observations], default=0.0),
        float(len(observations)),
        _safe_mean([_safe_float(obs.get("num_images")) for obs in observations], default=0.0),
    ]
    semantic_summary = [
        _safe_mean(positive_scores, default=0.0),
        _safe_mean(negative_scores, default=0.0),
        _safe_mean(unknown_scores, default=0.0),
        _safe_mean(confidences, default=0.0),
        _safe_mean(beliefs, default=0.0),
        _safe_mean(evidences, default=0.0),
        float(max(positive_scores, default=0.0)),
        float(max(negative_scores, default=0.0)),
    ]
    return raw_summary, semantic_summary


def _infer_intent_summary(
    scene_type: str,
    delta_yaw: float,
    delta_vel: Sequence[float],
) -> List[float]:
    speed = math.sqrt(float(delta_vel[0]) ** 2 + float(delta_vel[1]) ** 2)
    lane_follow = 1.0
    turn_left = 0.0
    turn_right = 0.0
    stationary = 0.0
    if scene_type == "left_turn" or delta_yaw > 0.2:
        lane_follow = 0.0
        turn_left = 1.0
    elif scene_type == "right_turn" or delta_yaw < -0.2:
        lane_follow = 0.0
        turn_right = 1.0
    if speed < 0.15:
        lane_follow = 0.0
        stationary = 1.0
    return [lane_follow, turn_left, turn_right, stationary]


def _extract_sender_gain(record: Mapping[str, Any], sender_id: int) -> float:
    aggregated = record.get("aggregated_details", {}) if isinstance(record, Mapping) else {}
    per_sensor = aggregated.get("per_sensor", []) if isinstance(aggregated, Mapping) else []
    contribution = 0.0
    for item in per_sensor or []:
        if _safe_int(item.get("sender_id"), default=-1) == int(sender_id):
            contribution += _safe_float(item.get("confidence"))
    if contribution > 0.0:
        return float(contribution)

    sender_weights: Dict[int, float] = {}
    for sensor in record.get("per_sensor_scores", []) or []:
        if bool(sensor.get("is_ego", False)):
            continue
        sid = _safe_int(sensor.get("sender_id"), default=-1)
        if sid < 0:
            continue
        sender_weights[sid] = sender_weights.get(sid, 0.0) + _safe_float(
            sensor.get("importance_weight"),
            default=max(
                _safe_float(sensor.get("importance_positive")),
                _safe_float(sensor.get("importance_negative")),
            ),
        )

    total_weight = sum(max(weight, 0.0) for weight in sender_weights.values())
    if total_weight > 0.0 and int(sender_id) in sender_weights:
        gain_total = max(
            _safe_float(record.get("confidence_gain")),
            _extract_ego_sc(record) - _safe_float((record.get("ego_only") or {}).get("confidence")),
        )
        return float(max(sender_weights[int(sender_id)], 0.0) / total_weight * gain_total)
    return 0.0


def _extract_ego_sc(record: Mapping[str, Any]) -> float:
    if not isinstance(record, Mapping):
        return 0.0
    if "confidence_with_part2" in record:
        return _safe_float(record.get("confidence_with_part2"))
    ego_plus_shared = record.get("ego_plus_shared", {})
    if isinstance(ego_plus_shared, Mapping):
        return _safe_float(ego_plus_shared.get("confidence"))
    return 0.0


def _safe_mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return float(default)
    return float(sum(vals) / len(vals))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
