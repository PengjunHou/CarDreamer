from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..emulation.features import (
    build_observable_region,
    compute_accessibility,
    compute_complementarity,
    compute_task_relevance,
)
from ..emulation.queries import make_query_records
from ..emulation.schema import (
    CanonicalEpisodeRecord,
    CanonicalStepRecord,
    CandidateVehicleState,
    EgoState,
)


RAW_SHARED_SUMMARY_DIM = 8
SEMANTIC_SHARED_SUMMARY_DIM = 8
INTENT_SUMMARY_DIM = 4


def build_runtime_emulation_step(
    *,
    scene_id: str,
    episode_id: str,
    scene_type: str,
    policy_id: str,
    predictor_step: int,
    env_step: int,
    dt: float,
    ego_pose: Mapping[str, Any],
    ego_velocity: Mapping[str, Any],
    candidate_vehicle_states: Sequence[Mapping[str, Any]],
    question_results: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    feature_size: int,
    step_communication_stats: Mapping[str, Any] | None = None,
    step_metadata: Mapping[str, Any] | None = None,
) -> CanonicalStepRecord:
    ordered_question_ids = list(question_ids)
    missing = [qid for qid in ordered_question_ids if qid not in question_results]
    if missing:
        raise ValueError(f"Missing question results for predictor step: {missing}")

    queries = make_query_records(scene_type, ordered_question_ids)
    ego_region = build_observable_region((0.0, 0.0), 0.0, range_m=16.0, width_m=9.0, lookahead_m=8.0)
    ego_state = EgoState(
        pose_xy=(float(ego_pose.get("x", 0.0)), float(ego_pose.get("y", 0.0))),
        velocity_xy=(float(ego_velocity.get("vx", 0.0)), float(ego_velocity.get("vy", 0.0))),
        yaw=float(ego_pose.get("yaw_rad", 0.0)),
        observable_region=ego_region,
    )

    candidate_vehicles: List[CandidateVehicleState] = []
    payload_types_used = set()
    payload_overridden = False
    for candidate in candidate_vehicle_states:
        vehicle_id = int(candidate["vehicle_id"])
        pose = candidate.get("pose", {})
        velocity = candidate.get("velocity", {})
        selected_infos = list(candidate.get("selected_infos", []))
        window_messages = list(candidate.get("window_messages", []))
        shared_source = str(candidate.get("shared_source", "received_feat"))
        policy_action = dict(candidate.get("policy_action", {}))
        payload_action = dict(candidate.get("payload_action", {}))
        runtime_comm_stats = dict(candidate.get("runtime_comm_stats", {}))
        payload_type = str(
            payload_action.get(
                "payload_type",
                candidate.get("payload_type", "tokens"),
            )
        ).strip() or "tokens"
        payload_encoder_id = str(
            payload_action.get(
                "payload_encoder_id",
                candidate.get("payload_encoder_id", "tokens_v1"),
            )
        ).strip() or "tokens_v1"
        payload_types_used.add(payload_type)
        payload_overridden = payload_overridden or bool(payload_action.get("payload_overridden", False))

        delta_pos = (
            float(pose.get("x", 0.0)) - float(ego_pose.get("x", 0.0)),
            float(pose.get("y", 0.0)) - float(ego_pose.get("y", 0.0)),
        )
        delta_vel = (
            float(velocity.get("vx", 0.0)) - float(ego_velocity.get("vx", 0.0)),
            float(velocity.get("vy", 0.0)) - float(ego_velocity.get("vy", 0.0)),
        )
        delta_yaw = _wrap_angle_rad(float(pose.get("yaw_rad", 0.0)) - float(ego_pose.get("yaw_rad", 0.0)))
        sender_region = build_observable_region(delta_pos, delta_yaw)
        current_distance_m = math.sqrt(delta_pos[0] * delta_pos[0] + delta_pos[1] * delta_pos[1])
        latest_msg = window_messages[-1] if window_messages else {}
        latency_s = float(latest_msg.get("latency_s", latest_msg.get("received_age_s", 0.0)))
        complementarity = compute_complementarity(sender_region, ego_region)
        accessibility = compute_accessibility(current_distance_m, latency_s)

        shared_summary_raw = _build_shared_summary_raw(
            selected_infos=selected_infos,
            window_messages=window_messages,
            feature_size=feature_size,
        )
        shared_summary_semantic = _build_shared_summary_semantic(
            sender_id=vehicle_id,
            question_results=question_results,
            question_ids=ordered_question_ids,
        )
        intent_summary = _infer_intent_summary(scene_type, delta_yaw, delta_vel)
        task_relevance = {
            query.query_id: compute_task_relevance(sender_region, ego_region, query.required_region)
            for query in queries
        }
        sender_collab = {
            query_id: float(complementarity * task_relevance[query_id] * accessibility)
            for query_id in ordered_question_ids
        }
        sender_gain = {
            query_id: _extract_sender_gain(question_results[query_id], vehicle_id)
            for query_id in ordered_question_ids
        }

        has_selected_evidence = bool(selected_infos)
        candidate_vehicles.append(
            CandidateVehicleState(
                vehicle_id=vehicle_id,
                delta_pos=delta_pos,
                delta_vel=delta_vel,
                delta_yaw=delta_yaw,
                shared_summary_raw=shared_summary_raw,
                shared_summary_semantic=shared_summary_semantic,
                intent_summary=intent_summary,
                complementarity=complementarity,
                accessibility=accessibility,
                component_valid_mask={
                    "delta_pos": True,
                    "delta_vel": True,
                    "delta_yaw": True,
                    "shared_summary_raw": has_selected_evidence,
                    "shared_summary_semantic": has_selected_evidence,
                    "intent_summary": True,
                    "complementarity": True,
                    "accessibility": True,
                    "action": True,
                },
                observable_region=sender_region,
                communication_stats={
                    "window_message_count": float(len(window_messages)),
                    "selected_message_count": float(len(selected_infos)),
                    "latest_latency_s": float(latency_s),
                    "latest_payload_bytes": float(latest_msg.get("payload_bytes", 0.0)),
                    "latest_distance_m": float(latest_msg.get("distance_m", current_distance_m)),
                    "current_distance_m": float(current_distance_m),
                    "shared_source_received_feat": 1.0 if shared_source == "received_feat" else 0.0,
                    "shared_source_raw": 1.0 if shared_source == "raw" else 0.0,
                    **{str(key): float(value) for key, value in runtime_comm_stats.items()},
                },
                query_task_relevance=task_relevance,
                sender_collab=sender_collab,
                sender_gain=sender_gain,
                metadata={
                    "env_step": int(env_step),
                    "shared_source": shared_source,
                    "has_selected_evidence": has_selected_evidence,
                    "policy_id": str(candidate.get("policy_id", policy_id)),
                    "payload_type": payload_type,
                    "payload_encoder_id": payload_encoder_id,
                    "payload_selector_reason": str(payload_action.get("payload_selector_reason", "")),
                    "payload_overridden": bool(payload_action.get("payload_overridden", False)),
                },
                alpha=float(policy_action.get("alpha", 0.0)),
                nu=float(policy_action.get("nu", 0.0)),
                bandwidth=float(policy_action.get("bandwidth", 0.0)),
                payload_type=payload_type,
                payload_encoder_id=payload_encoder_id,
            )
        )

    ego_sc = {
        query_id: _extract_ego_sc(question_results[query_id])
        for query_id in ordered_question_ids
    }
    step_metadata = dict(step_metadata or {})
    selector_reason = str(
        step_metadata.get(
            "reason",
            step_metadata.get("policy_selector_reason", ""),
        )
    )
    policy_overridden = bool(
        step_metadata.get(
            "overridden",
            step_metadata.get("policy_overridden", False),
        )
    )
    return CanonicalStepRecord(
        scene_id=str(scene_id),
        episode_id=str(episode_id),
        scene_type=str(scene_type),
        step=int(predictor_step),
        dt=float(dt),
        ego_state=ego_state,
        candidate_vehicles=candidate_vehicles,
        queries=queries,
        ego_sc=ego_sc,
        communication_stats={
            "env_step": float(env_step),
            "num_candidate_vehicles": float(len(candidate_vehicles)),
            "num_questions": float(len(ordered_question_ids)),
            **{str(key): float(value) for key, value in dict(step_communication_stats or {}).items()},
        },
        metadata={
            "env_step": int(env_step),
            "policy_id": str(policy_id),
            "policy_selector_reason": selector_reason,
            "policy_overridden": bool(policy_overridden),
            "payload_types_used": sorted(payload_types_used),
            "payload_overridden": bool(payload_overridden),
        },
        policy_id=str(policy_id),
    )


def build_runtime_emulation_episode(
    *,
    scene_id: str,
    episode_id: str,
    scene_type: str,
    dt: float,
    policy_id: str,
    steps: Sequence[CanonicalStepRecord],
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalEpisodeRecord:
    step_policy_ids = [str(step.policy_id or "").strip() for step in steps if str(step.policy_id or "").strip()]
    unique_policy_ids = sorted({policy_id for policy_id in step_policy_ids if policy_id})
    step_payload_sets = [
        tuple(sorted({str(vehicle.payload_type or "tokens") for vehicle in step.candidate_vehicles}))
        for step in steps
    ]
    unique_payload_types = sorted(
        {
            str(vehicle.payload_type or "tokens")
            for step in steps
            for vehicle in step.candidate_vehicles
        }
    )
    policy_switch_count = 0
    last_policy_id = None
    for current_policy_id in step_policy_ids:
        if last_policy_id is not None and current_policy_id != last_policy_id:
            policy_switch_count += 1
        last_policy_id = current_policy_id
    payload_switch_count = 0
    last_payload_set = None
    for payload_set in step_payload_sets:
        if last_payload_set is not None and payload_set != last_payload_set:
            payload_switch_count += 1
        last_payload_set = payload_set
    episode_policy_id = str(policy_id)
    if len(unique_policy_ids) > 1:
        episode_policy_id = "mixed"
    elif unique_policy_ids:
        episode_policy_id = unique_policy_ids[0]
    metadata = dict(metadata or {})
    metadata.update(
        {
            "policy_ids_used": unique_policy_ids,
            "payload_types_used": unique_payload_types,
            "policy_switch_count": int(policy_switch_count),
            "payload_switch_count": int(payload_switch_count),
        }
    )
    return CanonicalEpisodeRecord(
        scene_id=str(scene_id),
        episode_id=str(episode_id),
        scene_type=str(scene_type),
        dt=float(dt),
        steps=list(steps),
        metadata=metadata,
        policy_id=episode_policy_id,
    )


def _build_shared_summary_raw(
    *,
    selected_infos: Sequence[Mapping[str, Any]],
    window_messages: Sequence[Mapping[str, Any]],
    feature_size: int,
) -> List[float]:
    if not selected_infos:
        return [0.0] * RAW_SHARED_SUMMARY_DIM
    latest_msg = window_messages[-1] if window_messages else {}
    ages = [_safe_float(info.get("received_age_s", 0.0)) for info in selected_infos]
    feat_ratios = [
        _safe_float(info.get("feat_dim", 0.0)) / max(float(feature_size), 1.0)
        for info in selected_infos
    ]
    scene_fraction = _mean(
        1.0 if str(info.get("scene_description", "")).strip() else 0.0
        for info in selected_infos
    )
    text_fraction = _mean(
        1.0 if str(info.get("text", "")).strip() else 0.0
        for info in selected_infos
    )
    return [
        float(math.log1p(len(window_messages))),
        float(math.log1p(len(selected_infos))),
        _mean(ages),
        _safe_float(latest_msg.get("latency_s", latest_msg.get("received_age_s", 0.0))),
        _safe_float(latest_msg.get("payload_bytes", 0.0)) / 1024.0,
        _mean(feat_ratios),
        scene_fraction,
        text_fraction,
    ]


def _build_shared_summary_semantic(
    *,
    sender_id: int,
    question_results: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
) -> List[float]:
    per_question: List[List[float]] = []
    for query_id in question_ids:
        result = question_results.get(query_id, {})
        sender_scores = [
            sensor
            for sensor in result.get("per_sensor_scores", []) or []
            if int(sensor.get("sender_id", -1)) == int(sender_id)
            and not bool(sensor.get("is_ego", False))
        ]
        if not sender_scores:
            continue
        per_question.append(
            [
                _mean(_safe_float(sensor.get("positive_score")) for sensor in sender_scores),
                _mean(_safe_float(sensor.get("negative_score")) for sensor in sender_scores),
                _mean(_safe_float(sensor.get("unknown_score", 0.1)) for sensor in sender_scores),
                _mean(_safe_float(sensor.get("confidence")) for sensor in sender_scores),
                _mean(_safe_float(sensor.get("belief")) for sensor in sender_scores),
                _mean(_safe_float(sensor.get("evidence")) for sensor in sender_scores),
                _mean(_safe_float(sensor.get("answerability_score")) for sensor in sender_scores),
                _mean(_safe_float(sensor.get("visibility_score")) for sensor in sender_scores),
            ]
        )
    if not per_question:
        return [0.0] * SEMANTIC_SHARED_SUMMARY_DIM
    # shared_summary_semantic 的构造是：

    # 对每个 query
    # 找出这个 sender_id 对应、且 is_ego=False 的 per_sensor_scores
    # 如果一个 query 下有多个 sensor，就先对这些 sensor 的各项分数求平均
    # 再跨所有 query 求平均
    return [            ## TODO：这里对所有query做平均会不会有问题，损失了query之间的差异性
        _mean(row[index] for row in per_question)
        for index in range(SEMANTIC_SHARED_SUMMARY_DIM)
    ]


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


def _extract_sender_gain(question_result: Mapping[str, Any], sender_id: int) -> float:
    aggregated = question_result.get("aggregated_details", {})
    direct = 0.0
    for item in aggregated.get("per_sensor", []) or []:
        if int(item.get("sender_id", -1)) == int(sender_id) and not bool(item.get("is_ego", False)):
            direct += _safe_float(item.get("confidence"))
    if direct > 0.0:
        return float(direct)

    sender_weights = _coerce_sender_weights(
        question_result.get("sender_importance_positive", {}),
        question_result.get("sender_importance_negative", {}),
    )
    total_weight = sum(max(weight, 0.0) for weight in sender_weights.values())
    sender_weight = max(sender_weights.get(int(sender_id), 0.0), 0.0)
    if total_weight > 0.0 and sender_weight > 0.0:
        return float(sender_weight / total_weight * _safe_float(question_result.get("confidence_gain")))
    return 0.0


def _coerce_sender_weights(
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    for raw_key, raw_value in dict(positive).items():
        weights[int(raw_key)] = max(weights.get(int(raw_key), 0.0), _safe_float(raw_value))
    for raw_key, raw_value in dict(negative).items():
        weights[int(raw_key)] = max(weights.get(int(raw_key), 0.0), _safe_float(raw_value))
    return weights


def _extract_ego_sc(question_result: Mapping[str, Any]) -> float:
    if "confidence_with_part2" in question_result:
        return _safe_float(question_result.get("confidence_with_part2"))
    ego_plus_shared = question_result.get("ego_plus_shared", {})
    return _safe_float(ego_plus_shared.get("confidence", 0.0))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _wrap_angle_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
