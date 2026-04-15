from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_QUESTION_ORDER: Tuple[str, ...] = (
    "clg_left_rear_vehicle",
    "clg_right_rear_vehicle",
    "clg_right_front_vehicle",
    "clg_left_front_vehicle",
    "clg_front_vehicle",
    "clg_rear_vehicle",
)


def load_vlm_records(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_question_order(records: Sequence[Dict[str, Any]]) -> List[str]:
    seen = {str(record.get("question_id", "")) for record in records if record.get("question_id")}
    ordered = [qid for qid in DEFAULT_QUESTION_ORDER if qid in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def infer_sender_order(records: Sequence[Dict[str, Any]]) -> List[int]:
    sender_ids = set()
    for record in records:
        for sensor in record.get("per_sensor_scores", []) or []:
            if bool(sensor.get("is_ego", False)):
                continue
            sender_id = _safe_int(sensor.get("sender_id"), default=-1)
            if sender_id >= 0:
                sender_ids.add(sender_id)
    return sorted(sender_ids)


def build_emulation_step_summaries(
    records: Sequence[Dict[str, Any]],
    question_order: Optional[Sequence[str]] = None,
    sender_order: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    question_order = list(question_order or infer_question_order(records))
    sender_order = list(sender_order or infer_sender_order(records))
    question_to_index = {qid: idx for idx, qid in enumerate(question_order)}
    sender_to_index = {int(sender_id): idx for idx, sender_id in enumerate(sender_order)}

    by_step: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        step = _safe_int(record.get("step"), default=-1)
        qid = str(record.get("question_id", ""))
        if step < 0 or qid not in question_to_index:
            continue
        by_step.setdefault(step, {})[qid] = record

    step_summaries: List[Dict[str, Any]] = []
    question_count = len(question_order)
    sender_count = len(sender_order)

    for step in sorted(by_step.keys()):
        records_at_step = by_step[step]
        ego_sc = np.zeros((question_count,), dtype=np.float32)
        fused_sc = np.zeros((question_count,), dtype=np.float32)
        gain = np.zeros((question_count,), dtype=np.float32)
        num_candidate_msgs = np.zeros((question_count,), dtype=np.float32)
        num_selected_shared_images = np.zeros((question_count,), dtype=np.float32)

        sender_collab = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_gain = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_age_s = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_distance_m = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_region_alignment = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_facing_alignment = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_distance_alignment = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_available = np.zeros((sender_count, question_count), dtype=np.float32)
        sender_selected = np.zeros((sender_count, question_count), dtype=np.float32)

        for qid, record in records_at_step.items():
            qidx = question_to_index[qid]
            ego_sc[qidx] = _safe_float(record.get("ego_only", {}).get("confidence"))
            fused_sc[qidx] = _safe_float(record.get("ego_plus_shared", {}).get("confidence"))
            gain[qidx] = _safe_float(record.get("confidence_gain"))
            num_candidate_msgs[qidx] = _safe_float(record.get("num_candidate_msgs"))
            num_selected_shared_images[qidx] = _safe_float(record.get("num_selected_shared_images"))

            selected_sender_ids = {
                _safe_int(sender_id, default=-1)
                for sender_id in record.get("selected_sender_ids", []) or []
            }
            per_sender_gain = _extract_sender_gain(record)

            for sensor in record.get("per_sensor_scores", []) or []:
                if bool(sensor.get("is_ego", False)):
                    continue
                sender_id = _safe_int(sensor.get("sender_id"), default=-1)
                if sender_id not in sender_to_index:
                    continue
                sidx = sender_to_index[sender_id]
                sender_available[sidx, qidx] = 1.0
                if sender_id in selected_sender_ids:
                    sender_selected[sidx, qidx] = 1.0

                importance = _extract_importance(sensor)
                age_s = _extract_age_s(sensor)
                timeliness = float(math.exp(-max(age_s, 0.0)))

                sender_collab[sidx, qidx] = max(
                    sender_collab[sidx, qidx],
                    np.float32(importance * timeliness),
                )
                sender_gain[sidx, qidx] = max(
                    sender_gain[sidx, qidx],
                    np.float32(per_sender_gain.get(sender_id, 0.0)),
                )
                sender_age_s[sidx, qidx] = max(sender_age_s[sidx, qidx], np.float32(age_s))
                sender_distance_m[sidx, qidx] = max(
                    sender_distance_m[sidx, qidx],
                    np.float32(_safe_float(sensor.get("distance_m"))),
                )
                sender_region_alignment[sidx, qidx] = max(
                    sender_region_alignment[sidx, qidx],
                    np.float32(_safe_float(sensor.get("region_alignment"))),
                )
                sender_facing_alignment[sidx, qidx] = max(
                    sender_facing_alignment[sidx, qidx],
                    np.float32(_safe_float(sensor.get("facing_alignment"))),
                )
                sender_distance_alignment[sidx, qidx] = max(
                    sender_distance_alignment[sidx, qidx],
                    np.float32(_safe_float(sensor.get("distance_alignment"))),
                )

        step_summaries.append(
            {
                "step": int(step),
                "question_ids": tuple(question_order),
                "sender_ids": tuple(sender_order),
                "ego_sc": ego_sc,
                "fused_sc": fused_sc,
                "gain": gain,
                "num_candidate_msgs": num_candidate_msgs,
                "num_selected_shared_images": num_selected_shared_images,
                "sender_collab": sender_collab,
                "sender_gain": sender_gain,
                "sender_age_s": sender_age_s,
                "sender_distance_m": sender_distance_m,
                "sender_region_alignment": sender_region_alignment,
                "sender_facing_alignment": sender_facing_alignment,
                "sender_distance_alignment": sender_distance_alignment,
                "sender_available": sender_available,
                "sender_selected": sender_selected,
            }
        )
    return step_summaries


def pack_step_features(step_summary: Dict[str, Any]) -> np.ndarray:
    blocks = [
        step_summary["ego_sc"],
        step_summary["fused_sc"],
        step_summary["gain"],
        step_summary["num_candidate_msgs"],
        step_summary["num_selected_shared_images"],
        step_summary["sender_collab"].reshape(-1),
        step_summary["sender_gain"].reshape(-1),
        step_summary["sender_age_s"].reshape(-1),
        step_summary["sender_distance_m"].reshape(-1),
        step_summary["sender_region_alignment"].reshape(-1),
        step_summary["sender_facing_alignment"].reshape(-1),
        step_summary["sender_distance_alignment"].reshape(-1),
        step_summary["sender_available"].reshape(-1),
        step_summary["sender_selected"].reshape(-1),
    ]
    return np.concatenate([np.asarray(block, dtype=np.float32).reshape(-1) for block in blocks], axis=0)


def build_emulation_rollout_examples(
    step_summaries: Sequence[Dict[str, Any]],
    horizon: int,
) -> List[Dict[str, Any]]:
    horizon = max(int(horizon), 0)
    if not step_summaries:
        return []

    examples: List[Dict[str, Any]] = []
    question_count = len(step_summaries[0]["question_ids"])
    sender_count = len(step_summaries[0]["sender_ids"])

    for index, current in enumerate(step_summaries):
        future_mask = np.zeros((horizon,), dtype=np.float32)
        future_ego_sc = np.zeros((horizon, question_count), dtype=np.float32)
        future_gain = np.zeros((horizon, question_count), dtype=np.float32)
        future_sender_collab = np.zeros((horizon, sender_count, question_count), dtype=np.float32)
        future_sender_gain = np.zeros((horizon, sender_count, question_count), dtype=np.float32)

        for offset in range(horizon):
            future_index = index + offset + 1
            if future_index >= len(step_summaries):
                break
            future = step_summaries[future_index]
            future_mask[offset] = 1.0
            future_ego_sc[offset] = future["fused_sc"]
            future_gain[offset] = future["gain"]
            future_sender_collab[offset] = future["sender_collab"]
            future_sender_gain[offset] = future["sender_gain"]

        examples.append(
            {
                "step": int(current["step"]),
                "question_ids": current["question_ids"],
                "sender_ids": current["sender_ids"],
                "input_feature": pack_step_features(current),
                "current_ego_sc": np.asarray(current["ego_sc"], dtype=np.float32),
                "current_fused_sc": np.asarray(current["fused_sc"], dtype=np.float32),
                "current_gain": np.asarray(current["gain"], dtype=np.float32),
                "current_sender_collab": np.asarray(current["sender_collab"], dtype=np.float32),
                "current_sender_gain": np.asarray(current["sender_gain"], dtype=np.float32),
                "future_mask": future_mask,
                "future_ego_sc": future_ego_sc,
                "future_gain": future_gain,
                "future_sender_collab": future_sender_collab,
                "future_sender_gain": future_sender_gain,
            }
        )
    return examples


def _extract_sender_gain(record: Dict[str, Any]) -> Dict[int, float]:
    gains: Dict[int, float] = {}
    ego_sender_ids = {
        _safe_int(sensor.get("sender_id"), default=-1)
        for sensor in record.get("per_sensor_scores", []) or []
        if bool(sensor.get("is_ego", False))
    }

    for item in record.get("aggregated_details", {}).get("per_sender", []) or []:
        sender_id = _safe_int(item.get("sender_id"), default=-1)
        if sender_id < 0 or sender_id in ego_sender_ids:
            continue
        gains[sender_id] = _safe_float(
            item.get("weighted_confidence", item.get("confidence"))
        )

    if gains:
        return gains

    for sensor in record.get("per_sensor_scores", []) or []:
        if bool(sensor.get("is_ego", False)):
            continue
        sender_id = _safe_int(sensor.get("sender_id"), default=-1)
        if sender_id < 0:
            continue
        importance = _extract_importance(sensor)
        gains[sender_id] = gains.get(sender_id, 0.0) + (
            _safe_float(sensor.get("confidence")) * importance
        )
    return gains


def _extract_importance(sensor: Dict[str, Any]) -> float:
    if "importance_weight" in sensor:
        return _safe_float(sensor.get("importance_weight"))
    if "importance_positive" in sensor or "importance_negative" in sensor:
        return max(
            _safe_float(sensor.get("importance_positive")),
            _safe_float(sensor.get("importance_negative")),
        )
    if "total_importance_weight" in sensor:
        return _safe_float(sensor.get("total_importance_weight"))
    return 0.0


def _extract_age_s(sensor: Dict[str, Any]) -> float:
    if "received_age_s_mean" in sensor:
        return _safe_float(sensor.get("received_age_s_mean"))
    if "received_age_s" in sensor:
        return _safe_float(sensor.get("received_age_s"))
    if "latency" in sensor:
        return _safe_float(sensor.get("latency"))
    return 0.0


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
