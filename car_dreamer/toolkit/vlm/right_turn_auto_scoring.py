from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from runtime_logging import get_runtime_logger

from .ego_query_direction_mapper import compute_query_direction_from_observer
from .right_turn_auto_context import RightTurnAutoVLMContextMixin


VLM_SCORING_LOGGER = get_runtime_logger("car_dreamer.vlm.scoring")


class RightTurnAutoVLMScoringMixin(RightTurnAutoVLMContextMixin):
    def _runtime_message_to_predictor_dict(self, msg: Any) -> Dict[str, Any]:
        fixed_dt = float(self._config.world.fixed_delta_seconds)
        received_age_steps = max(int(self._time_step) - int(getattr(msg, "created_step", self._time_step)), 0)
        return {
            "sender_id": int(getattr(msg, "sender_id", -1)),
            "receiver_id": int(getattr(msg, "receiver_id", -1)),
            "created_step": int(getattr(msg, "created_step", self._time_step)),
            "deliver_step": int(getattr(msg, "deliver_step", self._time_step)),
            "received_age_s": float(received_age_steps * fixed_dt),
            "latency_s": float(getattr(msg, "latency_s", 0.0)),
            "payload_bytes": float(getattr(msg, "payload_bytes", 0.0)),
            "distance_m": float(getattr(msg, "distance_m", 0.0)),
        }

    def _build_predictor_candidate_vehicle_states(
        self,
        shared_infos: List[Dict[str, Any]],
        shared_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        selected_infos_by_sender: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for info in shared_infos:
            sender_id = int(info.get("sender_id", -1))
            if sender_id >= 0:
                selected_infos_by_sender[sender_id].append(info)

        shared_source = str(shared_meta.get("shared_source", self._vlm_shared_source))
        window_messages_by_sender: Dict[int, List[Dict[str, Any]]] = {}
        if shared_source == "received_feat" and getattr(self, "ego", None) is not None:
            receiver_id = int(self.ego.id)
            window_msgs = self._get_received_messages_in_window(receiver_id, self._vlm_received_window_s)
            grouped = self._group_messages_by_sender(window_msgs)
            for sender_id, msgs in grouped.items():
                ordered = sorted(msgs, key=lambda msg: (int(msg.deliver_step), int(msg.created_step)))
                window_messages_by_sender[int(sender_id)] = [
                    self._runtime_message_to_predictor_dict(msg) for msg in ordered
                ]

        candidate_vehicle_states: List[Dict[str, Any]] = []
        for actor in self.group_vehs:
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            actor_id = int(actor.id)
            candidate_vehicle_states.append(
                {
                    "vehicle_id": actor_id,
                    "pose": {
                        "x": float(transform.location.x),
                        "y": float(transform.location.y),
                        "yaw": float(transform.rotation.yaw),
                        "yaw_rad": math.radians(float(transform.rotation.yaw)),
                    },
                    "velocity": {
                        "vx": float(velocity.x),
                        "vy": float(velocity.y),
                    },
                    "selected_infos": list(selected_infos_by_sender.get(actor_id, [])),
                    "window_messages": list(window_messages_by_sender.get(actor_id, [])),
                    "shared_source": shared_source,
                }
            )
        return candidate_vehicle_states

    def _maybe_record_emulation_step(
        self,
        eval_result: Dict[str, Any],
        shared_infos: List[Dict[str, Any]],
        shared_meta: Dict[str, Any],
    ) -> None:
        if not hasattr(self, "_emulation_episode_steps") or not hasattr(self, "_emulation_step_counter"):
            return
        if str(eval_result.get("status", "")) != "ok":
            return

        ordered_question_ids = [str(question_cfg["id"]) for question_cfg in self._vlm_questions]
        question_results = eval_result.get("questions", {})
        if any(
            question_id not in question_results
            or "error" in dict(question_results.get(question_id, {}))
            for question_id in ordered_question_ids
        ):
            return

        from .right_turn_auto_predictor_logging import build_runtime_emulation_step

        ego_transform = self.ego.get_transform()
        ego_velocity_actor = self.ego.get_velocity()
        step_record = build_runtime_emulation_step(
            scene_id=str(getattr(self, "_emulation_scene_id", "right_turn_scene")),
            episode_id=str(getattr(self, "_emulation_episode_id", "right_turn_episode")),
            scene_type=str(getattr(self, "_emulation_scene_type", "right_turn")),
            predictor_step=int(self._emulation_step_counter),
            env_step=int(self._time_step),
            dt=float(self._config.world.fixed_delta_seconds),
            ego_pose={
                "x": float(ego_transform.location.x),
                "y": float(ego_transform.location.y),
                "yaw": float(ego_transform.rotation.yaw),
                "yaw_rad": math.radians(float(ego_transform.rotation.yaw)),
            },
            ego_velocity={
                "vx": float(ego_velocity_actor.x),
                "vy": float(ego_velocity_actor.y),
            },
            candidate_vehicle_states=self._build_predictor_candidate_vehicle_states(
                shared_infos,
                shared_meta,
            ),
            question_results=question_results,
            question_ids=ordered_question_ids,
            feature_size=int(self.feature_size),
        )
        self._emulation_episode_steps.append(step_record)
        self._emulation_step_counter += 1

    def _is_rear_question(self, question_cfg: Dict[str, Any]) -> bool:
        qid = str(question_cfg.get("id", "")).lower()
        return "rear" in qid and "front" not in qid

    def _compose_language_evidence(self, sensor_info: Dict[str, Any]) -> str:
        parts: List[str] = []
        scene_description = str(sensor_info.get("scene_description", "")).strip()
        text = str(sensor_info.get("text", "")).strip()
        if scene_description:
            parts.append(f"scene_description:\n{scene_description}")
        if text:
            parts.append(f"message_text:\n{text}")
        return "\n\n".join(parts)

    def _should_use_sensor_for_question(
        self,
        question_cfg: Dict[str, Any],
        sensor_info: Dict[str, Any],
        scoring: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if sensor_info.get("is_ego") and self._is_rear_question(question_cfg):
            return False, "ego_forward_camera_not_expected_to_cover_rear_region"
        if str(scoring.get("question_answerability", "")) == "not_answerable":
            return False, "question_not_answerable_from_sensor_view"
        return True, ""

    def _compute_fov_alignment(
        self,
        sensor_yaw_rad: float,
        target_angle_rad: float,
        fov_deg: float,
    ) -> float:
        half_fov = math.radians(float(fov_deg) / 2.0)
        angle_diff = self._wrap_angle(target_angle_rad - sensor_yaw_rad)
        if abs(angle_diff) > half_fov:
            return 0.0
        return 0.5 * (1.0 + math.cos(angle_diff))

    def _wrap_angle(self, angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def _question_target_offset_xy(self, question_cfg: Dict[str, Any]) -> Tuple[float, float]:
        qid = str(question_cfg.get("id", "")).lower()
        front_d = 4.0
        side_d = 1.5
        if "left_rear" in qid:
            return (-front_d, -side_d)
        if "right_rear" in qid:
            return (-front_d, side_d)
        if "right_front" in qid:
            return (front_d, side_d)
        if "left_front" in qid:
            return (front_d, -side_d)
        if "rear" in qid:
            return (-front_d, 0.0)
        if "left_vehicle_speed" in qid or "left_side" in qid:
            return (0.0, -side_d)
        return (front_d, 0.0)

    def _question_target_angle_rad(self, question_cfg: Dict[str, Any]) -> float:
        dx_local, dy_local = self._question_target_offset_xy(question_cfg)
        return self._wrap_angle(math.atan2(dy_local, dx_local))

    def _question_target_angle_global_rad(
        self,
        question_cfg: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> float:
        target_local = self._question_target_angle_rad(question_cfg)
        ego_yaw_rad = math.radians(float(ego_pose.get("yaw", 0.0)))
        return self._wrap_angle(ego_yaw_rad + target_local)

    def _question_target_point_global(
        self,
        question_cfg: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Dict[str, float]:
        dx_local, dy_local = self._question_target_offset_xy(question_cfg)
        ego_x = float(ego_pose.get("x", 0.0))
        ego_y = float(ego_pose.get("y", 0.0))
        ego_yaw_rad = math.radians(float(ego_pose.get("yaw", 0.0)))
        c = math.cos(ego_yaw_rad)
        s = math.sin(ego_yaw_rad)
        dx_global = c * dx_local - s * dy_local
        dy_global = s * dx_local + c * dy_local
        return {"x": ego_x + dx_global, "y": ego_y + dy_global}

    def _build_sensor_position_features(
        self,
        sensor_score: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Dict[str, float]:
        sx = float(sensor_score.get("pose", {}).get("x", 0.0))
        sy = float(sensor_score.get("pose", {}).get("y", 0.0))
        ex = float(ego_pose.get("x", 0.0))
        ey = float(ego_pose.get("y", 0.0))
        dx = sx - ex
        dy = sy - ey
        d = math.sqrt(dx * dx + dy * dy)
        if d > 1e-6:
            bearing_from_ego = math.atan2(dy, dx)
            bearing_to_ego = math.atan2(-dy, -dx)
        else:
            bearing_from_ego = 0.0
            bearing_to_ego = 0.0
        psi = float(sensor_score.get("sensor_yaw_rad", 0.0))
        return {
            "sensor_x": sx,
            "sensor_y": sy,
            "ego_x": ex,
            "ego_y": ey,
            "dx": dx,
            "dy": dy,
            "distance_m": d,
            "bearing_from_ego_rad": bearing_from_ego,
            "bearing_to_ego_rad": bearing_to_ego,
            "bearing_from_ego_deg": float((math.degrees(bearing_from_ego) + 180.0) % 360.0 - 180.0),
            "bearing_to_ego_deg": float((math.degrees(bearing_to_ego) + 180.0) % 360.0 - 180.0),
            "sin_theta": float(math.sin(bearing_from_ego)),
            "cos_theta": float(math.cos(bearing_from_ego)),
            "sensor_yaw_rad": psi,
            "sensor_yaw_deg": float((math.degrees(psi) + 180.0) % 360.0 - 180.0),
            "sin_psi": float(math.sin(psi)),
            "cos_psi": float(math.cos(psi)),
        }

    def _aggregate_sensor_scores(
        self,
        per_sensor_instance_scores: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in per_sensor_instance_scores:
            key = f"{int(item['sender_id'])}:{str(item['sensor_name'])}"
            grouped[key].append(item)

        aggregated: Dict[str, Dict[str, Any]] = {}
        for key, items in grouped.items():
            n = float(len(items))
            aggregated[key] = {
                "sensor_key": key,
                "sender_id": int(items[0]["sender_id"]),
                "sensor_name": str(items[0]["sensor_name"]),
                "is_ego": bool(items[0].get("is_ego", False)),
                "positive_score": float(sum(x["positive_score"] for x in items) / n),
                "negative_score": float(sum(x["negative_score"] for x in items) / n),
                "unknown_score": float(
                    sum(x.get("unknown_score", x.get("uncertainty", 0.0)) for x in items) / n
                ),
                "belief": float(sum(x.get("belief", 0.0) for x in items) / n),
                "evidence": float(sum(x.get("evidence", 0.0) for x in items) / n),
                "ambiguity": float(sum(x.get("ambiguity", 1.0) for x in items) / n),
                "confidence": float(sum(float(x.get("confidence", 0.0)) for x in items) / n),
                "visibility_score": float(
                    sum(1.0 - float(x.get("uncertainty", x.get("unknown_score", 1.0))) for x in items) / n
                ),
                "answer": str(items[-1].get("answer", "uncertain")),
                "num_images": int(len(items)),
                "latency": float(min(float(x.get("received_age_s", 0.0)) for x in items)),
                "received_age_s_mean": float(
                    sum(float(x.get("received_age_s", 0.0)) for x in items) / n
                ),
                "pose": dict(items[0].get("pose", {})),
                "sensor_yaw_rad": float(items[0].get("sensor_yaw_rad", 0.0)),
                "reason": str(items[-1].get("reason", "")),
                "reasons": [str(x.get("reason", "")) for x in items],
                "raw_outputs": [str(x.get("raw_text", "")) for x in items],
                "scene_description": str(items[-1].get("scene_description", "")),
                "language_evidence": str(items[-1].get("language_evidence", "")),
                "converted_query": str(items[-1].get("converted_query", "")),
                "evaluation_mode": str(items[-1].get("evaluation_mode", "")),
                "visibility_status": str(items[-1].get("visibility_status", "partial")),
                "question_answerability": str(
                    items[-1].get("question_answerability", "partially_answerable")
                ),
                "answerability_score": float(items[-1].get("answerability_score", 0.0)),
                "used_for_aggregation": bool(items[-1].get("used_for_aggregation", True)),
                "skip_reason": str(items[-1].get("skip_reason", "")),
                "image_available": bool(items[-1].get("image_available", False)),
                "raw_vlm_json": items[-1].get("raw_vlm_json", {}),
            }
        return aggregated

    def _aggregate_sender_scores_from_sensor_means(
        self,
        sensor_mean_scores: Dict[str, Dict[str, Any]],
    ) -> Dict[int, Dict[str, float]]:
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for sensor_score in sensor_mean_scores.values():
            grouped[int(sensor_score["sender_id"])].append(sensor_score)

        aggregated: Dict[int, Dict[str, float]] = {}
        for sender_id, items in grouped.items():
            n = float(len(items))
            aggregated[sender_id] = {
                "positive_score": float(sum(x["positive_score"] for x in items) / n),
                "negative_score": float(sum(x["negative_score"] for x in items) / n),
                "unknown_score": float(sum(x.get("unknown_score", 0.0) for x in items) / n),
                "belief": float(sum(x.get("belief", 0.0) for x in items) / n),
                "evidence": float(sum(x.get("evidence", 0.0) for x in items) / n),
                "confidence": float(sum(x["confidence"] for x in items) / n),
                "num_sensors": int(len(items)),
            }
        return aggregated

    def _compute_sensor_importance_maps(
        self,
        question_cfg: Dict[str, Any],
        sensor_mean_scores: Dict[str, Dict[str, Any]],
        ego_pose: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        target_angle = self._question_target_angle_global_rad(question_cfg, ego_pose)
        target_point = self._question_target_point_global(question_cfg, ego_pose)
        target_angle_deg = (math.degrees(target_angle) + 180.0) % 360.0 - 180.0
        tau = max(float(self._vlm_importance_distance_tau), 1e-6)
        half_fov = 0.5 * math.radians(float(self._vlm_sensor_fov_deg))

        sensor_keys: List[str] = []
        weights: List[float] = []
        details: Dict[str, Dict[str, Any]] = {}

        for sensor_key, sensor_score in sensor_mean_scores.items():
            feats = self._build_sensor_position_features(sensor_score, ego_pose)
            sensor_x = float(feats["sensor_x"])
            sensor_y = float(feats["sensor_y"])
            sensor_yaw_rad = float(feats["sensor_yaw_rad"])
            bearing_to_target_rad = math.atan2(
                float(target_point["y"]) - sensor_y,
                float(target_point["x"]) - sensor_x,
            )
            angle_diff_target = self._wrap_angle(sensor_yaw_rad - bearing_to_target_rad)
            facing_alignment = 0.5 * (1.0 + math.cos(angle_diff_target))
            norm = angle_diff_target / max(half_fov, 1e-6)
            if abs(norm) <= 1.0:
                fov_alignment = 0.5 * (1.0 + math.cos(0.5 * norm * math.pi))
            else:
                fov_alignment = 0.0

            if sensor_score["is_ego"]:
                region_alignment = 1.0
                distance_alignment = 1.0
            else:
                region_alignment = 0.5 * (
                    1.0
                    + math.cos(
                        self._wrap_angle(float(feats["bearing_from_ego_rad"]) - target_angle)
                    )
                )
                distance_alignment = math.exp(-float(feats["distance_m"]) / tau)

            weight = (
                self._vlm_importance_region_weight * float(region_alignment)
                * self._vlm_importance_facing_weight * float(facing_alignment)
                * float(fov_alignment)
                * self._vlm_importance_distance_weight * float(distance_alignment)
            )
            if sensor_score["is_ego"]:
                weight += float(self._vlm_importance_ego_bias)
            weight = max(weight, 0.0)

            sensor_keys.append(sensor_key)
            weights.append(weight)
            details[sensor_key] = {
                **feats,
                "target_point_x": float(target_point["x"]),
                "target_point_y": float(target_point["y"]),
                "target_angle_global_rad": float(target_angle),
                "target_angle": float(target_angle_deg),
                "bearing_to_target_rad": float(bearing_to_target_rad),
                "bearing_to_target_deg": float(
                    (math.degrees(bearing_to_target_rad) + 180.0) % 360.0 - 180.0
                ),
                "angle_diff_target_rad": float(angle_diff_target),
                "angle_diff_target_deg": float(
                    (math.degrees(angle_diff_target) + 180.0) % 360.0 - 180.0
                ),
                "region_alignment": float(region_alignment),
                "fov_alignment": float(fov_alignment),
                "facing_alignment": float(facing_alignment),
                "distance_alignment": float(distance_alignment),
                "importance_weight": float(weight),
            }

        per_sensor: Dict[str, Dict[str, Any]] = {}
        per_sender_positive: Dict[int, float] = defaultdict(float)
        per_sender_negative: Dict[int, float] = defaultdict(float)
        for sensor_key, weight in zip(sensor_keys, weights):
            sender_id = int(sensor_mean_scores[sensor_key]["sender_id"])
            per_sensor[sensor_key] = {**details[sensor_key]}
            per_sender_positive[sender_id] += float(weight)
            per_sender_negative[sender_id] += float(weight)

        return {
            "per_sensor": per_sensor,
            "per_sender_positive": {int(k): float(v) for k, v in per_sender_positive.items()},
            "per_sender_negative": {int(k): float(v) for k, v in per_sender_negative.items()},
        }

    def _compute_information(
        self,
        sensor_mean_scores: Dict[str, Dict[str, Any]],
        importance_maps: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        per_sensor_information: Dict[str, float] = {}
        total_information = 0.0
        for sensor_key, sensor_score in sensor_mean_scores.items():
            imp = importance_maps.get("per_sensor", {}).get(sensor_key, {})
            sender_id = int(sensor_score["sender_id"])
            is_ego = bool(sensor_score.get("is_ego", False))
            sender_weight = self._get_sender_weight(sender_id, is_ego)
            multiplier = float(imp.get("importance_weight", 0.0)) * sender_weight
            s_pos = float(sensor_score.get("positive_score", 0.0))
            s_neg = float(sensor_score.get("negative_score", 0.0))
            evidence = float(sensor_score.get("evidence", 0.0))
            information = multiplier * (
                math.log1p(max(s_pos, 0.0) + max(s_neg, 0.0))
                + 0.5 * math.log1p(max(evidence, 0.0))
            )
            per_sensor_information[sensor_key] = float(information)
            total_information += float(information)
        return {
            "beta": float(self._vlm_sc_beta),
            "unweighted_information": float(total_information),
            "weighted_information": float(self._vlm_sc_beta * total_information),
            "per_sensor_information": per_sensor_information,
        }

    def _get_sender_weight(self, sender_id: int, is_ego: bool) -> float:
        if is_ego:
            return self._vlm_ego_conf_weight
        return float(
            self._vlm_shared_conf_weights.get(
                int(sender_id), self._vlm_default_shared_conf_weight
            )
        )

    def _weighted_confidence_aggregate(
        self,
        sensor_mean_scores: Dict[str, Dict[str, Any]],
        importance_maps: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        per_sensor_importance = importance_maps.get("per_sensor", {})
        per_sensor_details: List[Dict[str, Any]] = []
        total_pos = total_neg = total_unk = total_evidence = total_belief = 0.0
        total_effective_weight = 0.0
        total_coverage_weight = 0.0
        ego_only = {
            "positive_score": 0.0,
            "negative_score": 0.0,
            "unknown_score": 0.1,
            "belief": 0.0,
            "evidence": 0.0,
            "confidence": 0.0,
            "answer": "uncertain",
        }

        if not sensor_mean_scores:
            return {
                "positive_score": 0.0,
                "negative_score": 0.0,
                "unknown_score": 0.1,
                "belief": 0.0,
                "evidence": 0.0,
                "confidence": 0.0,
                "coverage_weight": 0.0,
                "effective_weight": 0.0,
                "ego_only": ego_only,
                "per_sensor_details": per_sensor_details,
                "answer": "uncertain",
            }

        for sensor_key in sorted(sensor_mean_scores.keys()):
            sensor_score = sensor_mean_scores[sensor_key]
            imp = per_sensor_importance.get(sensor_key, {})
            latency = float(sensor_score.get("received_age_s_mean", sensor_score.get("latency", 0.0)))
            timeliness = math.exp(-latency)
            importance_weight = float(imp.get("importance_weight", 0.0))
            sender_id = int(sensor_score["sender_id"])
            is_ego = bool(sensor_score.get("is_ego", False))
            sender_weight = self._get_sender_weight(sender_id, is_ego)
            sensor_weight = importance_weight * sender_weight * timeliness

            s_pos = float(sensor_score.get("positive_score", 0.0))
            s_neg = float(sensor_score.get("negative_score", 0.0))
            s_unk = float(sensor_score.get("unknown_score", 0.1))
            belief = float(sensor_score.get("belief", s_pos - s_neg))
            evidence = float(sensor_score.get("evidence", 1.0 - s_unk))
            answerability_score = float(sensor_score.get("answerability_score", 1.0))
            coverage_weight = sensor_weight * answerability_score
            effective_weight = coverage_weight * max(evidence, 0.0)

            total_pos += effective_weight * s_pos
            total_neg += effective_weight * s_neg
            total_unk += sensor_weight * s_unk
            total_evidence += effective_weight
            total_belief += effective_weight * belief
            total_coverage_weight += coverage_weight
            total_effective_weight += effective_weight

            sensor_detail = {
                "sensor_key": sensor_key,
                "sender_id": sender_id,
                "is_ego": is_ego,
                "sensor_weight": sensor_weight,
                "coverage_weight": coverage_weight,
                "answerability_score": answerability_score,
                "positive_score": s_pos,
                "negative_score": s_neg,
                "unknown_score": s_unk,
                "belief": belief,
                "evidence": evidence,
                "timeliness": timeliness,
                "importance_weight": importance_weight,
                "sender_weight": sender_weight,
                "weight": effective_weight,
                "confidence": abs(belief) * effective_weight,
                "answer": str(sensor_score.get("answer", "uncertain")),
                "visibility_status": str(sensor_score.get("visibility_status", "partial")),
                "question_answerability": str(
                    sensor_score.get("question_answerability", "partially_answerable")
                ),
            }
            per_sensor_details.append(sensor_detail)
            if is_ego:
                ego_only = dict(sensor_detail)

        if total_effective_weight <= 1e-6 or total_coverage_weight <= 1e-6:
            final_answer = "uncertain"
        elif total_belief > 0.02:
            final_answer = "positive"
        elif total_belief < -0.02:
            final_answer = "negative"
        else:
            final_answer = "uncertain"

        norm = max(total_effective_weight, 1e-6)
        coverage_norm = max(total_coverage_weight, 1e-6)
        avg_belief = total_belief / norm if total_effective_weight > 1e-6 else 0.0
        confidence = abs(avg_belief) * min(1.0, total_effective_weight)

        return {
            "positive_score": total_pos / norm,
            "negative_score": total_neg / norm,
            "unknown_score": min(total_unk / coverage_norm, 1.0),
            "belief": total_belief,
            "evidence": min(total_effective_weight / coverage_norm, 1.0),
            "confidence": confidence,
            "coverage_weight": total_coverage_weight,
            "effective_weight": total_effective_weight,
            "ego_only": ego_only,
            "per_sensor_details": per_sensor_details,
            "answer": final_answer,
        }

    # =========================================================
    # Group vehicles and observer lifecycle
    # =========================================================

    def _build_ego_sensor_instances(self, ego_image: Image.Image) -> List[Dict[str, Any]]:
        tf = self.ego.get_transform()
        scene_description = self._compute_single_image_description(ego_image, token_size=self.feature_size)
        return [{
            "sender_id": int(self.ego.id),
            "sensor_name": "cam0",
            "image": ego_image,
            "img_emb": None,
            "scene_description": scene_description,
            "text": "",
            "pose": {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "yaw": float(tf.rotation.yaw),
            },
            "sensor_yaw_rad": math.radians(float(tf.rotation.yaw)),
            "received_age_s": 0.0,
            "is_ego": True,
        }]

    def _build_shared_sensor_instances(self, shared_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sensor_infos: List[Dict[str, Any]] = []
        for info in shared_infos:
            pose = self._normalize_pose_dict(info.get("pose"))
            sensor_infos.append({
                "sender_id": int(info["sender_id"]),
                "sensor_name": str(info.get("sensor_name", "cam0")),
                "image": info.get("image"),
                "img_emb": info.get("img_emb"),
                "scene_description": str(info.get("scene_description", "")).strip(),
                "text": str(info.get("text", "")).strip(),
                "pose": pose,
                "sensor_yaw_rad": math.radians(float(pose["yaw"])),
                "received_age_s": float(info.get("received_age_s", 0.0)),
                "is_ego": False,
            })
        return sensor_infos

    def _build_sensor_score_record(
        self,
        sensor_info: Dict[str, Any],
        scene_description: str,
        language_evidence: str,
        scoring: Dict[str, Any],
        converted_question_cfg: Optional[Dict[str, Any]],
        evaluation_mode: str,
        used_for_aggregation: bool,
        skip_reason: str,
    ) -> Dict[str, Any]:
        return {
            "sender_id": int(sensor_info["sender_id"]),
            "sensor_name": str(sensor_info["sensor_name"]),
            "is_ego": bool(sensor_info.get("is_ego", False)),
            "received_age_s": float(sensor_info.get("received_age_s", 0.0)),
            "pose": dict(sensor_info.get("pose", {})),
            "sensor_yaw_rad": float(sensor_info.get("sensor_yaw_rad", 0.0)),
            "converted_query": converted_question_cfg["query"] if converted_question_cfg else None,
            "scene_description": scene_description,
            "language_evidence": language_evidence,
            "evaluation_mode": evaluation_mode,
            "image_available": bool(sensor_info.get("image") is not None),
            "used_for_aggregation": bool(used_for_aggregation),
            "skip_reason": str(skip_reason),
            **scoring,
        }

    def _evaluate_single_question(
        self,
        question_cfg: Dict[str, Any],
        ego_image: Image.Image,
        shared_images: List[Image.Image],
        shared_infos: List[Dict[str, Any]],
        shared_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        ego_sensor_infos = self._build_ego_sensor_instances(ego_image)
        shared_sensor_infos = self._build_shared_sensor_instances(shared_infos)
        all_sensor_infos = ego_sensor_infos + shared_sensor_infos

        ego_tf = self.ego.get_transform()
        ego_pose = {
            "x": float(ego_tf.location.x),
            "y": float(ego_tf.location.y),
            "yaw": float(ego_tf.rotation.yaw),
        }
        per_sensor_instance_scores: List[Dict[str, Any]] = []

        for sensor_info in all_sensor_infos:
            scene_description = str(sensor_info.get("scene_description", "")).strip()
            language_evidence = self._compose_language_evidence(sensor_info)
            converted_question_cfg = None
            question_for_sensor = question_cfg
            if sensor_info.get("is_ego"):
                pass
            else:
                converted = compute_query_direction_from_observer(
                    ego_pose=ego_pose,
                    observer_pose=sensor_info.get("pose"),
                    question_id=question_cfg["id"],
                )
                converted_question_cfg = {
                    "id": question_cfg["id"],
                    "type": question_cfg["type"],
                    "query": converted.query,
                    "positive": converted.positive,
                    "negative": converted.negative,
                }
                question_for_sensor = converted_question_cfg

            image = sensor_info.get("image")
            if image is not None:
                scoring = self._score_question_from_visual_evidence(question_for_sensor, image)
                evaluation_mode = "visual_question"
            elif language_evidence:
                scoring = self._score_question_from_language_evidence(
                    question_for_sensor,
                    language_evidence,
                )
                evaluation_mode = "language_fallback"
            else:
                scoring = {
                    "positive_score": 0.0,
                    "negative_score": 0.0,
                    "unknown_score": 0.1,
                    "uncertainty": 0.1,
                    "answer": "uncertain",
                    "reason": "No image or textual evidence available for this sensor.",
                    "belief": 0.0,
                    "evidence": 0.0,
                    "ambiguity": 1.0,
                    "confidence": 0.0,
                    "visibility_status": "not_visible",
                    "question_answerability": "not_answerable",
                    "answerability_score": 0.0,
                    "raw_vlm_json": {},
                    "raw_text": "",
                }
                evaluation_mode = "empty_evidence"

            used_for_aggregation, skip_reason = self._should_use_sensor_for_question(
                question_cfg,
                sensor_info,
                scoring,
            )
            per_sensor_instance_scores.append(
                self._build_sensor_score_record(
                    sensor_info,
                    scene_description,
                    language_evidence,
                    scoring,
                    converted_question_cfg,
                    evaluation_mode,
                    used_for_aggregation,
                    skip_reason,
                )
            )

        sensor_mean_scores_all = self._aggregate_sensor_scores(per_sensor_instance_scores)
        usable_instance_scores = [
            item for item in per_sensor_instance_scores if bool(item.get("used_for_aggregation", True))
        ]
        sensor_mean_scores_used = self._aggregate_sensor_scores(usable_instance_scores)
        importance_maps = self._compute_sensor_importance_maps(question_cfg, sensor_mean_scores_used, ego_pose)
        information_level = self._compute_information(sensor_mean_scores_used, importance_maps)
        aggregated = self._weighted_confidence_aggregate(sensor_mean_scores_used, importance_maps)
        ego_score = aggregated["ego_only"]

        per_sensor_scores_out: List[Dict[str, Any]] = []
        for sensor_key in sorted(sensor_mean_scores_all.keys()):
            sensor_score = sensor_mean_scores_all[sensor_key]
            imp = importance_maps.get("per_sensor", {}).get(sensor_key, {})
            per_sensor_scores_out.append(
                {
                    **sensor_score,
                    **imp,
                    "information_term": float(
                        information_level["per_sensor_information"].get(sensor_key, 0.0)
                    ),
                }
            )

        return {
            "question_id": question_cfg["id"],
            "question_type": question_cfg["type"],
            "shared_source": shared_meta.get("shared_source", self._vlm_shared_source),
            "num_candidate_msgs": shared_meta.get("num_candidate_msgs", 0),
            "num_selected_shared_images": shared_meta.get(
                "num_selected_shared_images", len(shared_images)
            ),
            "selected_sender_ids": shared_meta.get("selected_sender_ids", []),
            "received_window_s": shared_meta.get("window_s", 0.0),
            "sampling_strategy": shared_meta.get("sampling_strategy", ""),
            "query_text": question_cfg["query"],
            "positive_text": question_cfg["positive"],
            "negative_text": question_cfg["negative"],
            "per_sensor_scores": per_sensor_scores_out,
            "sender_importance_positive": importance_maps.get("per_sender_positive", {}),
            "sender_importance_negative": importance_maps.get("per_sender_negative", {}),
            "ego_only": ego_score,
            "ego_plus_shared": {
                "positive_score": aggregated["positive_score"],
                "negative_score": aggregated["negative_score"],
                "unknown_score": aggregated["unknown_score"],
                "belief": aggregated["belief"],
                "evidence": aggregated["evidence"],
                "confidence": aggregated["confidence"],
                "answer": aggregated["answer"],
                "coverage_weight": aggregated.get("coverage_weight", 0.0),
                "effective_weight": aggregated.get("effective_weight", 0.0),
            },
            "weighted_fused": {
                "positive_score": aggregated["positive_score"],
                "negative_score": aggregated["negative_score"],
                "unknown_score": aggregated["unknown_score"],
                "belief": aggregated["belief"],
                "evidence": aggregated["evidence"],
                "confidence": aggregated["confidence"],
                "answer": aggregated["answer"],
                "coverage_weight": aggregated.get("coverage_weight", 0.0),
                "effective_weight": aggregated.get("effective_weight", 0.0),
            },
            "aggregated_details": {"per_sensor": aggregated["per_sensor_details"]},
            "sc_part1": float(aggregated["confidence"]),
            "sc_part2_information": float(information_level["weighted_information"]),
            "sc_part2_details": information_level,
            "confidence_with_part2": aggregated["confidence"]
            + float(information_level["weighted_information"]),
            "confidence_gain": float(aggregated["confidence"] - ego_score.get("confidence", 0.0)),
        }

    def _evaluate_vlm_questions(self) -> Dict[str, Any]:
        if not self._vlm_enabled:
            VLM_SCORING_LOGGER.debug("VLM evaluation skipped because VLM is disabled.")
            return {}
        ego_image, shared_images, shared_infos, shared_meta = self._get_ego_and_shared_images_info()
        VLM_SCORING_LOGGER.debug(
            "Evaluating VLM questions step=%d shared_source=%s candidate_msgs=%s selected_images=%s",
            int(self._time_step),
            shared_meta.get("shared_source", self._vlm_shared_source),
            shared_meta.get("num_candidate_msgs", 0),
            shared_meta.get("num_selected_shared_images", len(shared_images)),
        )
        if ego_image is None:
            self._vlm_last_eval = {
                "step": int(self._time_step),
                "status": "skipped_no_ego_image",
            }
            return self._vlm_last_eval

        eval_result: Dict[str, Any] = {
            "step": int(self._time_step),
            "status": "ok",
            "shared_source": shared_meta.get("shared_source", self._vlm_shared_source),
            "num_candidate_msgs": shared_meta.get("num_candidate_msgs", 0),
            "num_selected_shared_images": shared_meta.get(
                "num_selected_shared_images", len(shared_images)
            ),
            "selected_sender_ids": shared_meta.get("selected_sender_ids", []),
            "received_window_s": shared_meta.get("window_s", 0.0),
            "sampling_strategy": shared_meta.get("sampling_strategy", ""),
            "questions": {},
        }

        for qcfg in self._vlm_questions:
            try:
                qres = self._evaluate_single_question(
                    qcfg,
                    ego_image,
                    shared_images,
                    shared_infos,
                    shared_meta,
                )
                eval_result["questions"][qcfg["id"]] = qres
                self._vlm_records.append({"step": int(self._time_step), **qres})
            except Exception as exc:
                VLM_SCORING_LOGGER.exception(
                    "VLM single-question evaluation failed step=%d question_id=%s",
                    int(self._time_step),
                    qcfg["id"],
                )
                eval_result["questions"][qcfg["id"]] = {
                    "question_id": qcfg["id"],
                    "question_type": qcfg["type"],
                    "error": str(exc),
                }

        self._vlm_last_eval = eval_result
        try:
            self._maybe_record_emulation_step(eval_result, shared_infos, shared_meta)
        except Exception:
            VLM_SCORING_LOGGER.exception(
                "Predictor-ready canonical logging failed step=%d",
                int(self._time_step),
            )
        VLM_SCORING_LOGGER.debug(
            "Completed VLM evaluation step=%d questions=%d",
            int(self._time_step),
            len(eval_result["questions"]),
        )
        return eval_result

    # =========================================================
    # Graph export and episode output
    # =========================================================
