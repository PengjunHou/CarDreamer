from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from runtime_logging import get_runtime_logger

from .ego_query_direction_mapper import compute_query_direction_from_observer
from .right_turn_auto_context import RightTurnAutoVLMContextMixin


VLM_SCORING_LOGGER = get_runtime_logger("car_dreamer.vlm.scoring")

_SCENE_DESCRIPTION_REGION_ROWS: Tuple[str, ...] = (
    "Front",
    "Left-front",
    "Right-front",
    "Rear",
    "Left-rear",
    "Right-rear",
)
_SCENE_DESCRIPTION_REGION_ALIASES: Dict[str, str] = {
    row.lower(): row for row in _SCENE_DESCRIPTION_REGION_ROWS
}
_SCENE_DESCRIPTION_DIRECTION_TO_ROWS: Dict[str, Tuple[str, ...]] = {
    "front": ("Front",),
    "front-left": ("Left-front",),
    "left-front": ("Left-front",),
    "front-right": ("Right-front",),
    "right-front": ("Right-front",),
    "rear": ("Rear",),
    "rear-left": ("Left-rear",),
    "left-rear": ("Left-rear",),
    "rear-right": ("Right-rear",),
    "right-rear": ("Right-rear",),
    "left": ("Left-front", "Left-rear"),
    "right": ("Right-front", "Right-rear"),
}
_SCENE_DESCRIPTION_NEGATIVE_PHRASES: Tuple[str, ...] = (
    "no vehicle",
    "no vehicles",
    "no visible vehicle",
    "no visible vehicles",
    "no other vehicles",
    "vehicles are not visible",
    "vehicle is not visible",
    "road ahead is clear",
    "road is clear",
    "road appears clear",
)
_SCENE_DESCRIPTION_POSITIVE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(car|cars|truck|trucks|bus|buses|van|vans|suv|suvs|pickup|pickups)\b"),
    re.compile(r"\b(?:a|an)\s+(?:\w+\s+){0,2}vehicle\b"),
    re.compile(r"\bvehicle\s+(?:is\s+)?visible\b"),
    re.compile(r"\bvehicles\s+are\s+visible\b"),
    re.compile(r"\bvehicle\s+(?:directly\s+)?ahead\b"),
)


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
            "payload_type": str(getattr(msg, "payload", {}).get("payload_type", "tokens")),
            "payload_encoder_id": str(getattr(msg, "payload", {}).get("payload_encoder_id", "tokens_v1")),
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
            policy_action = {}
            payload_action = {}
            runtime_comm_stats = {}
            if hasattr(self, "_get_policy_action_value"):
                policy_action = dict(self._get_policy_action_value(actor_id))
            if hasattr(self, "_get_payload_action_value"):
                payload_action = dict(self._get_payload_action_value(actor_id))
            if hasattr(self, "_get_latest_comm_link_analysis"):
                runtime_comm_stats = dict(self._get_latest_comm_link_analysis(actor_id))
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
                    "policy_action": policy_action,
                    "payload_action": payload_action,
                    "runtime_comm_stats": runtime_comm_stats,
                    "policy_id": str(shared_meta.get("policy_id", getattr(self, "_collaboration_policy_id", ""))),
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
            policy_id=str(shared_meta.get("policy_id", getattr(self, "_collaboration_policy_id", ""))),
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
            step_communication_stats=(
                dict(self._get_current_comm_step_summary())
                if hasattr(self, "_get_current_comm_step_summary")
                else None
            ),
            step_metadata={
                **dict(shared_meta or {}),
                **(
                    dict(self._get_current_policy_decision())
                    if hasattr(self, "_get_current_policy_decision")
                    else {}
                ),
            },
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

    def _compose_caption_only_evidence(self, scene_description: str) -> str:
        scene_description = str(scene_description).strip()
        if not scene_description:
            return ""
        return f"scene_description:\n{scene_description}"

    def _parse_scene_description_regions(self, scene_description: str) -> Dict[str, str]:
        regions = {row: "" for row in _SCENE_DESCRIPTION_REGION_ROWS}
        for raw_line in str(scene_description).splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            canonical = _SCENE_DESCRIPTION_REGION_ALIASES.get(key.strip().lower())
            if canonical is None:
                continue
            # Later rows override earlier ones so merged multi-age captions use the newest snapshot.
            regions[canonical] = value.strip()
        return regions

    def _is_scene_region_explicitly_not_visible(self, region_text: str) -> bool:
        normalized = str(region_text).strip().strip(".").strip().lower().replace(" ", "_")
        return normalized == "not_visible"

    def _is_scene_region_visible_text(self, region_text: str) -> bool:
        region_text = str(region_text).strip()
        return bool(region_text) and not self._is_scene_region_explicitly_not_visible(region_text)

    def _question_direction_from_id(self, question_cfg: Dict[str, Any]) -> str:
        qid = str(question_cfg.get("id", "")).strip().lower()
        if "left_rear" in qid:
            return "rear-left"
        if "right_rear" in qid:
            return "rear-right"
        if "right_front" in qid:
            return "front-right"
        if "left_front" in qid:
            return "front-left"
        if "rear" in qid and "front" not in qid:
            return "rear"
        if "left_vehicle_speed" in qid or "left_side" in qid:
            return "left"
        if "right_vehicle_speed" in qid or "right_side" in qid:
            return "right"
        return "front"

    def _query_direction_from_text(self, query_text: Any) -> str:
        normalized = str(query_text).strip().lower().replace("_", "-")
        for direction in (
            "front-right",
            "front-left",
            "rear-right",
            "rear-left",
            "right-front",
            "left-front",
            "right-rear",
            "left-rear",
            "right",
            "left",
            "rear",
            "front",
        ):
            if f"{direction} region" in normalized:
                return direction
        return ""

    def _resolve_scene_description_direction(
        self,
        question_cfg: Dict[str, Any],
        sensor_info: Dict[str, Any],
        converted_question_cfg: Optional[Dict[str, Any]],
    ) -> str:
        if not sensor_info.get("is_ego") and converted_question_cfg:
            direction = str(converted_question_cfg.get("direction", "")).strip().lower()
            if direction:
                return direction
            direction = self._query_direction_from_text(converted_question_cfg.get("query", ""))
            if direction:
                return direction
        return self._question_direction_from_id(question_cfg)

    def _scene_region_has_explicit_negative_vehicle_evidence(self, region_text: str) -> bool:
        normalized = " ".join(str(region_text).strip().lower().split())
        if not normalized or self._is_scene_region_explicitly_not_visible(normalized):
            return False
        return any(phrase in normalized for phrase in _SCENE_DESCRIPTION_NEGATIVE_PHRASES)

    def _scene_region_has_explicit_positive_vehicle_evidence(self, region_text: str) -> bool:
        normalized = " ".join(str(region_text).strip().lower().split())
        if not normalized or self._scene_region_has_explicit_negative_vehicle_evidence(normalized):
            return False
        return any(pattern.search(normalized) for pattern in _SCENE_DESCRIPTION_POSITIVE_PATTERNS)

    def _derive_scene_description_consistency_hint(
        self,
        question_cfg: Dict[str, Any],
        sensor_info: Dict[str, Any],
        scene_description: str,
        converted_question_cfg: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        direction = self._resolve_scene_description_direction(
            question_cfg,
            sensor_info,
            converted_question_cfg,
        )
        region_labels = list(
            _SCENE_DESCRIPTION_DIRECTION_TO_ROWS.get(
                str(direction).strip().lower().replace("_", "-"),
                ("Front",),
            )
        )
        regions = self._parse_scene_description_regions(scene_description)
        region_texts = {label: str(regions.get(label, "")).strip() for label in region_labels}
        visible_texts = {
            label: text for label, text in region_texts.items() if self._is_scene_region_visible_text(text)
        }
        visibility_status = "visible" if visible_texts else "not_visible"

        answer_hint = "unknown"
        if visible_texts:
            if any(
                self._scene_region_has_explicit_positive_vehicle_evidence(text)
                for text in visible_texts.values()
            ):
                answer_hint = "positive"
            elif len(visible_texts) == len(region_labels) and all(
                self._scene_region_has_explicit_negative_vehicle_evidence(text)
                for text in visible_texts.values()
            ):
                answer_hint = "negative"

        return {
            "direction": direction,
            "region_labels": region_labels,
            "region_texts": region_texts,
            "visibility_status": visibility_status,
            "answer_hint": answer_hint,
        }

    def _apply_scene_description_consistency_override(
        self,
        question_cfg: Dict[str, Any],
        sensor_info: Dict[str, Any],
        scene_description: str,
        scoring: Dict[str, Any],
        converted_question_cfg: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        visibility_status = str(scoring.get("visibility_status", "")).strip()
        question_answerability = str(scoring.get("question_answerability", "")).strip()
        if visibility_status != "not_visible" and question_answerability != "not_answerable":
            return scoring

        hint = self._derive_scene_description_consistency_hint(
            question_cfg,
            sensor_info,
            scene_description,
            converted_question_cfg,
        )
        if hint["visibility_status"] != "visible":
            return scoring

        answer_hint = str(hint["answer_hint"])
        if answer_hint == "positive":
            payload = {
                "answer": "positive",
                "visibility_status": "visible",
                "question_answerability": "answerable",
                "support_strength": "moderate",
                "reason": "Caption consistency override: target region description indicates a visible vehicle.",
            }
        elif answer_hint == "negative":
            payload = {
                "answer": "negative",
                "visibility_status": "visible",
                "question_answerability": "answerable",
                "support_strength": "moderate",
                "reason": "Caption consistency override: target region description indicates the visible region is clear of vehicles.",
            }
        else:
            payload = {
                "answer": "insufficient",
                "visibility_status": "visible",
                "question_answerability": "partially_answerable",
                "support_strength": "none",
                "reason": "Caption consistency override: target region is visible, but vehicle presence remains unresolved.",
            }

        payload.update(
            {
                "caption_consistency_override": True,
                "caption_target_direction": hint["direction"],
                "caption_target_regions": list(hint["region_labels"]),
                "caption_target_region_texts": dict(hint["region_texts"]),
                "original_visibility_status": visibility_status,
                "original_question_answerability": question_answerability,
                "original_reason": str(scoring.get("reason", "")).strip(),
                "original_vlm_json": dict(scoring.get("raw_vlm_json", {}) or {}),
            }
        )
        return self._parse_language_scores(json.dumps(payload, ensure_ascii=False))

    def _get_converted_question_cfg(
        self,
        question_cfg: Dict[str, Any],
        sensor_info: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if sensor_info.get("is_ego"):
            return None, question_cfg
        sender_id = int(sensor_info.get("sender_id", -1))
        question_id = str(question_cfg.get("id", "unknown_question"))
        bucket = self._get_vlm_step_cache_bucket("converted_queries")
        cache_key = (sender_id, question_id)
        if bucket is not None and cache_key in bucket:
            converted_question_cfg = dict(bucket[cache_key])
        else:
            converted = compute_query_direction_from_observer(
                ego_pose=ego_pose,
                observer_pose=sensor_info.get("pose"),
                question_id=question_id,
            )
            converted_question_cfg = {
                "id": question_id,
                "type": question_cfg["type"],
                "query": converted.query,
                "positive": converted.positive,
                "negative": converted.negative,
                "direction": str(getattr(converted, "converted_direction", "")).strip().lower(),
            }
            if bucket is not None:
                bucket[cache_key] = dict(converted_question_cfg)
        return converted_question_cfg, converted_question_cfg

    def _get_question_cfgs_for_sensor(
        self,
        question_cfgs: List[Dict[str, Any]],
        sensor_info: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[Dict[str, Any]]]]:
        sensor_question_cfgs: List[Dict[str, Any]] = []
        converted_by_id: Dict[str, Optional[Dict[str, Any]]] = {}
        for question_cfg in question_cfgs:
            converted_question_cfg, question_for_sensor = self._get_converted_question_cfg(
                question_cfg,
                sensor_info,
                ego_pose,
            )
            question_id = str(question_cfg["id"])
            sensor_question_cfgs.append(question_for_sensor)
            converted_by_id[question_id] = converted_question_cfg
        return sensor_question_cfgs, converted_by_id

    def _evaluate_sensor_questions(
        self,
        sensor_info: Dict[str, Any],
        question_cfgs: List[Dict[str, Any]],
        ego_pose: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        sensor_question_cfgs, converted_by_id = self._get_question_cfgs_for_sensor(
            question_cfgs,
            sensor_info,
            ego_pose,
        )
        scene_description = str(sensor_info.get("scene_description", "")).strip()
        language_evidence = self._compose_language_evidence(sensor_info)
        image = sensor_info.get("image")
        sender_id = int(sensor_info.get("sender_id", -1))

        if image is not None:
            if not scene_description:
                try:
                    scene_description = self._compute_single_image_description(
                        image,
                        cache_key=("scene_description", sender_id),
                    )
                except Exception as exc:
                    failure_reason = (
                        f"scene_description generation failed for sender_id={sender_id}: {exc}"
                    )
                    scores_by_id = {
                        str(question_cfg["id"]): self._default_question_score(
                            reason=failure_reason,
                            raw_text="",
                            raw_vlm_json={"caption_error": str(exc)},
                        )
                        for question_cfg in sensor_question_cfgs
                    }
                    evaluation_mode = "caption_failure_safe_default"
                    language_evidence = ""
                else:
                    language_evidence = self._compose_caption_only_evidence(scene_description)
                    scores_by_id = self._score_multi_questions_from_language_evidence(
                        language_evidence,
                        sensor_question_cfgs,
                    )
                    evaluation_mode = "caption_then_language_multi_query"
            else:
                language_evidence = self._compose_caption_only_evidence(scene_description)
                scores_by_id = self._score_multi_questions_from_language_evidence(
                    language_evidence,
                    sensor_question_cfgs,
                )
                evaluation_mode = "caption_then_language_multi_query"
        elif language_evidence:
            if bool(getattr(self, "_vlm_enable_multi_query_scoring", True)):
                scores_by_id = self._score_multi_questions_from_language_evidence(
                    language_evidence,
                    sensor_question_cfgs,
                )
                evaluation_mode = "language_multi_query"
            else:
                scores_by_id = {
                    str(question_cfg["id"]): self._score_question_from_language_evidence(
                        question_cfg,
                        language_evidence,
                    )
                    for question_cfg in sensor_question_cfgs
                }
                evaluation_mode = "language_fallback"
        else:
            scores_by_id = {
                str(question_cfg["id"]): self._default_question_score(
                    reason="No image or textual evidence available for this sensor.",
                )
                for question_cfg in sensor_question_cfgs
            }
            evaluation_mode = "empty_evidence"

        sensor_records: Dict[str, Dict[str, Any]] = {}
        for question_cfg in question_cfgs:
            question_id = str(question_cfg["id"])
            scoring = dict(
                scores_by_id.get(question_id)
                or self._default_question_score(
                    reason=f"Missing score for question_id={question_id}.",
                )
            )
            scoring = self._apply_scene_description_consistency_override(
                question_cfg,
                sensor_info,
                scene_description,
                scoring,
                converted_by_id.get(question_id),
            )
            used_for_aggregation, skip_reason = self._should_use_sensor_for_question(
                question_cfg,
                sensor_info,
                scoring,
            )
            sensor_records[question_id] = self._build_sensor_score_record(
                sensor_info,
                scene_description,
                language_evidence,
                scoring,
                converted_by_id.get(question_id),
                evaluation_mode,
                used_for_aggregation,
                skip_reason,
            )
        return sensor_records

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
                "converted_positive": str(items[-1].get("converted_positive", "")),
                "converted_negative": str(items[-1].get("converted_negative", "")),
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
        logits: List[float] = []
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
            norm = angle_diff_target / max(half_fov, 1e-6)
            if abs(norm) <= 1.0:
                fov_alignment = 0.5 * (1.0 + math.cos(0.5 * norm * math.pi))
            else:
                fov_alignment = 0.0

            if sensor_score["is_ego"]:
                region_base = 1.0
                distance_alignment = 1.0
            else:
                region_base = 0.5 * (
                    1.0
                    + math.cos(
                        self._wrap_angle(float(feats["bearing_from_ego_rad"]) - target_angle)
                    )
                )
                distance_alignment = math.exp(-float(feats["distance_m"]) / tau)

            region_alignment = region_base * fov_alignment
            facing_alignment = fov_alignment

            ego_bias = (
                float(self._vlm_importance_ego_bias)
                if sensor_score["is_ego"]
                else 0.0
            )
            logit = (
                float(self._vlm_importance_region_weight) * float(region_alignment)
                + float(self._vlm_importance_facing_weight) * float(facing_alignment)
                + float(self._vlm_importance_distance_weight) * float(distance_alignment)
                + ego_bias
            )

            sensor_keys.append(sensor_key)
            logits.append(float(logit))
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
                "raw_logit": float(logit),
            }

        if logits:
            max_logit = max(logits)
            exps = [math.exp(l - max_logit) for l in logits]
            denom = sum(exps) or 1.0
            weights = [e / denom for e in exps]
        else:
            weights = []
        for sensor_key, w in zip(sensor_keys, weights):
            details[sensor_key]["importance_weight"] = float(w)

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
        return [{
            "sender_id": int(self.ego.id),
            "sensor_name": "cam0",
            "image": ego_image,
            "img_emb": None,
            "scene_description": "",
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

    def _build_question_result_from_sensor_records(
        self,
        question_cfg: Dict[str, Any],
        per_sensor_instance_scores: List[Dict[str, Any]],
        shared_images: List[Image.Image],
        shared_meta: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Dict[str, Any]:
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
            "converted_positive": converted_question_cfg["positive"] if converted_question_cfg else None,
            "converted_negative": converted_question_cfg["negative"] if converted_question_cfg else None,
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
            sensor_records = self._evaluate_sensor_questions(sensor_info, [question_cfg], ego_pose)
            per_sensor_instance_scores.append(sensor_records[str(question_cfg["id"])])

        return self._build_question_result_from_sensor_records(
            question_cfg,
            per_sensor_instance_scores,
            shared_images,
            shared_meta,
            ego_pose,
        )

    def _evaluate_all_questions_multi(
        self,
        question_cfgs: List[Dict[str, Any]],
        ego_image: Image.Image,
        shared_images: List[Image.Image],
        shared_infos: List[Dict[str, Any]],
        shared_meta: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        ego_sensor_infos = self._build_ego_sensor_instances(ego_image)
        shared_sensor_infos = self._build_shared_sensor_instances(shared_infos)
        all_sensor_infos = ego_sensor_infos + shared_sensor_infos

        ego_tf = self.ego.get_transform()
        ego_pose = {
            "x": float(ego_tf.location.x),
            "y": float(ego_tf.location.y),
            "yaw": float(ego_tf.rotation.yaw),
        }
        per_question_scores: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sensor_info in all_sensor_infos:
            sensor_records = self._evaluate_sensor_questions(sensor_info, question_cfgs, ego_pose)
            for question_cfg in question_cfgs:
                question_id = str(question_cfg["id"])
                per_question_scores[question_id].append(sensor_records[question_id])

        return {
            str(question_cfg["id"]): self._build_question_result_from_sensor_records(
                question_cfg,
                per_question_scores[str(question_cfg["id"])],
                shared_images,
                shared_meta,
                ego_pose,
            )
            for question_cfg in question_cfgs
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

        try:
            question_results = self._evaluate_all_questions_multi(
                self._vlm_questions,
                ego_image,
                shared_images,
                shared_infos,
                shared_meta,
            )
        except Exception as exc:
            VLM_SCORING_LOGGER.exception(
                "VLM multi-query evaluation failed step=%d; falling back to single-question safe mode.",
                int(self._time_step),
            )
            question_results = {}
            eval_result["fallback_used"] = True
            eval_result["fallback_error"] = str(exc)
            eval_result["fallback_mode"] = "single_question_safe_mode"
        for qcfg in self._vlm_questions:
            question_id = str(qcfg["id"])
            qres = question_results.get(question_id)
            if qres is not None:
                eval_result["questions"][question_id] = qres
                self._vlm_records.append({"step": int(self._time_step), **qres})

        missing_question_ids = [
            str(qcfg["id"])
            for qcfg in self._vlm_questions
            if str(qcfg["id"]) not in eval_result["questions"]
        ]
        for question_id in missing_question_ids:
            qcfg = next(q for q in self._vlm_questions if str(q["id"]) == question_id)
            try:
                qres = self._evaluate_single_question(
                    qcfg,
                    ego_image,
                    shared_images,
                    shared_infos,
                    shared_meta,
                )
                eval_result["questions"][question_id] = qres
                self._vlm_records.append({"step": int(self._time_step), **qres})
            except Exception as exc:
                VLM_SCORING_LOGGER.exception(
                    "VLM single-question evaluation failed step=%d question_id=%s",
                    int(self._time_step),
                    qcfg["id"],
                )
                eval_result["questions"][question_id] = {
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
