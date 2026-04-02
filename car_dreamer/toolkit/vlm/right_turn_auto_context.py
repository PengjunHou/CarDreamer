from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from runtime_logging import get_runtime_logger, get_runtime_logging_config, should_log_periodic

from .right_turn_auto_prompts import RightTurnAutoVLMPromptMixin


VLM_CONTEXT_LOGGER = get_runtime_logger("car_dreamer.vlm.context")


class RightTurnAutoVLMContextMixin(RightTurnAutoVLMPromptMixin):
    """
    Helpers for assembling ego/shared VLM inputs.

    Host contract:
    - runtime state: `ego`, `group_vehs`, `group_obs`, `obs`, `_received`, `_time_step`
    - VLM config: `_vlm_*`, `feature_size`
    - communication messages store `sender_id/created_step/deliver_step/payload`
    """

    def _get_fixed_dt(self) -> float:
        return float(self._config.world.fixed_delta_seconds)

    def _window_steps_from_seconds(self, window_s: float) -> int:
        return max(int(math.floor(window_s / max(self._get_fixed_dt(), 1e-6))), 0)

    def _get_received_messages_in_window(self, receiver_id: int, window_s: float) -> List[Any]:
        msgs = list(self._received.get(int(receiver_id), deque()))
        if not msgs:
            return []
        current_step = int(self._time_step)
        window_steps = self._window_steps_from_seconds(window_s)
        return [msg for msg in msgs if (current_step - int(msg.deliver_step)) <= window_steps]

    def _group_messages_by_sender(self, msgs: List[Any]) -> Dict[int, List[Any]]:
        grouped: Dict[int, List[Any]] = defaultdict(list)
        for msg in msgs:
            grouped[int(msg.sender_id)].append(msg)
        return grouped

    def _sample_messages(self, msgs: List[Any], max_k: int, strategy: str) -> List[Any]:
        if max_k <= 0 or not msgs:
            return []
        msgs = sorted(msgs, key=lambda m: (int(m.deliver_step), int(m.created_step)))
        if len(msgs) <= max_k:
            return msgs
        if strategy == "latest":
            return msgs[-max_k:]
        if max_k == 1:
            return [msgs[-1]]
        idxs = np.linspace(0, len(msgs) - 1, num=max_k, dtype=int).tolist()
        return [msgs[i] for i in idxs]

    def _cap_total_shared_infos(
        self,
        per_sender_infos: Dict[int, List[Dict[str, Any]]],
        max_total: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        if max_total <= 0:
            return {}
        capped: Dict[int, List[Dict[str, Any]]] = {}
        total = 0
        for sender_id in sorted(per_sender_infos.keys()):
            if total >= max_total:
                break
            remain = max_total - total
            selected = per_sender_infos[sender_id][:remain]
            if selected:
                capped[sender_id] = selected
                total += len(selected)
        return capped

    def _normalize_pose_dict(self, pose: Optional[Dict[str, Any]]) -> Dict[str, float]:
        pose = pose or {}
        return {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
        }

    def _normalize_velocity_dict(self, vel: Optional[Dict[str, Any]]) -> Dict[str, float]:
        vel = vel or {}
        return {"vx": float(vel.get("vx", 0.0)), "vy": float(vel.get("vy", 0.0))}

    def _make_shared_info_from_actor(self, actor: Any, image: Image.Image) -> Dict[str, Any]:
        tf = actor.get_transform()
        vel = actor.get_velocity()
        return {
            "sender_id": int(actor.id),
            "sensor_name": "cam0",
            "pose": {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "yaw": float(tf.rotation.yaw),
            },
            "vel": {"vx": float(vel.x), "vy": float(vel.y)},
            "received_age_s": 0.0,
            "deliver_step": int(self._time_step),
            "created_step": int(self._time_step),
            "image": image,
            "img_emb": None,
            "scene_description": "",
            "text": "",
        }

    def _make_shared_info_from_message(
        self,
        msg: Any,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = payload if isinstance(payload, dict) else {}
        if not payload:
            return None
        received_age_steps = max(int(self._time_step) - int(msg.created_step), 0)
        received_age_s = received_age_steps * self._get_fixed_dt()
        return {
            "sender_id": int(msg.sender_id),
            "sensor_name": str(payload.get("sensor_name", "cam0")),
            "pose": self._normalize_pose_dict(payload.get("pose")),
            "vel": self._normalize_velocity_dict(payload.get("vel")),
            "received_age_s": float(received_age_s),
            "deliver_step": int(msg.deliver_step),
            "created_step": int(msg.created_step),
            "image": None,
            "img_emb": payload.get("img_emb"),
            "scene_description": str(payload.get("scene_description", "")).strip(),
            "text": str(payload.get("text", "")).strip(),
            "feat": payload.get("feat"),
            "feat_dim": int(payload.get("feat_dim", 0)),
        }

    def _get_raw_shared_images_info(self) -> Tuple[List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        shared_infos: List[Dict[str, Any]] = []
        for actor in self.group_vehs:
            obs = self.group_obs.get(int(actor.id), {})
            image_np = obs.get(self._vlm_image_obs_key)
            image = self._coerce_to_pil_image(image_np)
            if image is None:
                continue
            info = self._make_shared_info_from_actor(actor, image)
            info["scene_description"] = self._compute_single_image_description(
                image, self.feature_size
            )
            shared_infos.append(info)
            if len(shared_infos) >= self._vlm_max_total_shared_images:
                break
        shared_images = [info["image"] for info in shared_infos]
        meta = {
            "shared_source": "raw",
            "num_candidate_msgs": 0,
            "num_selected_shared_images": len(shared_infos),
            "selected_sender_ids": [info["sender_id"] for info in shared_infos],
            "window_s": 0.0,
            "sampling_strategy": "raw_current",
        }
        return shared_images, shared_infos, meta

    def _get_received_shared_images_info(
        self,
    ) -> Tuple[List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        receiver_id = int(self.ego.id)
        window_msgs = self._get_received_messages_in_window(receiver_id, self._vlm_received_window_s)
        grouped = self._group_messages_by_sender(window_msgs)
        per_sender_infos: Dict[int, List[Dict[str, Any]]] = {}

        for sender_id, msgs in grouped.items():
            msgs = sorted(msgs, key=lambda m: (int(m.deliver_step), int(m.created_step)))
            if self._vlm_max_msgs_per_sender > 0:
                msgs = msgs[-self._vlm_max_msgs_per_sender :]
            chosen_msgs = self._sample_messages(
                msgs,
                self._vlm_max_images_per_sender_for_inference,
                self._vlm_sampling_strategy,
            )
            infos: List[Dict[str, Any]] = []
            for msg in chosen_msgs:
                info = self._make_shared_info_from_message(msg, msg.payload)
                if info is None:
                    continue
                if info.get("scene_description") or info.get("img_emb") is not None or info.get("text"):
                    infos.append(info)
            if infos:
                per_sender_infos[int(sender_id)] = infos

        per_sender_infos = self._cap_total_shared_infos(
            per_sender_infos, self._vlm_max_total_shared_images
        )
        shared_infos: List[Dict[str, Any]] = []
        for sender_id in sorted(per_sender_infos.keys()):
            shared_infos.extend(per_sender_infos[sender_id])
        meta = {
            "shared_source": "received_feat",
            "num_candidate_msgs": len(window_msgs),
            "num_selected_shared_images": len(shared_infos),
            "selected_sender_ids": [info["sender_id"] for info in shared_infos],
            "window_s": self._vlm_received_window_s,
            "sampling_strategy": self._vlm_sampling_strategy,
        }
        runtime_cfg = get_runtime_logging_config()
        if should_log_periodic(
            int(self._time_step),
            int(runtime_cfg["step_debug_interval"]),
            logger=VLM_CONTEXT_LOGGER,
        ):
            per_sender_counts = {sid: len(items) for sid, items in per_sender_infos.items()}
            VLM_CONTEXT_LOGGER.debug(
                "Shared image selection step=%d shared_source=%s candidate_msgs=%d selected=%d sender_counts=%s",
                self._time_step,
                meta["shared_source"],
                meta["num_candidate_msgs"],
                meta["num_selected_shared_images"],
                per_sender_counts,
            )
        return [info["image"] for info in shared_infos], shared_infos, meta

    def _get_ego_and_shared_images_info(
        self,
    ) -> Tuple[Optional[Image.Image], List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        ego_image = self._coerce_to_pil_image(self.obs.get(self._vlm_image_obs_key))
        if self._vlm_shared_source == "raw":
            shared_images, shared_infos, meta = self._get_raw_shared_images_info()
        else:
            shared_images, shared_infos, meta = self._get_received_shared_images_info()
        return ego_image, shared_images, shared_infos, meta
