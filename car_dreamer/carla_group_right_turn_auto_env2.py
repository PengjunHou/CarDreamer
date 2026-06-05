from __future__ import annotations

import math
import json
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import carla
from agents.navigation.basic_agent import BasicAgent

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import _dist_m
from .toolkit import NetResource, V2VMessage, LatencyModel, SimpleWirelessLatency, _tx_bytes_for_latency
from .toolkit import Observer, payload_fn_cnn
from .toolkit import get_vehicle_pos
from .toolkit import VehicleNodeGraphBuilder, GraphBuildConfig


class CarlaGroupRightTurnAutoEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn right) and avoid collision.

    **Provided Tasks**: ``carla_right_turn_simple``, ``carla_right_turn_medium``, ``carla_right_turn_hard``
    """

    def __init__(self, config):
        print("[CARLA Group Right Turn Auto Env] Initializing environment with config:")
        super().__init__(config)

        # Initialize car flow
        self.groups = {}
        self.group_vehs: List[carla.Actor] = []
        self.num_group_vehs = int(getattr(self._config, "num_group_vehs", 2))
        self._other_observers = {}
        self.group_obs = {}
        self._prev_action = None

        # --- Communication parameters ---
        comm_cfg = getattr(self._config, "communication", None)
        self.group_update_period = int(getattr(comm_cfg, "group_update_period", 20))
        self.comm_period = int(getattr(comm_cfg, "comm_period", 5))
        bandwidth_hz = float(getattr(comm_cfg, "bandwidth_hz", 10e6))
        overhead_base_s = float(getattr(comm_cfg, "overhead_base_s", 0.030))
        overhead_per_kb_s = float(getattr(comm_cfg, "overhead_per_kb_s", 0.0015))
        pathloss_model = str(getattr(comm_cfg, "pathloss_model", "urban_los"))
        margin_db = float(getattr(comm_cfg, "margin_db", 10.0))
        margin_sigma_db = float(getattr(comm_cfg, "margin_sigma_db", 0.0))
        jitter_s = float(getattr(comm_cfg, "jitter_s", 0.0))
        overhead_bytes = int(getattr(comm_cfg, "overhead_bytes", 64))

        self._default_net_res = NetResource(bandwidth_hz=bandwidth_hz)
        self.latency_model: LatencyModel = SimpleWirelessLatency(
            overhead_base_s=overhead_base_s,
            overhead_per_kb_s=overhead_per_kb_s,
            pathloss_model=pathloss_model,
            margin_db=margin_db,
            margin_sigma_db=margin_sigma_db,
            jitter_s=jitter_s,
            overhead_bytes=overhead_bytes,
        )

        self.payload_fn = payload_fn_cnn

        # comm buffers
        self._in_flight: List[V2VMessage] = []
        self._received: Dict[int, Deque[V2VMessage]] = defaultdict(lambda: deque(maxlen=256))

        # Per-vehicle network resources
        self._veh_net_res: Dict[int, NetResource] = {}

        # feature size
        self.feature_size = int(getattr(self._config, "feature_size", 1024))

        # graph builder
        graph_cfg = getattr(self._config, "graph", None)
        cfg = GraphBuildConfig(
            window_s=float(getattr(graph_cfg, "window_s", 2.0)),
            Tmax=int(getattr(graph_cfg, "tmax", 15)),
            max_nodes=int(getattr(graph_cfg, "max_nodes", 3)),
            feat_dim_max=int(getattr(graph_cfg, "feat_dim_max", 1024)),
            star_graph=bool(getattr(graph_cfg, "star_graph", True)),
        )
        self._graph_builder = VehicleNodeGraphBuilder(cfg)

        # -----------------------------
        # VLM configuration
        # -----------------------------
        vlm_cfg = getattr(self._config, "vlm", None)
        self._vlm_enabled = bool(getattr(vlm_cfg, "enabled", True))
        self._vlm_model_name = str(getattr(vlm_cfg, "model_name", "Qwen/Qwen2-VL-7B-Instruct"))
        self._vlm_image_template = str(getattr(vlm_cfg, "image_template", "Analyze the driving scene."))
        self._vlm_eval_period = int(getattr(vlm_cfg, "eval_period", 1))
        self._vlm_image_obs_key = str(getattr(vlm_cfg, "image_obs_key", "camera"))
        self._vlm_local_files_only = bool(getattr(vlm_cfg, "local_files_only", False))
        self._vlm_shared_source = str(getattr(vlm_cfg, "shared_source", "received_feat"))  # received_feat | raw
        self._vlm_received_window_s = float(getattr(vlm_cfg, "received_window_s", 1.0))
        self._vlm_max_msgs_per_sender = int(getattr(vlm_cfg, "max_msgs_per_sender", 20))
        self._vlm_max_images_per_sender_for_inference = int(
            getattr(vlm_cfg, "max_images_per_sender_for_inference", 4)
        )
        self._vlm_sampling_strategy = str(getattr(vlm_cfg, "sampling_strategy", "latest"))  # uniform | latest
        self._vlm_max_total_shared_images = int(getattr(vlm_cfg, "max_total_shared_images", 12))
        self._vlm_age_decay_tau_s = float(getattr(vlm_cfg, "age_decay_tau_s", 0.5))
        self._vlm_semantic_ema_alpha = float(getattr(vlm_cfg, "semantic_ema_alpha", 0.4))

        # sender-level confidence weights
        self._vlm_ego_conf_weight = float(getattr(vlm_cfg, "ego_conf_weight", 1.0))
        default_shared_weight = float(getattr(vlm_cfg, "shared_conf_weight", 1.0))
        shared_weights_cfg = getattr(vlm_cfg, "shared_weights", None)
        self._vlm_default_shared_conf_weight = default_shared_weight
        self._vlm_shared_conf_weights: Dict[int, float] = {}
        self._vlm_sensor_fov_deg = float(getattr(vlm_cfg, "sensor_fov_deg", 90.0)) #TODO
        if shared_weights_cfg is not None:
            try:
                self._vlm_shared_conf_weights = {
                    int(k): float(v) for k, v in dict(shared_weights_cfg).items()
                }
            except Exception:
                self._vlm_shared_conf_weights = {}

        # sensor-level importance / SC-part2 parameters
        self._vlm_importance_distance_tau = float(getattr(vlm_cfg, "importance_distance_tau", 20.0))
        self._vlm_importance_region_weight = float(getattr(vlm_cfg, "importance_region_weight", 1.0))
        self._vlm_importance_facing_weight = float(getattr(vlm_cfg, "importance_facing_weight", 1.0))
        self._vlm_importance_distance_weight = float(getattr(vlm_cfg, "importance_distance_weight", 1.0))
        self._vlm_importance_ego_bias = float(getattr(vlm_cfg, "importance_ego_bias", 0.0))
        self._vlm_sc_beta = float(getattr(vlm_cfg, "sc_beta", 1.0))

        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_records: List[Dict[str, Any]] = []
        self._vlm_last_eval: Dict[str, Any] = {}
        self._vlm_questions = self._build_vlm_questions()
        self._text_embedding_cache: Dict[str, torch.Tensor] = {}

        if self._vlm_enabled:
            self._init_vlm()

    # =========================================================
    # VLM init / questions / embeddings
    # =========================================================

    def _init_vlm(self) -> None:
        print(f"[VLM] Loading model: {self._vlm_model_name}")
        self._vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._vlm_model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=self._vlm_local_files_only,
        )
        self._vlm_model.eval()
        self._vlm_processor = AutoProcessor.from_pretrained(
            self._vlm_model_name,
            local_files_only=self._vlm_local_files_only,
        )
        print("[VLM] Model loaded.")

    def _build_vlm_questions(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "clg_left_rear_vehicle",
                "type": "clg",
                "positive": "There were no vehicles to the left rear of the ego vehicle.",
                "negative": "There were vehicles to the left rear of the ego vehicle.",
            },
            {
                "id": "clg_right_rear_vehicle",
                "type": "clg",
                "positive": "There were no vehicles to the right rear of the ego vehicle.",
                "negative": "There were vehicles to the right rear of the ego vehicle.",
            },
            {
                "id": "clg_right_front_vehicle",
                "type": "clg",
                "positive": "There were no vehicles to the right front of the ego vehicle.",
                "negative": "There were vehicles to the right front of the ego vehicle.",
            },
            {
                "id": "clg_left_front_vehicle",
                "type": "clg",
                "positive": "There were no vehicles to the left front of the ego vehicle.",
                "negative": "There were vehicles to the left front of the ego vehicle.",
            },
            {
                "id": "clg_left_vehicle_speed",
                "type": "clg",
                "positive": "The left-side vehicles were either far away or moving slowly, posing no risk to the ego vehicle's right turn.",
                "negative": "There were vehicles on the left side that were close or moving fast, posing a potential risk to the ego vehicle's right turn.",
            },
        ]

    def _masked_mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
        return pooled.squeeze(0)

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

    def _normalize_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.detach().float()
        return F.normalize(emb, p=2, dim=-1)

    def _build_position_embedding_tensor(
        self,
        sensor_pose: Dict[str, float],
        ego_pose: Dict[str, float],
        device: torch.device,
    ) -> torch.Tensor:
        dx = float(sensor_pose.get("x", 0.0)) - float(ego_pose.get("x", 0.0))
        dy = float(sensor_pose.get("y", 0.0)) - float(ego_pose.get("y", 0.0))
        yaw = math.radians(float(sensor_pose.get("yaw", 0.0)))

        d = math.sqrt(dx * dx + dy * dy) + 1e-6

        feat = torch.tensor([
            dx, dy, d,
            dx / d, dy / d,
            math.sin(yaw), math.cos(yaw)
        ], dtype=torch.float32, device=device)

        return feat


    def _fuse_image_with_position(
        self,
        image_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        gamma: float = 0.3,
    ) -> torch.Tensor:

        if pos_emb.shape[0] != image_emb.shape[0]:
            repeat = image_emb.shape[0] // pos_emb.shape[0] + 1
            pos_emb = pos_emb.repeat(repeat)[:image_emb.shape[0]]

        fused = image_emb + gamma * pos_emb
        return F.normalize(fused, p=2, dim=-1)

    def _compute_text_embedding(self, text: str) -> torch.Tensor:
        if text in self._text_embedding_cache:
            return self._text_embedding_cache[text].clone()

        if self._vlm_model is None or self._vlm_processor is None:
            raise RuntimeError("VLM is not initialized.")

        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt_text = self._vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self._vlm_processor(text=[prompt_text], return_tensors="pt")
        model_device = next(self._vlm_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._vlm_model(**inputs, output_hidden_states=True, return_dict=True)

        last_hidden = outputs.hidden_states[-1]
        pooled = self._masked_mean_pool(last_hidden, inputs["attention_mask"])
        pooled = self._normalize_embedding(pooled).cpu()
        self._text_embedding_cache[text] = pooled
        return pooled.clone()

    def _compute_single_image_embedding(self, image: Image.Image) -> torch.Tensor:
        if self._vlm_model is None or self._vlm_processor is None:
            raise RuntimeError("VLM is not initialized.")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._vlm_image_template},
            ],
        }]
        prompt_text = self._vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self._vlm_processor(text=[prompt_text], images=[image], return_tensors="pt")
        model_device = next(self._vlm_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._vlm_model(**inputs, output_hidden_states=True, return_dict=True)

        last_hidden = outputs.hidden_states[-1]
        pooled = self._masked_mean_pool(last_hidden, inputs["attention_mask"])
        return self._normalize_embedding(pooled).cpu()

    def _compute_similarity_score(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> float:
        sim = F.cosine_similarity(image_emb.unsqueeze(0), text_emb.unsqueeze(0), dim=-1).item()
        score = 0.5 * (sim + 1.0)
        return float(max(0.0, min(1.0, score)))

    def _compute_clg_scores_from_embeddings(
        self,
        image_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> Dict[str, float]:
        positive_score = self._compute_similarity_score(image_emb, positive_emb)
        negative_score = self._compute_similarity_score(image_emb, negative_emb)
        confidence = abs(positive_score - negative_score)
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "confidence": confidence,
        }

    def _aggregate_sender_scores(self, per_sender_scores: Dict[int, List[Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
        aggregated: Dict[int, Dict[str, float]] = {}
        for sender_id, scores in per_sender_scores.items():
            if not scores:
                continue
            n = float(len(scores))
            aggregated[sender_id] = {
                "positive_score": float(sum(s["positive_score"] for s in scores) / n),
                "negative_score": float(sum(s["negative_score"] for s in scores) / n),
                "confidence": float(sum(s["confidence"] for s in scores) / n),
                "num_images": int(len(scores)),
            }
        return aggregated


    def _wrap_angle(self, angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def _question_target_angle_rad(self, question_cfg: Dict[str, Any]) -> float:
        """
        Return the target angle in the ego-local frame.
        0     : front
        +pi/2 : left
        -pi/2 : right
        +/-pi : rear
        """
        qid = str(question_cfg.get("id", "")).lower()
        if "left_rear" in qid:
            return 3.0 * math.pi / 4.0
        if "right_rear" in qid:
            return -3.0 * math.pi / 4.0
        if "right_front" in qid:
            return -math.pi / 4.0
        if "left_front" in qid:
            return math.pi / 4.0
        if "left_vehicle_speed" in qid or "left_side" in qid:
            return math.pi / 2.0
        return 0.0


    def _question_target_angle_global_rad(
        self,
        question_cfg: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> float:
        """
        Convert the ego-local target angle into the global frame.

        bearing_from_ego_rad is currently computed in the global frame:
            atan2(sensor_y - ego_y, sensor_x - ego_x)

        So the target angle used for region_alignment must also be in the
        global frame, otherwise we compare angles from different frames.
        """
        target_local = self._question_target_angle_rad(question_cfg)
        ego_yaw_rad = math.radians(float(ego_pose.get("yaw", 0.0)))
        return self._wrap_angle(ego_yaw_rad + target_local)

    def _build_ego_sensor_instances(self, ego_image: Image.Image) -> List[Dict[str, Any]]:
        tf = self.ego.get_transform()
        yaw_rad = math.radians(float(tf.rotation.yaw))
        return [{
            "sender_id": int(self.ego.id),
            "sensor_name": "cam0",
            "image": ego_image,
            "pose": {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "yaw": float(tf.rotation.yaw),
            },
            "sensor_yaw_rad": yaw_rad,
            "received_age_s": 0.0,
            "is_ego": True,
        }]

    def _build_shared_sensor_instances(self, shared_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sensor_infos: List[Dict[str, Any]] = []
        for info in shared_infos:
            pose = info.get("pose", {})
            yaw_deg = float(pose.get("yaw", 0.0))
            sensor_infos.append({
                "sender_id": int(info["sender_id"]),
                "sensor_name": str(info.get("sensor_name", "cam0")),
                "image": info["image"],
                "pose": {
                    "x": float(pose.get("x", 0.0)),
                    "y": float(pose.get("y", 0.0)),
                    "yaw": yaw_deg,
                },
                "sensor_yaw_rad": math.radians(yaw_deg),
                "received_age_s": float(info.get("received_age_s", 0.0)),
                "is_ego": False,
            })
        return sensor_infos

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
            sender_id = int(items[0]["sender_id"])
            sensor_name = str(items[0]["sensor_name"])
            aggregated[key] = {
                "sensor_key": key,
                "sender_id": sender_id,
                "sensor_name": sensor_name,
                "is_ego": bool(items[0].get("is_ego", False)),
                "positive_score": float(sum(x["positive_score"] for x in items) / n),
                "negative_score": float(sum(x["negative_score"] for x in items) / n),
                "confidence": float(sum(x["confidence"] for x in items) / n),
                "num_images": int(len(items)),
                "received_age_s_mean": float(sum(float(x.get("received_age_s", 0.0)) for x in items) / n),
                "pose": dict(items[0].get("pose", {})),
                "sensor_yaw_rad": float(items[0].get("sensor_yaw_rad", 0.0)),
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
                "confidence": float(sum(x["confidence"] for x in items) / n),
                "num_sensors": int(len(items)),
            }
        return aggregated

    def _build_sensor_position_features(
        self,
        sensor_score: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Dict[str, float]:
        sx = float(sensor_score.get("pose", {}).get("x", 0.0))
        sy = float(sensor_score.get("pose", {}).get("y", 0.0))
        dx = sx - float(ego_pose.get("x", 0.0))
        dy = sy - float(ego_pose.get("y", 0.0))
        d = math.sqrt(dx * dx + dy * dy)
        bearing_from_ego = math.atan2(dy, dx) if d > 1e-6 else 0.0
        bearing_to_ego = math.atan2(-dy, -dx) if d > 1e-6 else 0.0
        psi = float(sensor_score.get("sensor_yaw_rad", 0.0))
        return {
            "dx": float(dx),
            "dy": float(dy),
            "distance_m": float(d),
            "bearing_from_ego_rad": float(bearing_from_ego),
            "bearing_to_ego_rad": float(bearing_to_ego),
            "sin_theta": float(math.sin(bearing_from_ego)),
            "cos_theta": float(math.cos(bearing_from_ego)),
            "sensor_yaw_rad": float(psi),
            "sin_psi": float(math.sin(psi)),
            "cos_psi": float(math.cos(psi)),
        }

    def _compute_sensor_importance_maps(
        self,
        question_cfg: Dict[str, Any],
        sensor_mean_scores: Dict[str, Dict[str, Any]],
        ego_pose: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        target_angle = self._question_target_angle_global_rad(question_cfg, ego_pose)
        tau = max(float(self._vlm_importance_distance_tau), 1e-6)
        logits: List[float] = []
        sensor_keys: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        for sensor_key, sensor_score in sensor_mean_scores.items():
            feats = self._build_sensor_position_features(sensor_score, ego_pose)
            d = feats["distance_m"]

            if d <= 1e-6:
                region_alignment = 1.0
            else:
                region_alignment = 0.5 * (
                    1.0 + math.cos(
                        self._wrap_angle(feats["bearing_from_ego_rad"] - target_angle)
                    )
                )

            fov_alignment = self._compute_fov_alignment(
                sensor_yaw_rad=feats["sensor_yaw_rad"],
                target_angle_rad=target_angle,
                fov_deg=self._vlm_sensor_fov_deg,
            )

            distance_alignment = math.exp(-d / tau)

            region_alignment = region_alignment * fov_alignment
            facing_alignment = fov_alignment

            ego_bias = float(self._vlm_importance_ego_bias) if bool(sensor_score.get("is_ego", False)) else 0.0

            logit = (
                float(self._vlm_importance_region_weight) * region_alignment
                + float(self._vlm_importance_facing_weight) * facing_alignment
                + float(self._vlm_importance_distance_weight) * distance_alignment
                + ego_bias
            )

            logits.append(float(logit))
            sensor_keys.append(sensor_key)
            details[sensor_key] = {
                **feats,
                "target_angle_global_rad": float(target_angle),
                "region_alignment": float(region_alignment),
                "fov_alignment": float(fov_alignment),
                "facing_alignment": float(facing_alignment),
                "distance_alignment": float(distance_alignment),
                "raw_logit": float(logit),
            }

        if not sensor_keys:
            return {"per_sensor": {}, "per_sender_positive": {}, "per_sender_negative": {}}

        weights = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).tolist()
        per_sensor: Dict[str, Dict[str, Any]] = {}
        per_sender_positive: Dict[int, float] = defaultdict(float)
        per_sender_negative: Dict[int, float] = defaultdict(float)

        for sensor_key, w in zip(sensor_keys, weights):
            sensor_score = sensor_mean_scores[sensor_key]
            sender_id = int(sensor_score["sender_id"])
            per_sensor[sensor_key] = {
                "importance_positive": float(w),
                "importance_negative": float(w),
                **details[sensor_key],
            }
            per_sender_positive[sender_id] += float(w)
            per_sender_negative[sender_id] += float(w)

        return {
            "per_sensor": per_sensor,
            "per_sender_positive": {int(k): float(v) for k, v in per_sender_positive.items()},
            "per_sender_negative": {int(k): float(v) for k, v in per_sender_negative.items()},
        }

    def _compute_importance_penalty(
        self,
        sensor_mean_scores: Dict[str, Dict[str, Any]],
        importance_maps: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        per_sensor_penalty: Dict[str, float] = {}
        total_penalty = 0.0
        for sensor_key, sensor_score in sensor_mean_scores.items():
            imp = importance_maps.get("per_sensor", {}).get(sensor_key, {})
            m_pos = float(imp.get("importance_positive", 0.0))
            m_neg = float(imp.get("importance_negative", 0.0))
            s_pos = max(float(sensor_score["positive_score"]), 1e-6)
            s_neg = max(float(sensor_score["negative_score"]), 1e-6)
            penalty = m_pos * (-math.log(s_pos)) + m_neg * (-math.log(s_neg))
            per_sensor_penalty[sensor_key] = float(penalty)
            total_penalty += float(penalty)
        return {
            "beta": float(self._vlm_sc_beta),
            "unweighted_penalty": float(total_penalty),
            "weighted_penalty": float(self._vlm_sc_beta * total_penalty),
            "per_sensor_penalty": per_sensor_penalty,
        }

    def _get_sender_weight(self, sender_id: int, is_ego: bool) -> float:
        if is_ego:
            return self._vlm_ego_conf_weight
        return float(self._vlm_shared_conf_weights.get(int(sender_id), self._vlm_default_shared_conf_weight))

    def _weighted_confidence_aggregate(
        self,
        ego_score: Dict[str, float],
        sender_scores: Dict[int, Dict[str, float]],
    ) -> Dict[str, Any]:
        weighted_entries: List[Dict[str, Any]] = []

        ego_weight = self._get_sender_weight(int(self.ego.id), is_ego=True)
        weighted_entries.append({
            "sender_id": int(self.ego.id),
            "role": "ego",
            "weight": float(ego_weight),
            **ego_score,
        })

        for sender_id in sorted(sender_scores.keys()):
            w = self._get_sender_weight(sender_id, is_ego=False)
            weighted_entries.append({
                "sender_id": int(sender_id),
                "role": "shared",
                "weight": float(w),
                **sender_scores[sender_id],
            })

        total_weight = sum(item["weight"] for item in weighted_entries)
        if total_weight <= 0:
            aggregated_conf = ego_score["confidence"]
            aggregated_pos = ego_score["positive_score"]
            aggregated_neg = ego_score["negative_score"]
        else:
            aggregated_pos = sum(item["weight"] * item["positive_score"] for item in weighted_entries) / total_weight
            aggregated_neg = sum(item["weight"] * item["negative_score"] for item in weighted_entries) / total_weight
            aggregated_conf = sum(item["weight"] * item["confidence"] for item in weighted_entries) / total_weight

        return {
            "positive_score": float(aggregated_pos),
            "negative_score": float(aggregated_neg),
            "confidence": float(aggregated_conf),
            "total_weight": float(total_weight),
            "per_sender": weighted_entries,
        }

    # =========================================================
    # Image helpers
    # =========================================================

    def _get_rgb_image_from_obs(self, obs: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        if not isinstance(obs, dict):
            return None
        img = obs.get(self._vlm_image_obs_key, None)
        if img is None:
            return None

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        img = np.asarray(img)
        if img.ndim != 3:
            return None

        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] != 3:
            return None

        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        return img

    def _to_pil_image(self, img_np: np.ndarray) -> Image.Image:
        return Image.fromarray(img_np).convert("RGB")

    def _reconstruct_image_from_feat(self, feat: np.ndarray, feature_size: int) -> np.ndarray:
        img = feat
        return img

    # =========================================================
    # Shared-message selection helpers
    # =========================================================

    def _get_fixed_dt(self) -> float:
        return float(self._config.world.fixed_delta_seconds)

    def _window_steps_from_seconds(self, window_s: float) -> int:
        dt = max(self._get_fixed_dt(), 1e-6)
        return max(int(math.floor(window_s / dt)), 0)

    def _get_received_messages_in_window(self, receiver_id: int, window_s: float) -> List[V2VMessage]:
        msgs = list(self._received.get(receiver_id, deque()))
        if not msgs:
            return []

        cur = int(self._time_step)
        window_steps = self._window_steps_from_seconds(window_s)
        selected = []
        for msg in msgs:
            if (cur - int(msg.deliver_step)) <= window_steps:
                selected.append(msg)
        return selected

    def _group_messages_by_sender(self, msgs: List[V2VMessage]) -> Dict[int, List[V2VMessage]]:
        grouped: Dict[int, List[V2VMessage]] = defaultdict(list)
        for msg in msgs:
            grouped[int(msg.sender_id)].append(msg)
        return grouped

    def _sample_messages(self, msgs: List[V2VMessage], max_k: int, strategy: str) -> List[V2VMessage]:
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
            use = per_sender_infos[sender_id][:remain]
            if use:
                capped[sender_id] = use
                total += len(use)
        return capped

    def _make_shared_info_from_actor(self, actor: carla.Actor, image: Image.Image) -> Dict[str, Any]:
        tf = actor.get_transform()
        return {
            "sender_id": int(actor.id),
            "pose": {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "yaw": float(tf.rotation.yaw),
            },
            "pose_text": f"x={tf.location.x:.2f}, y={tf.location.y:.2f}, yaw={tf.rotation.yaw:.2f}",
            "received_age_s": 0.0,
            "image": image,
        }

    def _make_shared_info_from_message(self, msg: V2VMessage, image: Image.Image) -> Optional[Dict[str, Any]]:
        sender_id = int(msg.sender_id)
        actor = None
        for veh in self.group_vehs:
            if int(veh.id) == sender_id:
                actor = veh
                break
        if actor is None:
            return None

        tf = actor.get_transform()
        received_age_steps = max(int(self._time_step) - int(msg.deliver_step), 0)
        received_age_s = received_age_steps * self._get_fixed_dt()

        return {
            "sender_id": sender_id,
            "pose": {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "yaw": float(tf.rotation.yaw),
            },
            "pose_text": f"x={tf.location.x:.2f}, y={tf.location.y:.2f}, yaw={tf.rotation.yaw:.2f}",
            "received_age_s": float(received_age_s),
            "deliver_step": int(msg.deliver_step),
            "created_step": int(msg.created_step),
            "image": image,
        }

    def _get_raw_shared_images_for_vlm(self) -> Tuple[List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        shared_infos: List[Dict[str, Any]] = []

        for veh in self.group_vehs:
            obs = self.group_obs.get(veh.id, None)
            img_np = self._get_rgb_image_from_obs(obs)
            if img_np is None:
                continue
            pil_img = self._to_pil_image(img_np)
            shared_infos.append(self._make_shared_info_from_actor(veh, pil_img))
            if len(shared_infos) >= self._vlm_max_total_shared_images:
                break

        shared_images = [info["image"] for info in shared_infos]
        meta = {
            "shared_source": "raw",
            "num_candidate_msgs": 0,
            "num_selected_shared_images": len(shared_images),
            "selected_sender_ids": [info["sender_id"] for info in shared_infos],
            "window_s": 0.0,
            "sampling_strategy": "raw_current",
        }
        return shared_images, shared_infos, meta

    def _get_received_shared_images_for_vlm(self) -> Tuple[List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        receiver_id = int(self.ego.id)
        window_msgs = self._get_received_messages_in_window(receiver_id, self._vlm_received_window_s)
        num_candidate_msgs = len(window_msgs)

        grouped = self._group_messages_by_sender(window_msgs)

        per_sender_infos: Dict[int, List[Dict[str, Any]]] = {}
        for sender_id, msgs in grouped.items():
            msgs = sorted(msgs, key=lambda m: (int(m.deliver_step), int(m.created_step)))
            print(f"[VLM] Sender {sender_id} has {len(msgs)} messages in the received window.")
            if self._vlm_max_msgs_per_sender > 0:
                msgs = msgs[-self._vlm_max_msgs_per_sender:]

            chosen_msgs = self._sample_messages(
                msgs,
                self._vlm_max_images_per_sender_for_inference,
                self._vlm_sampling_strategy,
            )

            infos: List[Dict[str, Any]] = []
            for msg in chosen_msgs:
                payload = msg.payload if isinstance(msg.payload, dict) else {}
                feat = payload.get("feat", None)
                if feat is None:
                    continue
                try:
                    img_np = self._reconstruct_image_from_feat(feat, self.feature_size)
                    pil_img = self._to_pil_image(img_np)
                except Exception:
                    continue

                info = self._make_shared_info_from_message(msg, pil_img)
                if info is not None:
                    infos.append(info)

            if infos:
                per_sender_infos[sender_id] = infos

        per_sender_infos = self._cap_total_shared_infos(
            per_sender_infos,
            self._vlm_max_total_shared_images,
        )

        shared_infos: List[Dict[str, Any]] = []
        for sender_id in sorted(per_sender_infos.keys()):
            infos = sorted(
                per_sender_infos[sender_id],
                key=lambda x: (x.get("deliver_step", 0), x.get("created_step", 0)),
            )
            shared_infos.extend(infos)

        shared_images = [info["image"] for info in shared_infos]
        meta = {
            "shared_source": "received_feat",
            "num_candidate_msgs": num_candidate_msgs,
            "num_selected_shared_images": len(shared_images),
            "selected_sender_ids": [info["sender_id"] for info in shared_infos],
            "window_s": self._vlm_received_window_s,
            "sampling_strategy": self._vlm_sampling_strategy,
        }
        return shared_images, shared_infos, meta

    def _get_ego_and_shared_pil_images(self) -> Tuple[Optional[Image.Image], List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        ego_img_np = self._get_rgb_image_from_obs(self.obs)
        ego_pil = self._to_pil_image(ego_img_np) if ego_img_np is not None else None

        if self._vlm_shared_source == "raw":
            shared_images, shared_infos, meta = self._get_raw_shared_images_for_vlm()
        else:
            shared_images, shared_infos, meta = self._get_received_shared_images_for_vlm()

        return ego_pil, shared_images, shared_infos, meta

    # =========================================================
    # VLM evaluation
    # =========================================================
 
    def _evaluate_single_question(
        self,
        question_cfg: Dict[str, Any],
        ego_image: Image.Image,
        shared_images: List[Image.Image],
        shared_infos: List[Dict[str, Any]],
        shared_meta: Dict[str, Any],
    ) -> Dict[str, Any]:

        positive_text = question_cfg["positive"]
        negative_text = question_cfg["negative"]

        positive_emb = self._compute_text_embedding(positive_text)
        negative_emb = self._compute_text_embedding(negative_text)

        ego_sensor_infos = self._build_ego_sensor_instances(ego_image)
        shared_sensor_infos = self._build_shared_sensor_instances(shared_infos)
        all_sensor_infos = ego_sensor_infos + shared_sensor_infos

        # ========= NEW: ego pose =========
        ego_tf = self.ego.get_transform()
        ego_pose = {
            "x": float(ego_tf.location.x),
            "y": float(ego_tf.location.y),
            "yaw": float(ego_tf.rotation.yaw),
        }

        per_sensor_instance_scores: List[Dict[str, Any]] = []

        for sensor_info in all_sensor_infos:

            img_emb = self._compute_single_image_embedding(sensor_info["image"])

            # ========= NEW: context-conditioned =========
            pos_emb = self._build_position_embedding_tensor(
                sensor_info["pose"],
                ego_pose,
                device=img_emb.device,
            )

            img_emb = self._fuse_image_with_position(
                img_emb,
                pos_emb,
                gamma=0.3,
            )
            # ===========================================

            score = self._compute_clg_scores_from_embeddings(
                img_emb,
                positive_emb,
                negative_emb,
            )

            per_sensor_instance_scores.append({
                "sender_id": int(sensor_info["sender_id"]),
                "sensor_name": str(sensor_info["sensor_name"]),
                "is_ego": bool(sensor_info.get("is_ego", False)),
                "received_age_s": float(sensor_info.get("received_age_s", 0.0)),
                "pose": dict(sensor_info.get("pose", {})),
                "sensor_yaw_rad": float(sensor_info.get("sensor_yaw_rad", 0.0)),
                **score,
            })

        sensor_mean_scores = self._aggregate_sensor_scores(per_sensor_instance_scores)
        sender_mean_scores = self._aggregate_sender_scores_from_sensor_means(sensor_mean_scores)

        ego_score = sender_mean_scores.get(int(self.ego.id), {"positive_score": 0.0, "negative_score": 0.0, "confidence": 0.0})
        shared_sender_scores = {sid: s for sid, s in sender_mean_scores.items() if int(sid) != int(self.ego.id)}
        aggregated = self._weighted_confidence_aggregate(ego_score, shared_sender_scores)

        importance_maps = self._compute_sensor_importance_maps(question_cfg, sensor_mean_scores, ego_pose)
        importance_penalty = self._compute_importance_penalty(sensor_mean_scores, importance_maps)

        per_sensor_scores_out: List[Dict[str, Any]] = []
        for sensor_key in sorted(sensor_mean_scores.keys()):
            sensor_score = sensor_mean_scores[sensor_key]
            imp = importance_maps.get("per_sensor", {}).get(sensor_key, {})
            per_sensor_scores_out.append({
                **sensor_score,
                **imp,
                "penalty_term": float(importance_penalty["per_sensor_penalty"].get(sensor_key, 0.0)),
            })
        
        # final_confidence_score_ego_only = ego_score - 

        return {
            "question_id": question_cfg["id"],
            "question_type": question_cfg["type"],
            "shared_source": shared_meta.get("shared_source", self._vlm_shared_source),
            "num_candidate_msgs": shared_meta.get("num_candidate_msgs", 0),
            "num_selected_shared_images": shared_meta.get("num_selected_shared_images", len(shared_images)),
            "selected_sender_ids": shared_meta.get("selected_sender_ids", []),
            "received_window_s": shared_meta.get("window_s", 0.0),
            "sampling_strategy": shared_meta.get("sampling_strategy", ""),
            "positive_text": positive_text,
            "negative_text": negative_text,
            "per_sensor_scores": per_sensor_scores_out,
            "sender_mean_scores": sender_mean_scores,
            "sender_importance_positive": importance_maps.get("per_sender_positive", {}),
            "sender_importance_negative": importance_maps.get("per_sender_negative", {}),
            "ego_only": ego_score,
            "ego_plus_shared": {
                "positive_score": aggregated["positive_score"],
                "negative_score": aggregated["negative_score"],
                "confidence": aggregated["confidence"],
            },
            "aggregated_details": {
                "total_weight": aggregated["total_weight"],
                "per_sender": aggregated["per_sender"],
            },
            "sc_part1": float(aggregated["confidence"]),
            "sc_part2_penalty": float(importance_penalty["weighted_penalty"]),
            "sc_part2_details": importance_penalty,
            "confidence_gain": float(aggregated["confidence"] - ego_score.get("confidence", 0.0)),
        }

    def _evaluate_vlm_questions(self) -> Dict[str, Any]:
        if not self._vlm_enabled:
            return {}

        ego_image, shared_images, shared_infos, shared_meta = self._get_ego_and_shared_pil_images()
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
            "num_selected_shared_images": shared_meta.get("num_selected_shared_images", len(shared_images)),
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
            except Exception as e:
                eval_result["questions"][qcfg["id"]] = {
                    "question_id": qcfg["id"],
                    "question_type": qcfg["type"],
                    "error": str(e),
                }

        self._vlm_last_eval = eval_result
        return eval_result

    # =========================================================
    # Environment
    # =========================================================

    def generate_group_vehicles(self):
        self.groups.setdefault(0, set())
        self.groups[0].add(self.ego.id)
        spawn_points = self._config.group_spawn_points
        assert spawn_points is not None and len(spawn_points) >= self.num_group_vehs, (
            "Not enough spawn points for the number of group vehicles"
        )

        for spawn_point in spawn_points[: self.num_group_vehs]:
            transform = carla.Transform(
                carla.Location(*spawn_point[:3]),
                carla.Rotation(yaw=spawn_point[3]),
            )
            vehicle = self._world.spawn_actor(transform=transform)
            group_observation = self._config.group_observation
            group_vhe_observer = Observer(self._world, group_observation)
            self._other_observers.setdefault(vehicle.id, group_vhe_observer)
            self._other_observers[vehicle.id].reset(vehicle)
            self.group_obs[vehicle.id], _ = self._other_observers[vehicle.id].get_observation(self.get_state())

            self.group_vehs.append(vehicle)
            self.groups[0].add(vehicle.id)

    def on_reset(self) -> None:
        self.group_vehs = []
        self.groups = {}
        self._prev_action = None

        self._in_flight = []
        self._received = defaultdict(lambda: deque(maxlen=256))
        self._veh_net_res = {}

        self._vlm_last_eval = {}

        for obs in self._other_observers.values():
            obs.destroy()
        self._other_observers = {}
        self.group_obs = {}

        traffic_lights = self._world.carla_actors(actor_type="traffic_light")
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(9999)
            tl.set_red_time(0)
            tl.set_yellow_time(0)

        super().on_reset()
        self.ego_end = self._config.lane_end_point
        ego_transform = carla.Transform(
            carla.Location(*self.ego_end[:3]),
            carla.Rotation(yaw=self.ego_end[3]),
        )
        self.agent = BasicAgent(self.ego)
        self.agent.set_destination(ego_transform.location)
        self.generate_group_vehicles()

    def on_step(self) -> None:
        self._deliver_messages()

        for v in self.group_vehs:
            actor_id = v.id
            observer = self._other_observers.get(actor_id, None)
            if observer is not None:
                self.group_obs[actor_id], _ = observer.get_observation(self.get_state())

        if self._time_step % max(self.comm_period, 1) == 0:
            self._run_group_communication()

        if self._vlm_enabled and self._time_step % max(self._vlm_eval_period, 1) == 0:
            try:
                self._evaluate_vlm_questions()
            except Exception as e:
                self._vlm_last_eval = {
                    "step": int(self._time_step),
                    "status": "error",
                    "error": str(e),
                }

        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(self.actor_flow[0])
            if y > -81.2 or x < -38.4 or x > 31.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()

        super().on_step()

    def _make_payload(self, sender: carla.Actor) -> Any:
        if self.payload_fn is not None:
            obs = self.obs if sender.id == self.ego.id else self.group_obs.get(sender.id, {})
            return self.payload_fn(sender, obs, self.feature_size)

        tf = sender.get_transform()
        vel = sender.get_velocity()
        payload = {
            "pose": np.array(
                [tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw],
                dtype=np.float32,
            ),
            "vel": np.array([vel.x, vel.y, vel.z], dtype=np.float32),
        }
        return payload

    def _run_group_communication(self) -> None:
        if not self.groups:
            return

        out_deg: Dict[int, int] = {}
        in_deg: Dict[int, int] = {}
        for gid, members in self.groups.items():
            m = list(members)
            for sender in m:
                out_deg[sender] = max(len(m) - 1, 0)
            for receiver in m:
                in_deg[receiver] = max(len(m) - 1, 0)

        id_to_actor: Dict[int, carla.Actor] = {}
        if self.ego is not None:
            id_to_actor[self.ego.id] = self.ego
        for v in self.group_vehs:
            id_to_actor[v.id] = v

        fixed_dt = float(self._world._settings.fixed_delta_seconds)

        for gid, members in self.groups.items():
            members = list(members)
            for sender_id in members:
                sender = id_to_actor.get(sender_id, None)
                if sender is None:
                    continue

                payload = self._make_payload(sender)
                payload_bytes = _tx_bytes_for_latency(
                    payload,
                    overhead_bytes=getattr(self.latency_model, "overhead_bytes", 64),
                )

                for receiver_id in members:
                    if receiver_id == sender_id:
                        continue
                    receiver = id_to_actor.get(receiver_id, None)
                    if receiver is None:
                        continue

                    s_res = self._veh_net_res.get(sender_id, self._default_net_res)
                    r_res = self._veh_net_res.get(receiver_id, self._default_net_res)

                    latency_s = self.latency_model.compute_latency_s(
                        sender=sender,
                        receiver=receiver,
                        payload_size_bytes=payload_bytes,
                        sender_res=s_res,
                        receiver_res=r_res,
                        out_degree=max(out_deg.get(sender_id, 1), 1),
                        in_degree=max(in_deg.get(receiver_id, 1), 1),
                    )

                    delay_steps = int(math.ceil(latency_s / max(fixed_dt, 1e-6)))
                    delay_steps = max(delay_steps, 0)
                    deliver_step = int(self._time_step + delay_steps)

                    msg = V2VMessage(
                        sender_id=int(sender_id),
                        receiver_id=int(receiver_id),
                        group_id=int(gid),
                        payload=payload,
                        payload_bytes=int(payload_bytes),
                        created_step=int(self._time_step),
                        deliver_step=int(deliver_step),
                        latency_s=float(latency_s),
                        distance_m=float(_dist_m(sender, receiver)),
                    )
                    self._in_flight.append(msg)

    def _deliver_messages(self) -> None:
        if not self._in_flight:
            return

        cur = int(self._time_step)
        remaining: List[V2VMessage] = []
        for msg in self._in_flight:
            if msg.deliver_step <= cur:
                self._received[msg.receiver_id].append(msg)
            else:
                remaining.append(msg)
        self._in_flight = remaining

    def apply_control(self, action):
        control = self.agent.run_step()
        self.ego.apply_control(control)

    def get_state(self):
        self._state = {"ego_waypoints": self.waypoints, "timesteps": self._time_step}
        return self._state

    def step(self, action):
        self.get_state()
        _, reward, terminated, truncated, info = super().step(action)

        ego_feature = self.payload_fn(self.ego, self.obs, self.feature_size)
        msgs = self._received.get(self.ego.id, deque())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shared_data = self._graph_builder.build(
            ego_actor=self.ego,
            carla_world=self._world._world,
            ego_feat=ego_feature.get("feat", None),
            ego_feat_dim=ego_feature.get("feat_dim", None),
            msgs=msgs,
            t_step=self._time_step,
            dt=float(self._config.world.fixed_delta_seconds),
            device=device,
        )

        partial_info = {
            k: v
            for k, v in info.items()
            if k.startswith("r_") or k in [
                "wpt_dis",
                "speed_parallel",
                "speed_perpendicular",
                "speed_norm",
                "ttc",
                "time_penalty",
            ]
        }
        partial_info["ego_x"] = self.ego.get_transform().location.x
        partial_info["ego_y"] = self.ego.get_transform().location.y
        shared_data.update(partial_info)
        info = shared_data

        if terminated or truncated:
            self.dump_vlm_records(os.path.join("data", f"vlm_records_step_{int(self._time_step)}.json"))
            raise RuntimeError(f"Episode ended at step {int(self._time_step)}. VLM records dumped.")

        return self.obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        print("[CARLA Group Right Turn Env] Reset environment")
        _, info = super().reset(seed=seed)

        ego_feature = self.payload_fn(self.ego, self.obs, self.feature_size)
        msgs = self._received.get(self.ego.id, deque())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shared_data = self._graph_builder.build(
            ego_actor=self.ego,
            carla_world=self._world._world,
            ego_feat=ego_feature.get("feat", None),
            ego_feat_dim=ego_feature.get("feat_dim", None),
            msgs=msgs,
            t_step=self._time_step,
            dt=float(self._config.world.fixed_delta_seconds),
            device=device,
        )

        ego_location = np.array([*get_vehicle_pos(self.ego)])
        reward_info = {
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": 0,
            "speed_perpendicular": 0,
            "speed_norm": 0,
            "wpt_dis": self.get_wpt_dist(ego_location),
            "r_waypoints": 0,
            "r_speed": 0,
            "r_collision": 0,
            "r_out_of_lane": 0,
            "r_destination": 0,
            "time_penalty": 0,
            "ttc": 0,
        }

        shared_data.update(reward_info)
        info = shared_data

        return self.obs, info

    def dump_vlm_records(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vlm_records, f, ensure_ascii=False, indent=2)
