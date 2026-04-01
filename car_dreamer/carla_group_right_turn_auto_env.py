from __future__ import annotations

import math
import json
import re
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import os
import traceback
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import carla
from agents.navigation.basic_agent import BasicAgent

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import _dist_m
from .toolkit import NetResource, V2VMessage, LatencyModel, SimpleWirelessLatency, _tx_bytes_for_latency
from .toolkit import Observer, payload_fn_llm
from .toolkit import get_vehicle_pos
from .toolkit import VehicleNodeGraphBuilder, GraphBuildConfig, compute_query_direction_from_observer


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
        uplink_bps = float(getattr(comm_cfg, "uplink_bps", 6e6))
        downlink_bps = float(getattr(comm_cfg, "downlink_bps", 12e6))
        base_rtt_s = float(getattr(comm_cfg, "base_rtt_s", 0.02))
        proc_delay_s = float(getattr(comm_cfg, "proc_delay_s", 0.005))
        distance_decay_m = float(getattr(comm_cfg, "distance_decay_m", 60.0))
        min_rate_factor = float(getattr(comm_cfg, "min_rate_factor", 0.2))
        jitter_s = float(getattr(comm_cfg, "jitter_s", 0.0))
        overhead_bytes = int(getattr(comm_cfg, "overhead_bytes", 64))

        self._default_net_res = NetResource(uplink_bps=uplink_bps, downlink_bps=downlink_bps)
        self.latency_model: LatencyModel = SimpleWirelessLatency(
            base_rtt_s=base_rtt_s,
            proc_delay_s=proc_delay_s,
            distance_decay_m=distance_decay_m,
            min_rate_factor=min_rate_factor,
            jitter_s=jitter_s,
            overhead_bytes=overhead_bytes,
        )

        self.payload_fn = payload_fn_llm
        self.trans_msg_type = str(getattr(self._config, "trans_msg_type", "image"))

        # comm buffers
        self._in_flight: List[V2VMessage] = []
        self._received: Dict[int, Deque[V2VMessage]] = defaultdict(lambda: deque(maxlen=256))

        # Per-vehicle network resources
        self._veh_net_res: Dict[int, NetResource] = {}

        # feature size
        self.feature_size = int(getattr(self._config, "feature_size", 64))

        # graph builder
        graph_cfg = getattr(self._config, "graph", None)
        cfg = GraphBuildConfig(
            window_s=float(getattr(graph_cfg, "window_s", 2.0)),
            Tmax=int(getattr(graph_cfg, "tmax", 15)),
            max_nodes=int(getattr(graph_cfg, "max_nodes", 3)),
            feat_dim_max=int(getattr(graph_cfg, "feat_dim_max", 128)),
            star_graph=bool(getattr(graph_cfg, "star_graph", True)),
        )
        self._graph_builder = VehicleNodeGraphBuilder(cfg)

        # -----------------------------
        # VLM (Qwen2-VL) configuration
        # -----------------------------
        vlm_cfg = getattr(self._config, "vlm", None)
        self._vlm_enabled = bool(getattr(vlm_cfg, "enabled", True))
        # Default to Qwen2-VL; can be overridden in config
        self._vlm_model_name = str(getattr(vlm_cfg, "model_name", "Qwen/Qwen2.5-VL-3B-Instruct"))
        # image_template is kept for config compatibility
        self._vlm_image_template = str(getattr(vlm_cfg, "image_template", "Analyze the driving scene."))
        self._vlm_eval_period = int(getattr(vlm_cfg, "eval_period", 1))
        self._vlm_image_obs_key = str(getattr(vlm_cfg, "image_obs_key", "camera"))
        self._vlm_local_files_only = bool(getattr(vlm_cfg, "local_files_only", False))
        self._vlm_shared_source = str(getattr(vlm_cfg, "shared_source", "received_feat"))  # received_feat | raw
        self._vlm_received_window_s = float(getattr(vlm_cfg, "received_window_s", 2.0))
        self._vlm_max_msgs_per_sender = int(getattr(vlm_cfg, "max_msgs_per_sender", 20))
        self._vlm_max_images_per_sender_for_inference = int(
            getattr(vlm_cfg, "max_images_per_sender_for_inference", 4)
        )
        self._vlm_sampling_strategy = str(getattr(vlm_cfg, "sampling_strategy", "uniform"))  # uniform | latest
        self._vlm_max_total_shared_images = int(getattr(vlm_cfg, "max_total_shared_images", 12))

        # sender-level confidence weights
        self._vlm_ego_conf_weight = float(getattr(vlm_cfg, "ego_conf_weight", 1.0))
        default_shared_weight = float(getattr(vlm_cfg, "shared_conf_weight", 1.0))
        shared_weights_cfg = getattr(vlm_cfg, "shared_weights", None)
        self._vlm_default_shared_conf_weight = default_shared_weight
        self._vlm_shared_conf_weights: Dict[int, float] = {}
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
        self._vlm_sensor_fov_deg = float(getattr(vlm_cfg, "sensor_fov_deg", 120.0)) #TODO

        self._vlm_do_sample = bool(getattr(vlm_cfg, "do_sample", False))
        self._vlm_score_max_new_tokens = int(getattr(vlm_cfg, "score_max_new_tokens", 128))
        self._vlm_temperature = float(getattr(vlm_cfg, "temperature", 0.0))
        self._vlm_top_p = float(getattr(vlm_cfg, "top_p", 0.9))

        # Qwen2-VL model and processor
        self._vlm_model  = None
        self._vlm_processor: Optional[AutoProcessor] = None

        self._vlm_records: List[Dict[str, Any]] = []
        self._vlm_last_eval: Dict[str, Any] = {}
        self._vlm_questions = self._build_vlm_questions()

        if self._vlm_enabled:
            self._init_vlm()

    # =========================================================
    # VLM language-mediated communication
    # =========================================================

    def _init_vlm(self) -> None:
        """Load Qwen2-VL model and processor."""
        print(f"[Qwen2-VL] Loading model: {self._vlm_model_name}")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._vlm_model_name,
            torch_dtype=dtype,
            local_files_only=self._vlm_local_files_only,
        )
        self._vlm_processor = AutoProcessor.from_pretrained(
            self._vlm_model_name,
            local_files_only=self._vlm_local_files_only,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vlm_model = self._vlm_model.to(device)
        self._vlm_model.eval()


    def _build_vlm_questions(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "clg_left_rear_vehicle",
                "type": "clg",
                "query": "Is there a vehicle in the left-rear region of the vehicle?",
                "positive": "There is a vehicle in the left-rear region of the vehicle.",
                "negative": "There is no vehicle in the left-rear region of the vehicle.",
            },
            {
                "id": "clg_right_rear_vehicle",
                "type": "clg",
                "query": "Is there a vehicle in the right-rear region of the vehicle?",
                "positive": "There is a vehicle in the right-rear region of the vehicle.",
                "negative": "There is no vehicle in the right-rear region of the vehicle.",
            },
            {
                "id": "clg_right_front_vehicle",
                "type": "clg",
                "query": "Is there a vehicle in the right-front region of the vehicle?",
                "positive": "There is a vehicle in the right-front region of the vehicle.",
                "negative": "There is no vehicle in the right-front region of the vehicle.",
            },
            {
                "id": "clg_left_front_vehicle",
                "type": "clg",
                "query": "Is there a vehicle in the left-front region of the vehicle?",
                "positive": "There is a vehicle in the left-front region of the vehicle.",
                "negative": "There is no vehicle in the left-front region of the vehicle.",
            },
        ]

    def _masked_mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Legacy helper retained for API compatibility."""
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

    def _wrap_angle(self, angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


    def _question_target_angle_rad(self, question_cfg: Dict[str, Any]) -> float:
        """
        Return the target angle in the ego-local semantic frame,
        consistent with the simulator convention:

            0       : front
            +pi/2   : right
            -pi/2   : left
            +/-pi   : rear

        This matches the environment's angle convention:
        - angle is measured from +X
        - clockwise is positive
        - counterclockwise is negative
        """
        # qid = str(question_cfg.get("id", "")).lower()

        # if "left_rear" in qid:
        #     return -3.0 * math.pi / 4.0
        # if "right_rear" in qid:
        #     return 3.0 * math.pi / 4.0
        # if "right_front" in qid:
        #     return math.pi / 4.0
        # if "left_front" in qid:
        #     return -math.pi / 4.0
        # if "left_vehicle_speed" in qid or "left_side" in qid:
        #     return -math.pi / 2.0

        # return 0.0
        dx_local, dy_local = self._question_target_offset_xy(question_cfg)
        return self._wrap_angle(math.atan2(dy_local, dx_local))


    def _question_target_angle_global_rad(
        self,
        question_cfg: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> float:
        """
        Convert the ego-local target angle into the global frame.

        In this simulator:
        - yaw is measured from global +X
        - clockwise is positive

        So the ego-local target angle can be added directly to ego yaw.
        """
        target_local = self._question_target_angle_rad(question_cfg)
        ego_yaw_rad = math.radians(float(ego_pose.get("yaw", 0.0)))
        return self._wrap_angle(ego_yaw_rad + target_local)

    def _question_target_offset_xy(self, question_cfg: Dict[str, Any]) -> tuple[float, float]:
        """
        Return the queried target-region center in the ego-local frame.

        Ego-local semantic frame:
            +x = front
            -x = rear
            +y = right
            -y = left

        This is chosen to match the simulator's clockwise-positive convention.
        """
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
        if "left_vehicle_speed" in qid or "left_side" in qid:
            return (0.0, -side_d)

        return (front_d, 0.0)


    def _question_target_point_global(
        self,
        question_cfg: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Convert the queried target-region center from ego-local frame to global frame.

        Environment convention:
        - global +X points right
        - global +Y points down
        - clockwise rotation is positive

        Under this convention, rotating a local vector by yaw uses:

            x_global = cos(yaw) * x_local + sin(yaw) * y_local
            y_global = -sin(yaw) * x_local + cos(yaw) * y_local
        """
        dx_local, dy_local = self._question_target_offset_xy(question_cfg)

        ego_x = float(ego_pose.get("x", 0.0))
        ego_y = float(ego_pose.get("y", 0.0))
        ego_yaw_rad = math.radians(float(ego_pose.get("yaw", 0.0)))

        c = math.cos(ego_yaw_rad)
        s = math.sin(ego_yaw_rad)

        dx_global = c * dx_local - s * dy_local
        dy_global = s * dx_local + c * dy_local

        return {
            "x": ego_x + dx_global,
            "y": ego_y + dy_global,
        }
    
    def _compute_single_image_embedding_from_array(self, img_np):
        raise RuntimeError(
            "Image embeddings are disabled for Qwen2-VL in this environment. "
            "Use _compute_single_image_description_from_array instead."
        )

    def _compute_single_image_embedding(self, image: Image.Image) -> torch.Tensor:
        raise RuntimeError(
            "Image embeddings are disabled for Qwen2-VL in this environment. "
            "Use _compute_single_image_description instead."
        )

    def _coerce_to_pil_image(self, image: Any) -> Optional[Image.Image]:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        return Image.fromarray(image).convert("RGB")

    def _build_scene_description_prompt(self, detail_level: str = "medium") -> str:
        if detail_level == "short":
            region_rule = "Use one short factual clause each row."
        elif detail_level == "long":
            region_rule = "Use four short factual clauses each row"
        else:
            region_rule = "Use two short factual clause each row."

        return (
            "This is a photograph captured by the vehicle's forward-facing camera, which is capable of capturing images only of the area directly in front of the vehicle.a view which includes the vehicle's own front end.\n"
            "Descirbe the driving image. Describe only safety-relevant facts, especially other vehicles.\n"
            "Use words [likely], [unknown], [certain], or [uncertain] to express the degree of certainty in your description.\n"

            "Focus on these regions relative to the vehicle in the image:\n"
            "front, left-front, right-front, rear, left-rear, right-rear.\n\n"

            f"Return exactly this format.{region_rule}:\n"
            "Front: ...\n"
            "Left-front: ...\n"
            "Right-front: =...\n"
            "Rear: ...\n"
            "Left-rear: ...\n"
            "Right-rear: ...\n"
        )

    def _build_language_scoring_prompt(
        self,
        question_cfg: Dict[str, Any],
        fused_evidence: str,
    ) -> str:
        question_id = str(question_cfg.get("id", "unknown_question"))
        question_type = str(question_cfg.get("type", "clg"))
        query = str(question_cfg.get("query", "")).strip()
        positive = str(question_cfg.get("positive", "")).strip()
        negative = str(question_cfg.get("negative", "")).strip()

        evidence_text = fused_evidence if fused_evidence else "No textual evidence provided."

        return (
            "You are evaluating cooperative driving evidence for one binary question.\n"
            f"Question ID: {question_id}\n"
            f"Question type: {question_type}\n"
            f"Query: {query}\n"
            f"Positive statement: {positive}\n"
            f"Negative statement: {negative}\n\n"
            "Use only the provided evidence. Do not assume unseen facts.\n"
            "Some evidence may be partial, occluded, weak, indirect, or conflicting.\n"
            "If the evidence does not clearly support either side, do not guess.\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "support_direction": "positive" or "negative" or "mixed" or "insufficient",\n'
            '  "support_strength": "none" or "weak" or "moderate" or "strong",\n'
            '  "visibility": "clear" or "partial" or "weak" or "insufficient",\n'
            '  "reason": "one short sentence"\n'
            "}\n\n"
            "Scoring rubric:\n"
            "- support_direction=positive: the evidence supports the positive statement more than the negative statement.\n"
            "- support_direction=negative: the evidence supports the negative statement more than the positive statement.\n"
            "- support_direction=mixed: there is support for both sides or conflicting evidence.\n"
            "- support_direction=insufficient: the evidence is not enough to support either side.\n"
            "- support_strength=strong: explicit, direct, and consistent evidence.\n"
            "- support_strength=moderate: meaningful but incomplete evidence.\n"
            "- support_strength=weak: slight indication only.\n"
            "- support_strength=none: no meaningful support.\n"
            "- visibility=clear: the relevant region/status is clearly described.\n"
            "- visibility=partial: partially visible or partially described.\n"
            "- visibility=weak: weakly described, vague, or low-quality evidence.\n"
            "- visibility=insufficient: cannot reliably determine from the evidence.\n\n"
            "Important rules:\n"
            "- Do not use 'strong' unless the evidence is explicit and unambiguous.\n"
            "- If evidence is partial, indirect, occluded, vague, or inferred, use at most 'moderate'.\n"
            "- If the evidence cannot reliably determine the answer, use support_direction='insufficient'.\n"
            "- Return JSON only."
        )

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _run_qwen_generation(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if self._vlm_model is None or self._vlm_processor is None:
            raise RuntimeError("Qwen2-VL model is not initialized.")

        model_device = next(self._vlm_model.parameters()).device

        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._vlm_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._vlm_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._vlm_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._vlm_processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(self._vlm_do_sample),
        }
        if bool(self._vlm_do_sample):
            gen_kwargs["temperature"] = float(self._vlm_temperature)
            gen_kwargs["top_p"] = float(self._vlm_top_p)

        with torch.no_grad():
            generated_ids = self._vlm_model.generate(**inputs, **gen_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        generated_only = generated_ids[:, prompt_len:] if prompt_len > 0 else generated_ids
        raw_text = self._vlm_processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return raw_text.strip()
    
    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _run_qwen_generation(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if self._vlm_model is None or self._vlm_processor is None:
            raise RuntimeError("Qwen2-VL model is not initialized.")

        model_device = next(self._vlm_model.parameters()).device

        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._vlm_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._vlm_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._vlm_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._vlm_processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(self._vlm_do_sample),
        }
        if bool(self._vlm_do_sample):
            gen_kwargs["temperature"] = float(self._vlm_temperature)
            gen_kwargs["top_p"] = float(self._vlm_top_p)

        with torch.no_grad():
            generated_ids = self._vlm_model.generate(**inputs, **gen_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        generated_only = generated_ids[:, prompt_len:] if prompt_len > 0 else generated_ids
        raw_text = self._vlm_processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return raw_text.strip()

    def _compute_single_image_description(self, image: Image.Image, token_size) -> str:
        if self._vlm_model is None or self._vlm_processor is None:
            raise RuntimeError("Qwen2-VL model is not initialized.")
        if image is None:
            raise ValueError("Image cannot be converted to PIL for Qwen2-VL captioning.")
        prompt = self._build_scene_description_prompt()
        return self._run_qwen_generation(prompt=prompt, image=image, max_new_tokens=token_size)

    def _compute_single_image_description_from_array(self, img_np, token_size):
        image = Image.fromarray(img_np).convert("RGB")
        if image is None:
            raise ValueError("img_np cannot be converted to PIL image")
        return self._compute_single_image_description(image, token_size)


    def _parse_language_scores(self, raw_text: str) -> Dict[str, Any]:
        parsed = self._extract_first_json_object(raw_text) or {}

        def _clip01(v: Any, default: float) -> float:
            try:
                return float(max(0.0, min(1.0, float(v))))
            except Exception:
                return float(default)

        def _norm_answer(ans: Any) -> str:
            s = str(ans).strip().lower()
            if s in {"positive", "negative", "uncertain"}:
                return s
            return "uncertain"

        # ------------------------------------------------------------------
        # New structured output path:
        # {
        #   "support_direction": "positive" | "negative" | "mixed" | "insufficient",
        #   "support_strength": "none" | "weak" | "moderate" | "strong",
        #   "visibility": "clear" | "partial" | "weak" | "insufficient",
        #   "reason": "..."
        # }
        # ------------------------------------------------------------------
        support_direction = str(parsed.get("support_direction", "")).strip().lower()
        support_strength = str(parsed.get("support_strength", "")).strip().lower()
        visibility = str(parsed.get("visibility", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()

        has_new_schema = (
            support_direction in {"positive", "negative", "mixed", "insufficient"}
            or support_strength in {"none", "weak", "moderate", "strong"}
            or visibility in {"clear", "partial", "weak", "insufficient"}
        )

        if has_new_schema:
            strength_map = {
                "none": 0.0,
                "weak": 0.35,
                "moderate": 0.75,
                "strong": 0.95,
            }
            visibility_map = {
                "clear": 1.00,
                "partial": 0.70,
                "weak": 0.40,
                "insufficient": 0.15,
            }

            base = float(strength_map.get(support_strength, 0.0))
            vis = float(visibility_map.get(visibility, 0.15))

            # Base uncertainty comes primarily from visibility.
            unc = 1.0 - vis

            if support_direction == "positive":
                pos = base * vis
                neg = 0.0
            elif support_direction == "negative":
                pos = 0.0
                neg = base * vis
            elif support_direction == "mixed":
                # Conflicting evidence: split support and keep uncertainty non-trivial.
                pos = 0.5 * base * vis
                neg = 0.5 * base * vis
                unc = max(unc, 0.35)
            else:  # insufficient
                pos = 0.0
                neg = 0.0
                unc = max(unc, 0.85)

            if support_direction == "positive":
                answer = "positive" if pos >= neg else "uncertain"
            elif support_direction == "negative":
                answer = "negative" if neg >= pos else "uncertain"
            else:
                answer = "uncertain"

        else:
            # ------------------------------------------------------------------
            # Backward-compatible old schema path:
            # {
            #   "positive_score": float,
            #   "negative_score": float,
            #   "uncertainty": float,
            #   "answer": ...
            # }
            # ------------------------------------------------------------------
            pos = _clip01(parsed.get("positive_score", 0.0), 0.0)
            neg = _clip01(parsed.get("negative_score", 0.0), 0.0)
            unc = _clip01(parsed.get("uncertainty", 1.0), 1.0)

            answer = _norm_answer(parsed.get("answer", "uncertain"))
            if answer not in {"positive", "negative", "uncertain"}:
                if pos > neg and pos > unc:
                    answer = "positive"
                elif neg > pos and neg > unc:
                    answer = "negative"
                else:
                    answer = "uncertain"

        belief = float(pos - neg)
        evidence = float(1.0 - unc)
        ambiguity = float(1.0 - abs(pos - neg))
        confidence = float(abs(belief) * evidence)

        return {
            "positive_score": float(pos),
            "negative_score": float(neg),
            "unknown_score": float(unc),
            "uncertainty": float(unc),
            "answer": answer,
            "reason": reason,
            "belief": belief,
            "evidence": evidence,
            "ambiguity": ambiguity,
            "confidence": confidence,
            "raw_text": raw_text,
        }


    def _score_question_from_language_evidence(
        self,
        question_cfg: Dict[str, Any],
        fused_evidence: str,
    ) -> Dict[str, Any]:
        prompt = self._build_language_scoring_prompt(question_cfg, fused_evidence)
        raw_text = self._run_qwen_generation(prompt=prompt, image=None, max_new_tokens=self._vlm_score_max_new_tokens)     # 这里的max_new_token不影响传输延迟
        return self._parse_language_scores(raw_text)

    def _build_ego_sensor_instances(self, ego_image: Image.Image) -> List[Dict[str, Any]]:
        tf = self.ego.get_transform()
        yaw_rad = math.radians(float(tf.rotation.yaw))
        scene_description = self._compute_single_image_description(ego_image, token_size=self.feature_size)
        return [{
            "sender_id": int(self.ego.id),
            "sensor_name": "cam0",
            "img_emb": None, #ego_image,
            "scene_description": scene_description,
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
                "img_emb": info.get("img_emb", None),
                "scene_description": str(info.get("scene_description", "")).strip(),
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
                "unknown_score": float(sum(x.get("unknown_score", x.get("uncertainty", 0.0)) for x in items) / n),
                "belief": float(sum(x.get("belief", 0.0) for x in items) / n),
                "evidence": float(sum(x.get("evidence", 0.0) for x in items) / n),
                "ambiguity": float(sum(x.get("ambiguity", 1.0) for x in items) / n),
                "confidence": float(sum(float(x.get("confidence", 0.0)) for x in items) / n),
                "visibility_score": float(sum(1.0 - float(x.get("uncertainty", x.get("unknown_score", 1.0))) for x in items) / n),
                "answer": str(items[-1].get("answer", "uncertain")),
                "num_images": int(len(items)),
                "latency": float(min(float(x.get("received_age_s", 0.0)) for x in items)),
                "received_age_s_mean": float(sum(float(x.get("received_age_s", 0.0)) for x in items) / n),
                "pose": dict(items[0].get("pose", {})),
                "sensor_yaw_rad": float(items[0].get("sensor_yaw_rad", 0.0)),
                "reason": str(items[-1].get("reason", "")),
                "reasons": [str(x.get("reason", "")) for x in items],
                "raw_outputs": [str(x.get("raw_text", "")) for x in items],
                "scene_description": str(items[-1].get("scene_description", "")),
                "language_evidence": str(items[-1].get("language_evidence", "")),
                "converted_query": str(items[-1].get("converted_query", ""))
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

    def _build_sensor_position_features(
        self,
        sensor_score: Dict[str, Any],
        ego_pose: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Build geometric features in the simulator's global frame.

        Assumed simulator convention:
        - +X points right
        - +Y points down
        - angle is measured from +X
        - clockwise is positive

        Under this convention, atan2(dy, dx) is consistent with the simulator angle.
        """
        sx = float(sensor_score.get("pose", {}).get("x", 0.0))
        sy = float(sensor_score.get("pose", {}).get("y", 0.0))

        ex = float(ego_pose.get("x", 0.0))
        ey = float(ego_pose.get("y", 0.0))

        dx = sx - ex
        dy = sy - ey
        d = math.sqrt(dx * dx + dy * dy)

        if d > 1e-6:
            # ego -> sensor
            bearing_from_ego = math.atan2(dy, dx)
            # sensor -> ego
            bearing_to_ego = math.atan2(-dy, -dx)
        else:
            bearing_from_ego = 0.0
            bearing_to_ego = 0.0

        psi = float(sensor_score.get("sensor_yaw_rad", 0.0))

        return {
            "sensor_x": float(sx),
            "sensor_y": float(sy),
            "ego_x": float(ex),
            "ego_y": float(ey),
            "dx": float(dx),
            "dy": float(dy),
            "distance_m": float(d),
            "bearing_from_ego_rad": float(bearing_from_ego),
            "bearing_to_ego_rad": float(bearing_to_ego),
            "bearing_from_ego_deg": float((math.degrees(bearing_from_ego) + 180.0) % 360.0 - 180.0),
            "bearing_to_ego_deg": float((math.degrees(bearing_to_ego) + 180.0) % 360.0 - 180.0),
            "sin_theta": float(math.sin(bearing_from_ego)),
            "cos_theta": float(math.cos(bearing_from_ego)),
            "sensor_yaw_rad": float(psi),
            "sensor_yaw_deg": float((math.degrees(psi) + 180.0) % 360.0 - 180.0),
            "sin_psi": float(math.sin(psi)),
            "cos_psi": float(math.cos(psi)),
        }
        
    def _compute_sensor_importance_maps(
        self,
        question_cfg: Dict[str, Any],
        sensor_mean_scores: Dict[str, Dict[str, Any]],
        ego_pose: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute per-sensor importance weights for the queried region.

        Definitions:
        - region_alignment:
            Is the sensor spatially located on the useful side of the ego vehicle?
        - facing_alignment:
            Is the sensor heading toward the queried target-region center?
        - fov_alignment:
            Is the queried target-region center inside the sensor FOV?
        - distance_alignment:
            Is the sensor close enough to be useful?
        """
        target_angle = self._question_target_angle_global_rad(question_cfg, ego_pose)
        target_angle_deg = (math.degrees(target_angle) + 180.0) % 360.0 - 180.0
        target_point = self._question_target_point_global(question_cfg, ego_pose)

        tau = max(float(self._vlm_importance_distance_tau), 1e-6)
        half_fov = 0.5 * math.radians(float(self._vlm_sensor_fov_deg))

        logits: List[float] = []
        sensor_keys: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        ego_x = float(ego_pose.get("x", 0.0))
        ego_y = float(ego_pose.get("y", 0.0))

        for sensor_key, sensor_score in sensor_mean_scores.items():
            feats = self._build_sensor_position_features(sensor_score, ego_pose)
            d = float(feats["distance_m"])

            sensor_x = feats.get("sensor_x", None)
            sensor_y = feats.get("sensor_y", None)
            if sensor_x is None or sensor_y is None:
                pose = sensor_score.get("pose", {})
                sensor_x = float(pose.get("x", ego_x))
                sensor_y = float(pose.get("y", ego_y))
            else:
                sensor_x = float(sensor_x)
                sensor_y = float(sensor_y)

            sensor_yaw_rad = float(feats["sensor_yaw_rad"])

            # sensor -> queried target region center
            bearing_to_target_rad = math.atan2(
                float(target_point["y"]) - sensor_y,
                float(target_point["x"]) - sensor_x,
            )

            # If your simulator angles are already produced in the same convention
            # as sensor_yaw_rad / bearing_from_ego_rad / bearing_to_ego_rad,
            # then wrap_angle difference is still valid.
            angle_diff_target = self._wrap_angle(sensor_yaw_rad - bearing_to_target_rad)

            # heading alignment to target-region center
            facing_alignment = 0.5 * (1.0 + math.cos(angle_diff_target))

            # FOV gate
            norm = angle_diff_target / max(half_fov, 1e-6)
            if abs(norm) <= 1.0:
                fov_alignment = 0.5 * (1.0 + math.cos(0.5 * norm * math.pi)) # TODO: 这里如果在边缘，就变成0了，不至于，稍微缩小下边缘衰退
            else:
                fov_alignment = 0.0

            if sensor_score["is_ego"]:
                region_alignment = 1.0
                distance_alignment = 1.0
            else:
                region_alignment = 0.5 * (
                    1.0
                    + math.cos(
                        self._wrap_angle(
                            float(feats["bearing_from_ego_rad"]) - target_angle
                        )
                    )
                )
                distance_alignment = math.exp(-d / tau)
                
            logit = (
                float(region_alignment)
                * float(facing_alignment)
                * float(fov_alignment)
                * float(distance_alignment)
            )

            logits.append(float(logit))
            sensor_keys.append(sensor_key)

            details[sensor_key] = {
                **feats,
                "sensor_x": float(sensor_x),
                "sensor_y": float(sensor_y),
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

        denom = sum(math.exp(v) for v in logits) + 1e-8
        for sensor_key, logit in zip(sensor_keys, logits):
            details[sensor_key]["importance_weight"] = logit # float(math.exp(logit) / denom)   # TODO: 要不要归一化，如果归一化，当只有ego时，ego权重就变成1了，反而不对

    # def _compute_sensor_importance_maps(
    #     self,
    #     question_cfg: Dict[str, Any],
    #     sensor_mean_scores: Dict[str, Dict[str, Any]],
    #     ego_pose: Dict[str, float],
    # ) -> Dict[str, Dict[str, Any]]:
    #     target_angle = self._question_target_angle_global_rad(question_cfg, ego_pose)
    #     tau = max(float(self._vlm_importance_distance_tau), 1e-6)
    #     logits: List[float] = []
    #     sensor_keys: List[str] = []
    #     details: Dict[str, Dict[str, Any]] = {}
    #     # 把rad转化为angle
    #     target_angle_deg = (math.degrees(target_angle) + 180) % 360 - 180

    #     for sensor_key, sensor_score in sensor_mean_scores.items():
    #         feats = self._build_sensor_position_features(sensor_score, ego_pose)
    #         d = feats["distance_m"]

    #         if sensor_score["is_ego"]:
    #             angle_diff = self._wrap_angle(feats["sensor_yaw_rad"] - target_angle)
    #             half_fov = math.radians(self._vlm_sensor_fov_deg)
    #             if abs(angle_diff) <= half_fov:
    #                 region_alignment = 0.5 * (1.0 + math.cos(angle_diff))
    #                 facing_alignment = region_alignment
    #             else:
    #                 region_alignment = 0.0
    #                 facing_alignment = 0.0
    #             distance_alignment = 1.0
    #         else:
    #             region_alignment = 0.5 * (1.0 + math.cos(self._wrap_angle(feats["bearing_from_ego_rad"] - target_angle)))
    #             facing_alignment = 0.5 * (1.0 + math.cos(self._wrap_angle(feats["sensor_yaw_rad"] - feats["bearing_to_ego_rad"])))
    #             distance_alignment = math.exp(-d / tau)

    #         fov_alignment = 0.0
    #         text_evidence = float(max(0.0, min(1.0, sensor_score.get("evidence", 0.0))))
    #         logit = (float(region_alignment) + float(facing_alignment)) * float(distance_alignment) * (0.5 + 0.5 * text_evidence)
    #         logits.append(float(logit))
    #         sensor_keys.append(sensor_key)
    #         details[sensor_key] = {
    #             **feats,
    #             "target_angle_global_rad": float(target_angle),
    #             "target_angle": float(target_angle_deg),
    #             "region_alignment": float(region_alignment),
    #             "fov_alignment": float(fov_alignment),
    #             "facing_alignment": float(facing_alignment),
    #             "distance_alignment": float(distance_alignment),
    #             "raw_logit": float(logit),
    #         }

        if not sensor_keys:
            return {"per_sensor": {}, "per_sender_positive": {}, "per_sender_negative": {}}

        weights = logits
        per_sensor: Dict[str, Dict[str, Any]] = {}
        per_sender_positive: Dict[int, float] = defaultdict(float)
        per_sender_negative: Dict[int, float] = defaultdict(float)

        for sensor_key, w in zip(sensor_keys, weights):
            sensor_score = sensor_mean_scores[sensor_key]
            sender_id = int(sensor_score["sender_id"])
            per_sensor[sensor_key] = {
                # "importance_positive": float(w),        # TODO：这里改个名吧，importance_positive 还以为是权重
                # "importance_negative": float(w),
                **details[sensor_key],
            }
            per_sender_positive[sender_id] += float(w)
            per_sender_negative[sender_id] += float(w)

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
            m = float(imp.get("importance_weight", 0.0)) * sender_weight
            s_pos, s_neg, s_unk = sensor_score["positive_score"], sensor_score["negative_score"], sensor_score["unknown_score"]
            information = m * (math.log(1+s_pos) + math.log(1+s_neg) + math.log(1+s_unk)) # TODO: 这里的定义，参考老师的意见
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
        return float(self._vlm_shared_conf_weights.get(int(sender_id), self._vlm_default_shared_conf_weight))

    def _weighted_confidence_aggregate(
        self,
        sensor_mean_scores: Dict[str, Dict[str, Any]],
        importance_maps: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        per_sensor_importance = importance_maps.get("per_sensor", {})
        per_sensor_details: List[Dict[str, Any]] = []

        total_pos = 0.0
        total_neg = 0.0
        total_unk = 0.0
        total_evidence = 0.0
        total_belief = 0.0
        ego_only = {
            "positive_score": 0.0,
            "negative_score": 0.0,
            "unknown_score": 1.0,
            "belief": 0.0,
            "evidence": 0.0,
            "confidence": 0.0,
            "answer": "uncertain",
        }

        for sensor_key in sorted(sensor_mean_scores.keys()):
            sensor_score = sensor_mean_scores[sensor_key]
            imp = per_sensor_importance.get(sensor_key, {})
            
            latency = imp.get("latency", 0.0)
            timeliness = math.exp(-latency)

            importance_weight = float(imp.get("importance_weight", 0.0))
            sender_id = int(sensor_score["sender_id"])
            is_ego = bool(sensor_score.get("is_ego", False))
            sender_weight = self._get_sender_weight(sender_id, is_ego)
            sensor_weight = importance_weight * sender_weight * timeliness

            s_pos = float(sensor_score.get("positive_score", 0.0))
            s_neg = float(sensor_score.get("negative_score", 0.0))
            s_unk = float(sensor_score.get("unknown_score", 1.0))
            belief = float(sensor_score.get("belief", s_pos - s_neg))
            evidence = float(sensor_score.get("evidence", 1.0 - s_unk))

            total_pos += sensor_weight * s_pos
            total_neg += sensor_weight * s_neg
            total_unk += sensor_weight * s_unk
            total_evidence += sensor_weight * evidence
            weight = sensor_weight * evidence       # final weight
            contrib = weight * belief         

            total_belief += contrib

            if is_ego:
                ego_only = {
                    "positive_score": s_pos,
                    "negative_score": s_neg,
                    "unknown_score": s_unk,
                    "belief": belief,
                    "evidence": evidence,
                    "timeliness": timeliness,
                    "importance_weight": importance_weight,
                    "sender_weight": sender_weight,
                    "weight": weight,
                    "confidence": abs(belief) * weight,
                    "answer": str(sensor_score.get("answer", "uncertain")),
                }

            per_sensor_details.append({
                "sensor_key": sensor_key,
                "sender_id": sender_id,
                "is_ego": is_ego,
                "sender_weight": sender_weight,
                "sensor_weight": sensor_weight,
                "positive_score": s_pos,
                "negative_score": s_neg,
                "unknown_score": s_unk,
                "belief": belief,
                "evidence": evidence,
                "timeliness": timeliness,
                "importance_weight": importance_weight,
                "sender_weight": sender_weight,
                "weight": weight,
                "confidence": abs(belief) * weight,
                "answer": str(sensor_score.get("answer", "uncertain")),
            })

        final_answer = "uncertain"
        if total_pos > total_neg and total_pos > total_unk:
            final_answer = "positive"
        elif total_neg > total_pos and total_neg > total_unk:
            final_answer = "negative"
        else:
            final_answer = "uncertain"

        return {
            "positive_score": total_pos,
            "negative_score": total_neg,
            "unknown_score": total_unk,
            "belief": total_belief,
            "evidence": total_evidence,
            "confidence": abs(total_belief),
            "ego_only": ego_only,
            "per_sensor_details": per_sensor_details,
            "answer": final_answer
        }

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
            "received_age_s": 0.0,
            "image": image,         # 只有raw获得的neighbor信息才有raw image，received 是image_emb
        }

    def _make_shared_info_from_message(self, msg: V2VMessage, payload) -> Optional[Dict[str, Any]]:
        sender_id = int(msg.sender_id)
        actor = None
        for veh in self.group_vehs:
            if int(veh.id) == sender_id:
                actor = veh
                break
        if actor is None:
            return None

        tf = actor.get_transform()      # TODO：这里应该用msg中的pos吧
        received_age_steps = max(int(self._time_step) - int(msg.created_step), 0)
        received_age_s = received_age_steps * self._get_fixed_dt()

        return {
            "sender_id": sender_id,
            "received_age_s": float(received_age_s),
            "deliver_step": int(msg.deliver_step),
            "created_step": int(msg.created_step),
            **payload,
        }

    def _get_raw_shared_images_info(self) -> Tuple[List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        shared_infos: List[Dict[str, Any]] = []

        for veh in self.group_vehs:
            obs = self.group_obs.get(veh.id, None)
            image_np = obs.get("camera", None)
            if image_np is None:
                continue
            image = Image.fromarray(image_np).convert("RGB")
            scene_description = self._compute_single_image_description(image, self.feature_size)
            info = self._make_shared_info_from_actor(veh, image)
            info["scene_description"] = scene_description
            shared_infos.append(info)
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

    def _get_received_shared_images_info(self) -> Tuple[List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        receiver_id = int(self.ego.id)
        window_msgs = self._get_received_messages_in_window(receiver_id, self._vlm_received_window_s)
        num_candidate_msgs = len(window_msgs)

        grouped = self._group_messages_by_sender(window_msgs)

        per_sender_infos: Dict[int, List[Dict[str, Any]]] = {}
        for sender_id, msgs in grouped.items():
            msgs = sorted(msgs, key=lambda m: (int(m.deliver_step), int(m.created_step)))
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
                if not payload:
                    continue
                info = self._make_shared_info_from_message(msg, payload)    #TODO: image修改
                if info is None:
                    continue
                if info.get("scene_description") or info.get("img_emb"):    # 要么是自然语言描述，要么是图片embedding
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

        shared_images = [None for info in shared_infos] # received 要么是自然语言描述，要么是图片embedding
        meta = {
            "shared_source": "received_feat",
            "num_candidate_msgs": num_candidate_msgs,
            "num_selected_shared_images": len(shared_infos),
            "selected_sender_ids": [info["sender_id"] for info in shared_infos],
            "window_s": self._vlm_received_window_s,
            "sampling_strategy": self._vlm_sampling_strategy,
        }
        return shared_images, shared_infos, meta

    def _get_ego_and_shared_images_info(self) -> Tuple[Optional[Image.Image], List[Image.Image], List[Dict[str, Any]], Dict[str, Any]]:
        ego_image_np = self.obs.get("camera", None)
        ego_image = Image.fromarray(ego_image_np).convert("RGB") if ego_image_np is not None else None

        if self._vlm_shared_source == "raw":
            shared_images, shared_infos, meta = self._get_raw_shared_images_info()
        else:
            shared_images, shared_infos, meta = self._get_received_shared_images_info()

        return ego_image, shared_images, shared_infos, meta

    # =========================================================
    # Qwen language evaluation
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
        query_text = question_cfg["query"]

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
            if not scene_description:
                continue
            
            if sensor_info.get("is_ego"):
                converted_question_cfg = None
                scoring = self._score_question_from_language_evidence(question_cfg, scene_description)
            else:
                converted = compute_query_direction_from_observer(ego_pose=ego_pose, observer_pose=sensor_info.get("pose"),
                                                                  question_id=question_cfg["id"])
                converted_question_cfg = {
                    "id": question_cfg["id"],
                    "type": question_cfg["type"],
                    "query": converted.query,
                    "positive": converted.positive,
                    "negative": converted.negative,
                }
                scoring = self._score_question_from_language_evidence(converted_question_cfg, scene_description)
                
            per_sensor_instance_scores.append({
                "sender_id": int(sensor_info["sender_id"]),
                "sensor_name": str(sensor_info["sensor_name"]),
                "is_ego": bool(sensor_info.get("is_ego", False)),
                "received_age_s": float(sensor_info.get("received_age_s", 0.0)),
                "pose": dict(sensor_info.get("pose", {})),
                "sensor_yaw_rad": float(sensor_info.get("sensor_yaw_rad", 0.0)),
                "converted_query": converted_question_cfg["query"] if converted_question_cfg else None,
                "scene_description": scene_description,
                "language_evidence": scene_description,
                **scoring,
            })

        sensor_mean_scores = self._aggregate_sensor_scores(per_sensor_instance_scores)
        importance_maps = self._compute_sensor_importance_maps(question_cfg, sensor_mean_scores, ego_pose)
        information_level = self._compute_information(sensor_mean_scores, importance_maps)  #TODO： 目前penalty没用
        aggregated = self._weighted_confidence_aggregate(sensor_mean_scores, importance_maps)
        ego_score = aggregated["ego_only"]

        # fused_lines: List[str] = []       # TODO: fused 当前有问题，至少缺少视角转换
        # for sensor_key in sorted(sensor_mean_scores.keys()):
        #     s = sensor_mean_scores[sensor_key]
        #     region = "ego" if bool(s.get("is_ego", False)) else f"sender_{int(s.get('sender_id', -1))}"
        #     desc = str(s.get("scene_description", "")).strip()
        #     if desc:
        #         fused_lines.append(f"[{region}] {desc}")
        # fused_evidence = "\n\n".join(fused_lines)
        # fused_reasoning = self._score_question_from_language_evidence(question_cfg, fused_evidence) if fused_evidence else {
        #     "positive_score": 0.0,
        #     "negative_score": 0.0,
        #     "unknown_score": 1.0,
        #     "uncertainty": 1.0,
        #     "answer": "uncertain",
        #     "reason": "",
        #     "belief": 0.0,
        #     "evidence": 0.0,
        #     "ambiguity": 1.0,
        #     "confidence": 0.0,
        #     "raw_text": "",
        # }

        per_sensor_scores_out: List[Dict[str, Any]] = []
        for sensor_key in sorted(sensor_mean_scores.keys()):
            sensor_score = sensor_mean_scores[sensor_key]
            imp = importance_maps.get("per_sensor", {}).get(sensor_key, {})
            per_sensor_scores_out.append({
                **sensor_score,
                **imp,
                "information_term": float(information_level["per_sensor_information"].get(sensor_key, 0.0)),
            })

        return {
            "question_id": question_cfg["id"],
            "question_type": question_cfg["type"],
            "shared_source": shared_meta.get("shared_source", self._vlm_shared_source),
            "num_candidate_msgs": shared_meta.get("num_candidate_msgs", 0),
            "num_selected_shared_images": shared_meta.get("num_selected_shared_images", len(shared_images)),
            "selected_sender_ids": shared_meta.get("selected_sender_ids", []),
            "received_window_s": shared_meta.get("window_s", 0.0),
            "sampling_strategy": shared_meta.get("sampling_strategy", ""),
            "query_text": query_text,
            "positive_text": positive_text,
            "negative_text": negative_text,
            # "fused_language_evidence": fused_evidence,
            # "fused_reasoning": fused_reasoning,
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
            },
            "aggregated_details": {
                "per_sensor": aggregated["per_sensor_details"],
            },
            "sc_part1": float(aggregated["confidence"]),
            "sc_part2_information": float(information_level["weighted_information"]),
            "sc_part2_details": information_level,
            "confidecen_with_part2": aggregated["confidence"] + float(information_level["weighted_information"]),
            "confidence_gain": float(aggregated["confidence"] - ego_score.get("confidence", 0.0)),
        }

    def _evaluate_vlm_questions(self) -> Dict[str, Any]:
        if not self._vlm_enabled:
            return {}

        ego_image, shared_images, shared_infos, shared_meta = self._get_ego_and_shared_images_info()
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
                traceback.print_exc()
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
                traceback.print_exc()
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
        payload = {}
        if self.payload_fn is not None:
            obs = self.obs if sender.id == self.ego.id else self.group_obs.get(sender.id, {})
            payload = self.payload_fn(sender, obs, self.feature_size, image_proc_fn=self._compute_single_image_description_from_array)

        tf = sender.get_transform()
        vel = sender.get_velocity()
        payload.update({
            "pose": {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "yaw": float(tf.rotation.yaw),
            },
            "vel": {
                "vx": float(vel.x),
                "vy": float(vel.y),
            },
        })
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
                print(f"payload bytes: {payload_bytes}")

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

        ego_feature = self.payload_fn(self.ego, self.obs, self.feature_size, image_proc_fn=self._compute_single_image_description_from_array)
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

        ego_feature = self.payload_fn(self.ego, self.obs, self.feature_size, image_proc_fn=self._compute_single_image_description_from_array)
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