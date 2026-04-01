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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class CarlaGroupRightTurnAutoEnv():
    """
    Vehicle passes the crossing (turn right) and avoid collision.

    **Provided Tasks**: ``carla_right_turn_simple``, ``carla_right_turn_medium``, ``carla_right_turn_hard``
    """

    def __init__(self):
        # Qwen2-VL model and processor
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        self._vlm_model_name = model_name

        self._vlm_records: List[Dict[str, Any]] = []
        self._vlm_last_eval: Dict[str, Any] = {}
        self._vlm_questions = self._build_vlm_questions()
        self._vlm_local_files_only = False
        self._vlm_do_sample = False
        self._vlm_score_max_new_tokens =  128
        self._vlm_temperature = 0.0
        self._vlm_top_p =  0.9


        self._init_vlm()

    # =========================================================
    # VLM language-mediated communication
    # =========================================================

    def _init_vlm(self) -> None:
        """Load Qwen2-VL model and processor."""
        print(f"[Qwen2-VL] Loading model: {self._vlm_model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,   # 或 torch.bfloat16
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._vlm_processor = AutoProcessor.from_pretrained(self._vlm_model_name)

        self._vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._vlm_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self._vlm_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._vlm_model = self._vlm_model.to(device)
        self._vlm_model.eval()
        print(f"[Qwen2-VL] Model loaded on {device}.")

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
        print(f"\n{raw_text}\n")
        return self._parse_language_scores(raw_text)

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
        token_size
    ) -> Dict[str, Any]:

        scene_description = self._compute_single_image_description(ego_image, token_size=token_size)
        print("======"*10)
        print(f"scene description: {scene_description}\n")
        
        scoring = self._score_question_from_language_evidence(question_cfg, scene_description)
        print(scoring)
        
        
if __name__ == "__main__":
    test = CarlaGroupRightTurnAutoEnv()
    ques = test._vlm_questions
    image = Image.open("/home/peh324/Codes/CarDreamer/data/camera_frames/vehicle_109/camera_000074.png").convert("RGB")
    
    for i in range(4):
        test._evaluate_single_question(ques[i], image, 512)