from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SCENE_DESCRIPTION_REGION_PROMPTS = {
    "short": "Use one short factual clause each row.",
    "medium": "Use two short factual clauses each row.",
    "long": "Use four short factual clauses each row.",
}


class RightTurnAutoVLMPromptMixin:
    def _init_vlm(self) -> None:
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
            {
                "id": "clg_front_vehicle",
                "type": "clg",
                "query": "Is there a vehicle in the front region of the vehicle?",
                "positive": "There is a vehicle in the front region of the vehicle.",
                "negative": "There is no vehicle in the front region of the vehicle.",
            },
            {
                "id": "clg_rear_vehicle",
                "type": "clg",
                "query": "Is there a vehicle in the rear region of the vehicle?",
                "positive": "There is a vehicle in the rear region of the vehicle.",
                "negative": "There is no vehicle in the rear region of the vehicle.",
            },
        ]

    def _masked_mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
        return pooled.squeeze(0)

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
        region_rule = SCENE_DESCRIPTION_REGION_PROMPTS.get(
            detail_level, SCENE_DESCRIPTION_REGION_PROMPTS["medium"]
        )
        return (
            "This is a photograph captured by the vehicle's forward-facing camera. "
            "Describe only safety-relevant facts that are actually visible in the image, especially nearby vehicles.\n"
            "Do not infer anything about regions that are outside the camera view.\n"
            "If a region is not visible in the image, write exactly 'not_visible' for that row.\n"
            "Focus on these regions relative to the vehicle in the image: front, left-front, "
            "right-front, rear, left-rear, right-rear.\n\n"
            f"Return exactly this format. {region_rule}\n"
            "Front: ...\n"
            "Left-front: ...\n"
            "Right-front: ...\n"
            "Rear: ...\n"
            "Left-rear: ...\n"
            "Right-rear: ...\n"
        )

    def _build_visual_question_prompt(
        self,
        question_cfg: Dict[str, Any],
    ) -> str:
        question_id = str(question_cfg.get("id", "unknown_question"))
        query = str(question_cfg.get("query", "")).strip()
        positive = str(question_cfg.get("positive", "")).strip()
        negative = str(question_cfg.get("negative", "")).strip()
        return (
            "You are answering one cooperative-driving question from a single camera image.\n"
            f"Question ID: {question_id}\n"
            f"Query: {query}\n"
            f"Positive statement: {positive}\n"
            f"Negative statement: {negative}\n\n"
            "Use only what is visible in this image.\n"
            "Do not infer from scene context if the queried region is outside the camera view.\n"
            "If the queried region is not visible or too ambiguous, answer 'insufficient'.\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "answer": "positive" or "negative" or "insufficient",\n'
            '  "visibility_status": "visible" or "partial" or "not_visible",\n'
            '  "question_answerability": "answerable" or "partially_answerable" or "not_answerable",\n'
            '  "support_strength": "none" or "weak" or "moderate" or "strong",\n'
            '  "reason": "one short sentence"\n'
            "}\n\n"
            "Important rules:\n"
            "- Use 'negative' only if the queried region is visible enough and no vehicle is present there.\n"
            "- Use 'positive' only if a vehicle is actually supported by the visible evidence.\n"
            "- Use 'insufficient' if the queried region is not visible or too ambiguous.\n"
            "- Return JSON only."
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
            "You are evaluating textual cooperative-driving evidence for one binary question.\n"
            f"Question ID: {question_id}\n"
            f"Question type: {question_type}\n"
            f"Query: {query}\n"
            f"Positive statement: {positive}\n"
            f"Negative statement: {negative}\n\n"
            "Use only the provided evidence. Do not assume unseen facts.\n"
            "Treat text like 'not_visible' or missing region evidence as lack of visibility, not as proof of absence.\n"
            "If the evidence does not clearly support either side, do not guess.\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "answer": "positive" or "negative" or "insufficient",\n'
            '  "visibility_status": "visible" or "partial" or "not_visible",\n'
            '  "question_answerability": "answerable" or "partially_answerable" or "not_answerable",\n'
            '  "support_strength": "none" or "weak" or "moderate" or "strong",\n'
            '  "reason": "one short sentence"\n'
            "}\n\n"
            "Important rules:\n"
            "- Do not use 'strong' unless the evidence is explicit and unambiguous.\n"
            "- If evidence is partial, indirect, vague, or inferred, use at most 'moderate'.\n"
            "- If the evidence cannot reliably determine the answer, use answer='insufficient'.\n"
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
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }]
            text = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._vlm_processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }]
            text = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._vlm_processor(text=[text], padding=True, return_tensors="pt")

        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else 128,
            "do_sample": bool(self._vlm_do_sample),
        }
        if self._vlm_do_sample:
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

    def _compute_single_image_description(self, image: Image.Image, token_size: int) -> str:
        if image is None:
            raise ValueError("Image cannot be converted to PIL for Qwen2-VL captioning.")
        prompt = self._build_scene_description_prompt()
        return self._run_qwen_generation(prompt=prompt, image=image, max_new_tokens=token_size)

    def _compute_single_image_description_from_array(self, img_np, token_size: int):
        image = self._coerce_to_pil_image(img_np)
        if image is None:
            raise ValueError("img_np cannot be converted to PIL image")
        return self._compute_single_image_description(image, token_size)

    def _normalize_visibility_status(self, visibility: Any) -> str:
        normalized = str(visibility).strip().lower()
        if normalized in {"visible", "clear"}:
            return "visible"
        if normalized in {"partial", "partially_visible", "weak"}:
            return "partial"
        if normalized in {"not_visible", "not visible", "insufficient", "occluded", "unseen"}:
            return "not_visible"
        return "partial"

    def _normalize_question_answerability(self, answerability: Any, visibility_status: str) -> str:
        normalized = str(answerability).strip().lower()
        if normalized in {"answerable", "yes"}:
            return "answerable"
        if normalized in {"partially_answerable", "partial", "limited"}:
            return "partially_answerable"
        if normalized in {"not_answerable", "no", "insufficient"}:
            return "not_answerable"
        if visibility_status == "visible":
            return "answerable"
        if visibility_status == "partial":
            return "partially_answerable"
        return "not_answerable"

    def _parse_language_scores(self, raw_text: str) -> Dict[str, Any]:
        parsed = self._extract_first_json_object(raw_text) or {}

        def _clip01(value: Any, default: float) -> float:
            try:
                return float(max(0.0, min(1.0, float(value))))
            except Exception:
                return float(default)

        def _norm_answer(answer: Any) -> str:
            normalized = str(answer).strip().lower()
            if normalized in {"positive", "negative", "uncertain", "insufficient"}:
                return normalized
            return "uncertain"

        answer_raw = _norm_answer(parsed.get("answer", parsed.get("support_direction", "uncertain")))
        support_direction = str(parsed.get("support_direction", answer_raw)).strip().lower()
        support_strength = str(parsed.get("support_strength", "")).strip().lower()
        visibility = self._normalize_visibility_status(
            parsed.get("visibility_status", parsed.get("visibility", "partial"))
        )
        question_answerability = self._normalize_question_answerability(
            parsed.get("question_answerability", parsed.get("answerability", "")),
            visibility,
        )
        reason = str(parsed.get("reason", "")).strip()

        has_new_schema = (
            answer_raw in {"positive", "negative", "insufficient"}
            or support_direction in {"positive", "negative", "mixed", "insufficient"}
            or support_strength in {"none", "weak", "moderate", "strong"}
            or visibility in {"visible", "partial", "not_visible"}
        )

        if has_new_schema:
            strength_map = {"none": 0.0, "weak": 0.3, "moderate": 0.65, "strong": 0.9}
            visibility_map = {"visible": 1.0, "partial": 0.6, "not_visible": 0.0}
            answerability_map = {
                "answerable": 1.0,
                "partially_answerable": 0.5,
                "not_answerable": 0.0,
            }

            normalized_answer = answer_raw
            if normalized_answer not in {"positive", "negative", "insufficient"}:
                if support_direction in {"positive", "negative"}:
                    normalized_answer = support_direction
                else:
                    normalized_answer = "insufficient"

            base = float(strength_map.get(support_strength, 0.0))
            vis = float(visibility_map.get(visibility, 0.0))
            answerability_score = float(answerability_map.get(question_answerability, 0.0))

            if normalized_answer == "positive":
                evidence = base * max(vis, 0.35) * max(answerability_score, 0.5)
                pos, neg = evidence, 0.0
                unc = max(0.0, 1.0 - max(vis, answerability_score))
                answer = "positive"
            elif normalized_answer == "negative":
                evidence = base * max(vis, 0.35) * max(answerability_score, 0.5)
                pos, neg = 0.0, evidence
                unc = max(0.0, 1.0 - max(vis, answerability_score))
                answer = "negative"
            else:
                pos = 0.0
                neg = 0.0
                unc = 0.1 if question_answerability == "not_answerable" else 0.25
                evidence = 0.0
                answer = "uncertain"
        else:
            pos = _clip01(parsed.get("positive_score", 0.0), 0.0)
            neg = _clip01(parsed.get("negative_score", 0.0), 0.0)
            unc = _clip01(parsed.get("uncertainty", 1.0), 1.0)
            answer = _norm_answer(parsed.get("answer", "uncertain"))
            if answer == "uncertain":
                if pos > neg and pos > unc:
                    answer = "positive"
                elif neg > pos and neg > unc:
                    answer = "negative"
            if visibility == "partial" and unc >= 0.9:
                question_answerability = "not_answerable"
            evidence = float(1.0 - unc)

        belief = float(pos - neg)
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
            "visibility_status": visibility,
            "question_answerability": question_answerability,
            "answerability_score": {
                "answerable": 1.0,
                "partially_answerable": 0.5,
                "not_answerable": 0.0,
            }.get(question_answerability, 0.0),
            "raw_vlm_json": parsed,
            "raw_text": raw_text,
        }

    def _score_question_from_visual_evidence(
        self,
        question_cfg: Dict[str, Any],
        image: Image.Image,
    ) -> Dict[str, Any]:
        prompt = self._build_visual_question_prompt(question_cfg)
        raw_text = self._run_qwen_generation(
            prompt=prompt,
            image=image,
            max_new_tokens=self._vlm_score_max_new_tokens,
        )
        return self._parse_language_scores(raw_text)

    def _score_question_from_language_evidence(
        self,
        question_cfg: Dict[str, Any],
        fused_evidence: str,
    ) -> Dict[str, Any]:
        prompt = self._build_language_scoring_prompt(question_cfg, fused_evidence)
        raw_text = self._run_qwen_generation(
            prompt=prompt,
            image=None,
            max_new_tokens=self._vlm_score_max_new_tokens,
        )
        return self._parse_language_scores(raw_text)

    # =========================================================
    # Geometric utilities and aggregation
    # =========================================================
