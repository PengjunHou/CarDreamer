from __future__ import annotations

import math
from typing import Dict, List, Optional


class ConfidenceTracker:
    """Aggregates VLM confidence across questions; smooths with EMA + slew limit.

    Reads `_vlm_last_eval` produced by the VLM mixin and outputs a scalar
    confidence in [c_min, c_max] each step.
    """

    def __init__(
        self,
        ema_alpha: float = 0.15,
        max_delta_per_step: float = 0.05,
        c_min: float = 0.1,
        c_max: float = 1.0,
        c_init: float = 0.5,
        question_weights: Optional[Dict[str, float]] = None,
        source_field: str = "ego_plus_shared",
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.max_delta_per_step = float(max_delta_per_step)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.c_init = float(c_init)
        self.question_weights: Dict[str, float] = {
            str(k): float(v) for k, v in (question_weights or {}).items()
        }
        self.source_field = str(source_field)

        self._c_smooth: float = self.c_init
        self._c_out: float = self.c_init
        self._c_raw_last: Optional[float] = None
        self._last_consumed_eval_step: int = -1
        self._history: List[Dict[str, float]] = []

    def reset(self) -> None:
        self._c_smooth = self.c_init
        self._c_out = self.c_init
        self._c_raw_last = None
        self._last_consumed_eval_step = -1
        self._history = []

    def update(self, time_step: int, vlm_last_eval: Optional[Dict]) -> float:
        eval_step = -1
        if isinstance(vlm_last_eval, dict):
            try:
                eval_step = int(vlm_last_eval.get("step", -1))
            except (TypeError, ValueError):
                eval_step = -1

        if eval_step > self._last_consumed_eval_step:
            c_raw = self._extract_raw(vlm_last_eval)
            if c_raw is not None:
                self._c_raw_last = c_raw
            self._last_consumed_eval_step = eval_step

        c_raw_used = (
            self._c_raw_last if self._c_raw_last is not None else self.c_init
        )

        self._c_smooth = (
            (1.0 - self.ema_alpha) * self._c_smooth + self.ema_alpha * c_raw_used
        )

        delta = self._c_smooth - self._c_out
        if delta > self.max_delta_per_step:
            self._c_out += self.max_delta_per_step
        elif delta < -self.max_delta_per_step:
            self._c_out -= self.max_delta_per_step
        else:
            self._c_out = self._c_smooth

        if self._c_out < self.c_min:
            self._c_out = self.c_min
        elif self._c_out > self.c_max:
            self._c_out = self.c_max

        self._history.append(
            {
                "step": float(time_step),
                "c_raw": float(c_raw_used),
                "c_smooth": float(self._c_smooth),
                "c_out": float(self._c_out),
            }
        )
        return float(self._c_out)

    def _extract_raw(self, eval_dict: Optional[Dict]) -> Optional[float]:
        if not isinstance(eval_dict, dict):
            return None
        questions = eval_dict.get("questions")
        if not isinstance(questions, dict) or not questions:
            return None
        weighted_sum = 0.0
        weight_total = 0.0
        for qid, qres in questions.items():
            if not isinstance(qres, dict):
                continue
            block = qres.get(self.source_field)
            if not isinstance(block, dict):
                continue
            value = block.get("confidence")
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value_f):
                continue
            w = self.question_weights.get(str(qid), 1.0)
            weighted_sum += w * value_f
            weight_total += w
        if weight_total <= 0.0:
            return None
        return weighted_sum / weight_total

    @property
    def history(self) -> List[Dict[str, float]]:
        return list(self._history)

    @property
    def current(self) -> float:
        return float(self._c_out)
