from __future__ import annotations

from typing import Any, Dict


class ControlModulator:
    """Maps a confidence scalar c in [0, 1] to BasicAgent control parameters.

    Adjusts target_speed and obstacle detection threshold each step.
    Optionally modulates emergency-brake intensity.
    """

    def __init__(
        self,
        v_min_kmh: float = 8.0,
        v_max_kmh: float = 30.0,
        d_base_m: float = 5.0,
        beta_threshold: float = 1.0,
        modulate_brake: bool = False,
        max_brake_base: float = 0.5,
    ) -> None:
        self.v_min_kmh = float(v_min_kmh)
        self.v_max_kmh = float(v_max_kmh)
        self.d_base_m = float(d_base_m)
        self.beta_threshold = float(beta_threshold)
        self.modulate_brake = bool(modulate_brake)
        self.max_brake_base = float(max_brake_base)

    def apply(self, agent: Any, c_out: float) -> Dict[str, float]:
        c = float(c_out)
        if c < 0.0:
            c = 0.0
        elif c > 1.0:
            c = 1.0

        v_target_kmh = self.v_min_kmh + (self.v_max_kmh - self.v_min_kmh) * c
        # d_thresh = self.d_base_m * (1.0 + self.beta_threshold * (1.0 - c))

        agent.set_target_speed(float(v_target_kmh))
        # agent._base_vehicle_threshold = float(d_thresh)

        applied: Dict[str, float] = {
            "confidence": float(c),
            "target_speed_kmh": float(v_target_kmh),
            # "base_vehicle_threshold_m": float(d_thresh),
        }
        # if self.modulate_brake:
        #     mb = self.max_brake_base * (0.6 + 0.4 * c)
        #     agent._max_brake = float(mb)
        #     applied["max_brake"] = float(mb)
        return applied
