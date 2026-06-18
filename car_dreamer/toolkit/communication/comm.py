from dataclasses import dataclass
from typing import Any, Dict, Tuple
import math
import numpy as np
import torch

def _safe_nbytes(payload: Any) -> int:
    """
    Best-effort estimate of payload size in bytes.
    - numpy arrays: use nbytes
    - bytes/bytearray: len
    - dict/list/tuple: sum recursive
    - scalars/strings: conservative fallback
    """
    if payload is None:
        return 0
    if isinstance(payload, (bytes, bytearray)):
        return int(len(payload))
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, str):
        # UTF-8 estimate
        return int(len(payload.encode("utf-8")))
    if isinstance(payload, (int, float, bool, np.number)):
        return 8
    if isinstance(payload, dict):
        return int(sum(_safe_nbytes(k) + _safe_nbytes(v) for k, v in payload.items()))
    if isinstance(payload, (list, tuple)):
        return int(sum(_safe_nbytes(x) for x in payload))
    # Fallback: rough estimate
    return 256


def _feature_nbytes(payload: Any) -> int:
    """Return the byte size of feature tensors carried alongside V2V payload metadata."""
    if payload is None:
        return 0
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, torch.Tensor):
        return int(payload.element_size() * payload.numel())
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    if isinstance(payload, dict):
        if "scene_description" in payload and payload["scene_description"] is not None:
            return _feature_nbytes(payload["scene_description"])
        if "feat" in payload:
            return _feature_nbytes(payload["feat"])
        if "data" in payload:
            return _feature_nbytes(payload["data"])
        return int(sum(_feature_nbytes(v) for v in payload.values()))
    if isinstance(payload, (list, tuple)):
        return int(sum(_feature_nbytes(x) for x in payload))
    return 0


@dataclass
class V2VMessage:
    """A single bundled V2V message (Communication Model §4).

    One message corresponds to one collaborator ``sender_id`` producing data at one sensor
    sampling time ``t_sense`` under policy ``policy_id`` and streaming it to the request
    vehicle ``receiver_id``. Per the project deviation from the design doc, a message bundles
    *all* modalities the collaborator sends under the current policy: ``modalities`` lists the
    types and ``data`` holds one field per modality (plus ``"feat"``/``"pose"``/``"vel"`` for
    the legacy vehicle-node graph).

    ``t_sense`` and ``t_recv`` are simulation **step** indices (the sensor tick and the delivery
    step); ``t_ready`` and ``t_send`` are continuous **seconds**; the delay fields are **seconds**:
        ``t_ready_s = t_sense * dt + T_proc``
        ``t_send_s  = max(t_ready_s, sender-queue busy_until)``     (queueing, §7)
        ``t_recv_s  = t_send_s + tx_delay``                          (transmission, §8)
        ``total_latency = t_recv_s - t_sense * dt`` == proc + queue + tx (summed exactly, §5)
        ``t_recv = t_sense + round(total_latency / dt)``  (the SUM is rounded to a step ONCE)
    """

    msg_id: int
    policy_id: int
    sender_id: int
    receiver_id: int
    modalities: Tuple[str, ...]
    payload_size: int
    data: Dict[str, Any]
    # --- per-message timeline ---
    t_sense: int          # sensor tick (step index)
    t_ready: float        # processing complete (seconds)
    t_send: float         # transmission start (seconds)
    t_recv: int           # delivery step (round of the summed latency)
    # --- delay decomposition (seconds) ---
    proc_delay: float = 0.0
    queue_delay: float = 0.0
    tx_delay: float = 0.0
    total_latency: float = 0.0
    distance_m: float = 0.0

    # ---- backward-compatible aliases (legacy graph_build / scripts) ----
    @property
    def created_step(self) -> int:
        return int(self.t_sense)

    @property
    def deliver_step(self) -> int:
        return int(self.t_recv)

    @property
    def latency_s(self) -> float:
        return float(self.total_latency)

    @property
    def payload_bytes(self) -> int:
        return int(self.payload_size)

    @property
    def payload(self) -> Dict[str, Any]:
        return self.data


def shannon_rate_bps(
    distance_m: float,
    bandwidth_hz: float,
    *,
    tx_power_dbm: float = 20.0,
    noise_figure_db: float = 9.0,
    carrier_freq_hz: float = 5.9e9,
    distance_decay_m: float = 60.0,
    min_rate_factor: float = 0.2,
) -> float:
    """Shannon link rate (bps) under a free-space-path-loss + noise-floor model.

    Used by the policy-conditioned per-message transmission (Communication Model §8), where
    ``bandwidth_hz`` is the policy-allocated bandwidth ``B^π_{m,q}``.

        FSPL(dB) = 20log10(d_km) + 20log10(f_MHz) + 32.44
        Pr(dBm)  = Pt(dBm) - FSPL
        N(dBm)   = -174 + 10log10(B) + NF
        C(bps)   = B * log2(1 + 10^((Pr - N)/10))

    ``distance_decay_m`` / ``min_rate_factor`` retain the original non-ideal attenuation factor.
    """
    if float(bandwidth_hz) <= 0.0:
        return 0.0
    d = max(float(distance_m), 0.0)
    distance_factor = max(float(min_rate_factor), math.exp(-d / max(float(distance_decay_m), 1e-6)))
    bandwidth_hz = max(float(bandwidth_hz) * distance_factor, 1.0)

    d_km = max(d, 1.0) / 1000.0
    f_mhz = float(carrier_freq_hz) / 1e6
    fspl_db = 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.44
    pr_dbm = float(tx_power_dbm) - fspl_db
    noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + float(noise_figure_db)
    snr_linear = 10.0 ** ((pr_dbm - noise_dbm) / 10.0)
    return max(bandwidth_hz * math.log2(1.0 + max(snr_linear, 0.0)), 1.0)
