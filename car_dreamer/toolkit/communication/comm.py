from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from dataclasses import dataclass
import math
import carla
import random
import numpy as np


from ..utils import _dist_m

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


# -----------------------------
# Feature-only size accounting
# -----------------------------

from typing import Any
import numpy as np
import torch


def _feature_nbytes(payload: Any) -> int:
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
            if isinstance(payload["feat"], np.ndarray):
                return int(payload["feat"].nbytes)
            if isinstance(payload["feat"], torch.Tensor):
                nbytes = payload["feat"].element_size() * payload["feat"].numel()
                return int(nbytes)

        if "data" in payload:
            if isinstance(payload["data"], np.ndarray):
                return int(payload["data"].nbytes)
            if isinstance(payload["data"], torch.Tensor):
                return int(payload["data"].element_size() * payload["data"].numel())

        total = 0
        for v in payload.values():
            total += _feature_nbytes(v)
        return int(total)

    if isinstance(payload, (list, tuple)):
        return int(sum(_feature_nbytes(x) for x in payload))

    return 0

def _tx_bytes_for_latency(payload: Any, overhead_bytes: int) -> int:
    """
    Bytes used for transmission/latency calculation:
        feature_bytes + fixed overhead bytes

    NOTE: This intentionally ignores text bytes and dict key bytes,
            to match your current research assumption.
    """
    feature_bytes = _feature_nbytes(payload)
    return int(feature_bytes + max(int(overhead_bytes), 0))

# -----------------------------
# Communication / Latency Model
# -----------------------------


@dataclass(frozen=True)
class NetResource:
    """
    Simple per-vehicle network resource.

    uplink_bps/downlink_bps:
        Kept for backward compatibility and as a practical throughput cap.

    bandwidth_hz/tx_power_dbm/noise_figure_db/carrier_freq_hz:
        Used to estimate Shannon capacity (rate) with a simple path-loss model.
    """
    uplink_bps: float
    downlink_bps: float

    # Shannon parameters (defaults are reasonable for V2X-like settings)
    bandwidth_hz: float = 10e6         # 10 MHz
    tx_power_dbm: float = 20.0         # 100 mW
    noise_figure_db: float = 9.0       # receiver noise figure
    carrier_freq_hz: float = 5.9e9     # 5.9 GHz


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

    Shared by :class:`SimpleWirelessLatency` (legacy contention path) and the new
    policy-conditioned per-message transmission (Communication Model §8), where ``bandwidth_hz``
    is the policy-allocated bandwidth ``B^π_{m,q}`` rather than a contention share.

        FSPL(dB) = 20log10(d_km) + 20log10(f_MHz) + 32.44
        Pr(dBm)  = Pt(dBm) - FSPL
        N(dBm)   = -174 + 10log10(B) + NF
        C(bps)   = B * log2(1 + 10^((Pr - N)/10))

    ``distance_decay_m`` / ``min_rate_factor`` retain the original non-ideal attenuation factor.
    """
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


class LatencyModel:
    """
    Latency model interface.
    """

    def compute_latency_s(
        self,
        sender: carla.Actor,
        receiver: carla.Actor,
        payload_size_bytes: int,
        sender_res: NetResource,
        receiver_res: NetResource,
        out_degree: int,
        in_degree: int,
        **kwargs,
    ) -> float:
        raise NotImplementedError


class SimpleWirelessLatency(LatencyModel):
    """
    A simple V2V latency model:
        latency = base_rtt + proc_delay + tx_time

    tx_time is computed using a Shannon-capacity-like rate:
      C = B * log2(1 + SNR)

    Where:
        - Effective bandwidth B is contention-aware and distance-degraded:
          B = min(sender_B/out_degree, receiver_B/in_degree) * distance_factor
        - SNR is estimated with a free-space path loss + noise floor model:
            Pr(dBm) = Pt(dBm) - FSPL(dB)
            N(dBm)  = -174 + 10log10(B) + NF
            SNR(dB) = Pr - N
        - Practical throughput is capped by min(uplink, downlink) (backward compat)

    distance_factor is also retained to capture non-ideal attenuation/conditions.
    """

    def __init__(
        self,
        base_rtt_s: float = 0.02,
        proc_delay_s: float = 0.005,
        distance_decay_m: float = 100.0,
        min_rate_factor: float = 0.2,
        jitter_s: float = 0.0,
        rng: Optional[random.Random] = None,
        overhead_bytes: int = 120,
    ):
        self.base_rtt_s = float(base_rtt_s)
        self.proc_delay_s = float(proc_delay_s)
        self.distance_decay_m = float(distance_decay_m)
        self.min_rate_factor = float(min_rate_factor)
        self.jitter_s = float(jitter_s)
        self.rng = rng
        self.overhead_bytes = int(overhead_bytes)

    def compute_latency_s(
        self,
        sender: carla.Actor,
        receiver: carla.Actor,
        payload_size_bytes: int,
        sender_res: NetResource,
        receiver_res: NetResource,
        out_degree: int,
        in_degree: int,
        **kwargs,
    ) -> float:
        d = _dist_m(sender, receiver)

        # contention-aware caps (KEEP variable names)
        out_degree = max(int(out_degree), 1)
        in_degree = max(int(in_degree), 1)
        uplink = max(sender_res.uplink_bps / out_degree, 1.0)
        downlink = max(receiver_res.downlink_bps / in_degree, 1.0)

        # ---- Shannon capacity part ----
        # Effective bandwidth under contention; FSPL/SNR + distance degradation in shannon_rate_bps.
        B_sender = max(float(sender_res.bandwidth_hz) / out_degree, 1.0)
        B_receiver = max(float(receiver_res.bandwidth_hz) / in_degree, 1.0)
        shannon_bps = shannon_rate_bps(
            d,
            min(B_sender, B_receiver),
            tx_power_dbm=float(sender_res.tx_power_dbm),
            noise_figure_db=float(receiver_res.noise_figure_db),
            carrier_freq_hz=float(sender_res.carrier_freq_hz),
            distance_decay_m=self.distance_decay_m,
            min_rate_factor=self.min_rate_factor,
        )

        # Practical throughput cap (KEEP uplink/downlink variables)
        rate_bps = min(shannon_bps, uplink, downlink)
        rate_bps = max(rate_bps, 1.0)

        # Transmission time
        tx_time = (8.0 * float(payload_size_bytes)) / rate_bps

        jitter = 0.0
        if self.jitter_s > 0 and self.rng is not None:
            jitter = self.rng.uniform(-self.jitter_s, self.jitter_s)

        latency = self.base_rtt_s + 2.0 * self.proc_delay_s + tx_time + jitter
        return max(latency, 0.0)
