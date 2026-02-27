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

def _feature_nbytes(payload: Any) -> int:
    """
    Only count the feature tensor/array bytes.
    Supports payload formats:
        - np.ndarray: treat as feature itself
        - dict with key 'feat' or 'data': np.ndarray
        - nested dict/list/tuple: best-effort search; strings/metadata ignored
    """
    if payload is None:
        return 0
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, dict):
        # common keys
        if "feat" in payload and isinstance(payload["feat"], np.ndarray):
            return int(payload["feat"].nbytes)
        if "data" in payload and isinstance(payload["data"], np.ndarray):
            return int(payload["data"].nbytes)
        # best-effort: sum feature bytes in values recursively
        total = 0
        for v in payload.values():
            total += _feature_nbytes(v)
        return int(total)
    if isinstance(payload, (list, tuple)):
        return int(sum(_feature_nbytes(x) for x in payload))
    # strings/metadata do not count as "feature bytes"
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
    sender_id: int
    receiver_id: int
    group_id: int
    payload: Any
    payload_bytes: int
    created_step: int
    deliver_step: int
    latency_s: float
    distance_m: float

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

        # distance attenuation factor (keep your original factor)
        distance_factor = math.exp(-d / max(self.distance_decay_m, 1e-6))
        distance_factor = max(self.min_rate_factor, float(distance_factor))

        # ---- Shannon capacity part ----
        # Effective bandwidth under contention + distance degradation
        B_sender = max(float(sender_res.bandwidth_hz) / out_degree, 1.0)
        B_receiver = max(float(receiver_res.bandwidth_hz) / in_degree, 1.0)
        bandwidth_hz = min(B_sender, B_receiver) * distance_factor
        bandwidth_hz = max(bandwidth_hz, 1.0)

        # Free-space path loss (FSPL)
        # FSPL(dB) = 20log10(d_km) + 20log10(f_MHz) + 32.44
        d_km = max(d, 1.0) / 1000.0
        f_mhz = float(sender_res.carrier_freq_hz) / 1e6
        fspl_db = 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.44

        # Received power (dBm)
        pr_dbm = float(sender_res.tx_power_dbm) - fspl_db

        # Noise floor (dBm): -174 dBm/Hz + 10log10(B) + NF
        noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + float(receiver_res.noise_figure_db)

        # SNR
        snr_db = pr_dbm - noise_dbm
        snr_linear = 10.0 ** (snr_db / 10.0)

        # Shannon capacity (bps)
        shannon_bps = bandwidth_hz * math.log2(1.0 + max(snr_linear, 0.0))

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
