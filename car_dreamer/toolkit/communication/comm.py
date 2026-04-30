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
        if "data_nbytes" in payload:
            try:
                return int(payload["data_nbytes"])
            except Exception:
                pass

        if "payload_type" in payload and "data" in payload:
            if isinstance(payload["data"], np.ndarray):
                return int(payload["data"].nbytes)
            if isinstance(payload["data"], torch.Tensor):
                return int(payload["data"].element_size() * payload["data"].numel())
            if isinstance(payload["data"], (bytes, bytearray)):
                return int(len(payload["data"]))
            if isinstance(payload["data"], str):
                return int(len(payload["data"].encode("utf-8")))

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

    bandwidth_hz/tx_power_dbm/noise_figure_db/carrier_freq_hz:
        Used to estimate Shannon capacity (rate) with a simple path-loss model.
    """
    bandwidth_hz: float
    tx_power_dbm: float = 20.0         # 100 mW
    noise_figure_db: float = 9.0       # receiver noise figure
    carrier_freq_hz: float = 5.9e9     # 5.9 GHz


@dataclass(frozen=True)
class LinkCapacityAnalysis:
    distance_m: float
    payload_size_bytes: int
    payload_size_bits: float
    required_load_bps: float
    link_rate_bps: float
    shannon_bps: float
    bandwidth_hz: float
    snr_db: float
    feasible: bool
    latency_s: float
    tx_time_s: float = 0.0
    processing_delay_s: float = 0.0


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

    def analyze_transmission(
        self,
        sender: carla.Actor,
        receiver: carla.Actor,
        payload_size_bytes: int,
        sender_res: NetResource,
        receiver_res: NetResource,
        out_degree: int,
        in_degree: int,
        *,
        alpha: float = 1.0,
        nu: float = 1.0,
        fixed_dt: float = 0.1,
        **kwargs,
    ) -> LinkCapacityAnalysis:
        raise NotImplementedError


_PATHLOSS_COEFS: Dict[str, Tuple[float, float, float]] = {
    # name: (intercept_db, beta_d, gamma_fc) for PL = intercept + beta_d*log10(d_m) + gamma_fc*log10(fc_GHz)
    # 3GPP TR 37.885 V2V channel models.
    "urban_los":   (38.77, 16.7, 18.2),
    "urban_nlos":  (36.85, 30.0, 18.9),
    "highway_los": (32.40, 20.0, 20.0),
}


class SimpleWirelessLatency(LatencyModel):
    """
    Two-segment V2V latency model:
        latency = t_overhead + t_tx + jitter
        t_overhead  = overhead_base_s + overhead_per_kb_s * payload_kb
        t_tx        = 8 * payload_bytes / shannon_bps
        shannon_bps = B_alloc * log2(1 + SNR_linear)
        SNR_dB      = Pt - PL(d, fc) - N0(B_alloc) - margin_db
        PL(d, fc)   = intercept + beta_d * log10(d_m) + gamma_fc * log10(fc_GHz)

    `t_overhead` lumps all application-layer / MAC-layer overhead that is
    insensitive to the physical channel (encoding, serialization, decoding,
    resource selection). `t_tx` carries the SNR-driven part of the latency
    and is the only term that depends on distance and bandwidth allocation.

    `pathloss_model` selects the 3GPP coefficients (urban_los, urban_nlos,
    highway_los). `margin_db` lumps shadow fading / vehicle blockage / RF
    implementation losses into one configurable knob; if `margin_sigma_db`
    is positive, the margin is sampled per call as N(margin_db, sigma^2).

    The `out_degree`/`in_degree` arguments are kept for API compatibility
    but no longer applied after policy bandwidth allocation. Bandwidth
    sharing across selected members must come from the upstream allocator
    (each `NetResource` carries that member's already-allocated share).
    """

    def __init__(
        self,
        overhead_base_s: float = 0.030,
        overhead_per_kb_s: float = 0.0015,
        pathloss_model: str = "urban_los",
        margin_db: float = 10.0,
        margin_sigma_db: float = 0.0,
        jitter_s: float = 0.0,
        rng: Optional[random.Random] = None,
        overhead_bytes: int = 64,
    ):
        self.overhead_base_s = float(overhead_base_s)
        self.overhead_per_kb_s = float(overhead_per_kb_s)
        if pathloss_model not in _PATHLOSS_COEFS:
            raise ValueError(
                f"unknown pathloss_model={pathloss_model!r}; expected one of {list(_PATHLOSS_COEFS)}"
            )
        self.pathloss_model = str(pathloss_model)
        self._pathloss_coefs = _PATHLOSS_COEFS[self.pathloss_model]
        self.margin_db = float(margin_db)
        self.margin_sigma_db = float(margin_sigma_db)
        self.jitter_s = float(jitter_s)
        self.rng = rng
        self.overhead_bytes = int(overhead_bytes)

    def _overhead_s(self, payload_size_bytes: int) -> float:
        payload_kb = max(float(payload_size_bytes), 0.0) / 1024.0
        return max(self.overhead_base_s + self.overhead_per_kb_s * payload_kb, 0.0)

    def _path_loss_db(self, d_m: float, fc_hz: float) -> float:
        intercept, beta_d, gamma_fc = self._pathloss_coefs
        fc_ghz = max(float(fc_hz) / 1e9, 1e-3)
        return intercept + beta_d * math.log10(max(d_m, 1.0)) + gamma_fc * math.log10(fc_ghz)

    def _sample_margin_db(self) -> float:
        if self.margin_sigma_db <= 0.0 or self.rng is None:
            return self.margin_db
        return self.margin_db + self.rng.gauss(0.0, self.margin_sigma_db)

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
        analysis = self.analyze_transmission(
            sender=sender,
            receiver=receiver,
            payload_size_bytes=payload_size_bytes,
            sender_res=sender_res,
            receiver_res=receiver_res,
            out_degree=out_degree,
            in_degree=in_degree,
            alpha=float(kwargs.get("alpha", 1.0)),
            nu=float(kwargs.get("nu", 1.0)),
            fixed_dt=float(kwargs.get("fixed_dt", 0.1)),
        )
        return float(analysis.latency_s)

    def analyze_transmission(
        self,
        sender: carla.Actor,
        receiver: carla.Actor,
        payload_size_bytes: int,
        sender_res: NetResource,
        receiver_res: NetResource,
        out_degree: int,
        in_degree: int,
        *,
        alpha: float = 1.0,
        nu: float = 1.0,
        fixed_dt: float = 0.1,
        **kwargs,
    ) -> LinkCapacityAnalysis:
        d = _dist_m(sender, receiver)

        # Kept for API compatibility; bandwidth contention is upstream now.
        out_degree = max(int(out_degree), 1)
        in_degree = max(int(in_degree), 1)
        del out_degree, in_degree

        # ---- Shannon capacity part ----
        # Effective bandwidth = policy-allocated share (sender ∩ receiver).
        B_sender = max(float(sender_res.bandwidth_hz), 1.0)
        B_receiver = max(float(receiver_res.bandwidth_hz), 1.0)
        bandwidth_hz = max(min(B_sender, B_receiver), 1.0)

        # 3GPP V2V path loss (selected by self.pathloss_model)
        pathloss_db = self._path_loss_db(d, float(sender_res.carrier_freq_hz))

        # Received power (dBm)
        pr_dbm = float(sender_res.tx_power_dbm) - pathloss_db

        # Noise floor (dBm): -174 dBm/Hz + 10log10(B) + NF
        noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + float(receiver_res.noise_figure_db)

        # SNR with lumped fading / blockage / implementation margin
        margin_db = self._sample_margin_db()
        snr_db = pr_dbm - noise_dbm - margin_db
        snr_linear = 10.0 ** (snr_db / 10.0)

        # Shannon capacity (bps)
        shannon_bps = bandwidth_hz * math.log2(1.0 + max(snr_linear, 0.0))
        rate_bps = max(shannon_bps, 1.0)

        # Transmission time
        tx_time = (8.0 * float(payload_size_bytes)) / rate_bps

        jitter = 0.0
        if self.jitter_s > 0 and self.rng is not None:
            jitter = self.rng.uniform(-self.jitter_s, self.jitter_s)

        processing_delay = self._overhead_s(payload_size_bytes)
        latency = processing_delay + tx_time + jitter
        payload_bits = 8.0 * float(payload_size_bytes)
        required_load_bps = (
            max(float(alpha), 0.0) * max(float(nu), 0.0) * payload_bits / max(float(fixed_dt), 1e-6)
        )
        link_rate_bps = max(rate_bps, 1.0)
        return LinkCapacityAnalysis(
            distance_m=float(d),
            payload_size_bytes=int(payload_size_bytes),
            payload_size_bits=float(payload_bits),
            required_load_bps=float(required_load_bps),
            link_rate_bps=float(link_rate_bps),
            shannon_bps=float(shannon_bps),
            bandwidth_hz=float(bandwidth_hz),
            snr_db=float(snr_db),
            feasible=bool(required_load_bps <= link_rate_bps + 1e-9),
            latency_s=float(max(latency, 0.0)),
            tx_time_s=float(max(tx_time, 0.0)),
            processing_delay_s=float(processing_delay),
        )
