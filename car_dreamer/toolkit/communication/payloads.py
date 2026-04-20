from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from PIL import Image


PAYLOAD_TYPE_ORDER: tuple[str, ...] = (
    "object_list",
    "occupancy",
    "images",
    "latent",
    "tokens",
)
DEFAULT_PAYLOAD_TYPE = "tokens"
DEFAULT_PAYLOAD_ENCODER_ID = "tokens_v1"


def canonicalize_payload_type(payload_type: str | None) -> str:
    payload_type = str(payload_type or DEFAULT_PAYLOAD_TYPE).strip().lower()
    if payload_type not in PAYLOAD_TYPE_ORDER:
        return DEFAULT_PAYLOAD_TYPE
    return payload_type


def get_payload_type_order() -> tuple[str, ...]:
    return PAYLOAD_TYPE_ORDER


def payload_type_to_one_hot(payload_type: str | None) -> np.ndarray:
    payload_type = canonicalize_payload_type(payload_type)
    return np.asarray(
        [1.0 if payload_type == payload_name else 0.0 for payload_name in PAYLOAD_TYPE_ORDER],
        dtype=np.float32,
    )


def _safe_to_text(value: Any, max_len: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = str(value)
    else:
        text = str(value)
    return text[:max_len]


def _coerce_to_pil_image(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return Image.fromarray(arr).convert("RGB")


def _empty_feature(feature_size: int) -> np.ndarray:
    return np.zeros((max(int(feature_size), 0),), dtype=np.float32)


@dataclass(frozen=True)
class PayloadEncoding:
    payload_type: str
    payload_encoder_id: str
    data: Any
    data_nbytes: int
    feat: Any
    feat_dim: int
    scene_description: str = ""
    text: str = ""
    has_image: bool = False
    img_emb: Any = None

    def to_payload_dict(self) -> Dict[str, Any]:
        return {
            "payload_type": str(self.payload_type),
            "payload_encoder_id": str(self.payload_encoder_id),
            "data": self.data,
            "data_nbytes": int(self.data_nbytes),
            "feat": self.feat,
            "feat_dim": int(self.feat_dim),
            "scene_description": str(self.scene_description),
            "text": str(self.text),
            "has_image": bool(self.has_image),
            "img_emb": self.img_emb,
        }


@dataclass(frozen=True)
class PayloadSelectorDecision:
    payload_type: str
    payload_encoder_id: str
    reason: str = ""
    overridden: bool = False


class PayloadEncoder:
    payload_type: str = DEFAULT_PAYLOAD_TYPE
    payload_encoder_id: str = DEFAULT_PAYLOAD_ENCODER_ID

    def encode(
        self,
        sender: Any,
        obs: Mapping[str, Any],
        feature_size: int,
        *,
        image_proc_fn: Optional[Callable[..., Any]] = None,
        jpeg_quality: int = 80,
        **kwargs,
    ) -> PayloadEncoding:
        raise NotImplementedError


class UnsupportedPayloadEncoder(PayloadEncoder):
    def __init__(self, payload_type: str, payload_encoder_id: str):
        self.payload_type = canonicalize_payload_type(payload_type)
        self.payload_encoder_id = str(payload_encoder_id)

    def encode(
        self,
        sender: Any,
        obs: Mapping[str, Any],
        feature_size: int,
        *,
        image_proc_fn: Optional[Callable[..., Any]] = None,
        jpeg_quality: int = 80,
        **kwargs,
    ) -> PayloadEncoding:
        del sender, obs, feature_size, image_proc_fn, jpeg_quality, kwargs
        raise NotImplementedError(
            f"Payload encoder '{self.payload_encoder_id}' for type '{self.payload_type}' is not implemented."
        )


class TokensPayloadEncoder(PayloadEncoder):
    payload_type = "tokens"
    payload_encoder_id = DEFAULT_PAYLOAD_ENCODER_ID

    def encode(
        self,
        sender: Any,
        obs: Mapping[str, Any],
        feature_size: int,
        *,
        image_proc_fn: Optional[Callable[..., Any]] = None,
        jpeg_quality: int = 80,
        **kwargs,
    ) -> PayloadEncoding:
        del sender, jpeg_quality, kwargs
        img = obs.get("camera", None)
        raw_message_text = _safe_to_text(obs.get("message", ""))
        scene_description = ""
        if img is not None and image_proc_fn is not None:
            try:
                proc_out = image_proc_fn(img, feature_size)
                if isinstance(proc_out, str):
                    scene_description = proc_out.strip()
                elif isinstance(proc_out, Mapping):
                    scene_description = str(proc_out.get("scene_description", "")).strip()
            except Exception as exc:  # pragma: no cover - exercised in runtime fallback
                scene_description = f"image_proc_fn failed: {type(exc).__name__}: {exc}"
        parts = []
        if raw_message_text:
            parts.append(f"observer_message: {raw_message_text}")
        if scene_description:
            parts.append(f"scene_description: {scene_description}")
        merged_text = "\n".join(parts)
        data = merged_text.encode("utf-8")
        return PayloadEncoding(
            payload_type=self.payload_type,
            payload_encoder_id=self.payload_encoder_id,
            data=data,
            data_nbytes=len(data),
            feat=_empty_feature(feature_size),
            feat_dim=0,
            scene_description=scene_description,
            text=merged_text,
            has_image=bool(img is not None),
            img_emb=None,
        )


class ImagesPayloadEncoder(PayloadEncoder):
    payload_type = "images"
    payload_encoder_id = "images_v1"

    def encode(
        self,
        sender: Any,
        obs: Mapping[str, Any],
        feature_size: int,
        *,
        image_proc_fn: Optional[Callable[..., Any]] = None,
        jpeg_quality: int = 80,
        **kwargs,
    ) -> PayloadEncoding:
        del sender, image_proc_fn, kwargs
        image = _coerce_to_pil_image(obs.get("camera", None))
        if image is None:
            data = b""
        else:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=max(min(int(jpeg_quality), 100), 1))
            data = buffer.getvalue()
        return PayloadEncoding(
            payload_type=self.payload_type,
            payload_encoder_id=self.payload_encoder_id,
            data=data,
            data_nbytes=len(data),
            feat=_empty_feature(feature_size),
            feat_dim=0,
            scene_description="",
            text="",
            has_image=image is not None,
            img_emb=None,
        )


class PayloadEncoderRegistry:
    def __init__(self, encoders: Optional[Iterable[PayloadEncoder]] = None) -> None:
        self._encoders_by_type: Dict[str, Dict[str, PayloadEncoder]] = {
            payload_type: {} for payload_type in PAYLOAD_TYPE_ORDER
        }
        self._default_encoder_by_type: Dict[str, str] = {}
        if encoders is not None:
            for encoder in encoders:
                self.register(encoder.payload_type, encoder, default=True)

    def register(self, payload_type: str, encoder: PayloadEncoder, *, default: bool = True) -> None:
        payload_type = canonicalize_payload_type(payload_type)
        self._encoders_by_type.setdefault(payload_type, {})
        self._encoders_by_type[payload_type][str(encoder.payload_encoder_id)] = encoder
        if default or payload_type not in self._default_encoder_by_type:
            self._default_encoder_by_type[payload_type] = str(encoder.payload_encoder_id)

    def get(self, payload_type: str, encoder_id: str | None = None) -> PayloadEncoder:
        payload_type = canonicalize_payload_type(payload_type)
        available = self._encoders_by_type.get(payload_type, {})
        if not available:
            raise KeyError(f"No payload encoders registered for type '{payload_type}'.")
        resolved_id = str(encoder_id or self._default_encoder_by_type.get(payload_type, ""))
        if resolved_id not in available:
            raise KeyError(
                f"Unknown payload encoder '{resolved_id}' for type '{payload_type}'. "
                f"Available: {sorted(available)}"
            )
        return available[resolved_id]

    def default_encoder_id(self, payload_type: str) -> str:
        payload_type = canonicalize_payload_type(payload_type)
        if payload_type not in self._default_encoder_by_type:
            raise KeyError(f"No default payload encoder registered for type '{payload_type}'.")
        return self._default_encoder_by_type[payload_type]

    def has_type(self, payload_type: str) -> bool:
        payload_type = canonicalize_payload_type(payload_type)
        return bool(self._encoders_by_type.get(payload_type))


class RuleBasedPayloadSelector:
    selector_id = "default"

    def __call__(
        self,
        *,
        sender_id: int,
        bandwidth: float,
        distance_m: float,
        latest_comm_stats: Mapping[str, Any] | None,
        registry: PayloadEncoderRegistry,
        enabled_types: Sequence[str],
    ) -> PayloadSelectorDecision:
        del sender_id
        enabled = [canonicalize_payload_type(item) for item in enabled_types if registry.has_type(item)]
        if not enabled:
            enabled = [DEFAULT_PAYLOAD_TYPE]
        latest_comm_stats = dict(latest_comm_stats or {})
        feasible = float(latest_comm_stats.get("comm_feasible", 1.0))
        link_rate_bps = float(latest_comm_stats.get("link_rate_bps", 0.0))
        preferred = DEFAULT_PAYLOAD_TYPE
        reason = "default_tokens"
        if feasible <= 0.5 or link_rate_bps < 2.0e6:
            preferred = "tokens"
            reason = "low_link_capacity"
        elif float(bandwidth) >= 0.45 and float(distance_m) <= 12.0:
            preferred = "images"
            reason = "high_bandwidth_near_sender"
        else:
            preferred = "tokens"
            reason = "default_tokens"
        if preferred not in enabled:
            preferred = enabled[0]
            reason = f"{reason}_fallback_enabled"
        return PayloadSelectorDecision(
            payload_type=preferred,
            payload_encoder_id=registry.default_encoder_id(preferred),
            reason=reason,
            overridden=False,
        )


def build_default_payload_registry() -> PayloadEncoderRegistry:
    registry = PayloadEncoderRegistry()
    registry.register("object_list", UnsupportedPayloadEncoder("object_list", "object_list_v1"))
    registry.register("occupancy", UnsupportedPayloadEncoder("occupancy", "occupancy_v1"))
    registry.register("images", ImagesPayloadEncoder())
    registry.register("latent", UnsupportedPayloadEncoder("latent", "latent_v1"))
    registry.register("tokens", TokensPayloadEncoder())
    return registry


def decode_payload_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload_type = canonicalize_payload_type(payload.get("payload_type"))
    payload_encoder_id = str(payload.get("payload_encoder_id", ""))
    data = payload.get("data")
    scene_description = str(payload.get("scene_description", "")).strip()
    text = str(payload.get("text", "")).strip()
    image: Optional[Image.Image] = None
    if payload_type == "images" and data:
        try:
            image = Image.open(BytesIO(bytes(data))).convert("RGB")
        except Exception:
            image = None
    elif payload_type == "tokens" and data and not text:
        try:
            text = bytes(data).decode("utf-8")
        except Exception:
            text = ""
    return {
        "payload_type": payload_type,
        "payload_encoder_id": payload_encoder_id or (
            "images_v1" if payload_type == "images" else DEFAULT_PAYLOAD_ENCODER_ID
        ),
        "image": image,
        "scene_description": scene_description,
        "text": text,
        "data_nbytes": int(payload.get("data_nbytes", len(data) if isinstance(data, (bytes, bytearray)) else 0)),
    }
