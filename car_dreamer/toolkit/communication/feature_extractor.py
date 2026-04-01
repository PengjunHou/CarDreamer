# feature_extractor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from .comm import _feature_nbytes


class MultiSizeCNNFeatureExtractor(nn.Module):
    """
    Shared CNN backbone -> 1024-d embedding, with projection heads to 256 and 64.
    Supports feature_size in {64, 256, 1024}.
    """
    def __init__(self):
        super().__init__()

        # A lightweight CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# /8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# /16
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # -> [B,256,1,1]
        self.fc_1024 = nn.Sequential(
            nn.Flatten(),                          # -> [B,256]
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
        )

        # Projection heads
        self.head_256 = nn.Linear(1024, 256)
        self.head_64  = nn.Linear(1024, 64)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        x: [B,3,H,W] float32 in [0,1]
        returns dict {1024: [B,1024], 256: [B,256], 64: [B,64]}
        """
        h = self.conv(x)
        h = self.pool(h)
        z1024 = self.fc_1024(h)
        z256 = self.head_256(z1024)
        z64 = self.head_64(z1024)
        return {1024: z1024, 256: z256, 64: z64}


@dataclass
class FeatureExtractorConfig:
    input_hw: int = 128           # resize image to input_hw x input_hw
    device: Optional[str] = None  # "cuda" / "cpu" / None(auto)


class FeatureExtractorService:
    """
    A thin wrapper providing:
      - image preprocessing (numpy uint8 HxWx3 -> torch float32 Bx3xHxW)
      - cached model and device placement
      - feature extraction for {64,256,1024}
    """
    def __init__(self, cfg: FeatureExtractorConfig = FeatureExtractorConfig()):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.model = MultiSizeCNNFeatureExtractor().to(self.device)
        self.model.eval()

    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        img: uint8 HxWx3 (RGB)
        returns: float32 [1,3,input_hw,input_hw] in [0,1]
        """
        if not isinstance(img, np.ndarray):
            raise TypeError(f"camera image must be np.ndarray, got {type(img)}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"camera image must be HxWx3, got shape={img.shape}")
        if img.dtype != np.uint8:
            # be tolerant
            img = img.astype(np.uint8, copy=False)

        # HWC -> CHW
        x = torch.from_numpy(img).to(torch.float32) / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

        # resize on tensor
        x = F.interpolate(
            x,
            size=(self.cfg.input_hw, self.cfg.input_hw),
            mode="bilinear",
            align_corners=False,
        )
        return x.to(self.device)

    @torch.no_grad()
    def extract(self, img: np.ndarray, feature_size: int) -> np.ndarray:
        if feature_size not in (64, 256, 1024):
            raise ValueError(f"feature_size must be one of {{64,256,1024}}, got {feature_size}")

        x = self._preprocess(img)
        feats = self.model.forward_features(x)[feature_size]  # [1,D]
        feat = feats[0].detach().to("cpu").to(torch.float32).numpy()
        # Ensure contiguous float32
        return np.ascontiguousarray(feat, dtype=np.float32)


# ---- singleton cache (so payload_fn doesn't re-create model repeatedly) ----
_EXTRACTOR_SINGLETON: Optional[FeatureExtractorService] = None


def get_extractor(cfg: Optional[FeatureExtractorConfig] = None) -> FeatureExtractorService:
    global _EXTRACTOR_SINGLETON
    if _EXTRACTOR_SINGLETON is None:
        _EXTRACTOR_SINGLETON = FeatureExtractorService(cfg or FeatureExtractorConfig())
    return _EXTRACTOR_SINGLETON


# def  payload_fn_cnn(sender, obs: Dict[str, Any], feature_size: int) -> Dict[str, Any]:
#     """
#     Your payload_fn(sender, obs, feature_size) implementation.

#     obs format:
#       {"camera": np.ndarray(H,W,3,uint8), "message": str}
#     We ignore text embedding for now.
#     """
#     img = obs.get("camera", None)
#     text = obs.get("message", "")

#     # Always return fixed-size feature (float32[feature_size])
#     if img is None:
#         feat = np.zeros((feature_size,), dtype=np.float32)
#         return {"feat": feat, "feat_dim": feature_size, "has_image": False, "text": text}

#     extractor = get_extractor()
#     feat = extractor.extract(img, feature_size)
#     return {"feat": feat, "feat_dim": feature_size, "has_image": True, "text": text}
# def payload_fn_cnn(sender, obs, feature_size):
#     img = obs.get("camera", None)
#     text = obs.get("message", "")
#     feat = img

#     # if img is None:
#     #     feat = np.zeros((feature_size,), dtype=np.float32)
#     #     return {"feat": feat, "feat_dim": feature_size, "has_image": False, "text": text}

#     # side = int(np.sqrt(feature_size / 3))
#     # resized = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
#     # flat = resized.astype(np.float32).reshape(-1) / 255.0

#     # # 如果不完全匹配，再截断/补零
#     # if flat.shape[0] >= feature_size:
#     #     feat = flat[:feature_size]
#     # else:
#     #     feat = np.zeros((feature_size,), dtype=np.float32)
#     #     feat[:flat.shape[0]] = flat

#     return {"feat": feat, "feat_dim": feature_size, "has_image": True, "text": text}

from typing import Any
import numpy as np


def _safe_to_text(value: Any, max_len: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = str(value)
    else:
        raise ValueError(f"unsupported type: {type(value)}")
    return text[:max_len]


def payload_fn_llm(sender, obs, feature_size, *args, image_proc_fn=None, **kwargs):
    img = obs.get("camera", None)
    raw_message = obs.get("message", "")
    raw_message_text = _safe_to_text(raw_message)

    has_image = bool(img is not None)
    scene_description = ""
    feat = np.zeros((feature_size,), dtype=np.float32)

    if has_image and image_proc_fn is not None:
        try:
            proc_out = image_proc_fn(img, feature_size) # feature_size 就是token size

            # 情况1：返回字符串 -> 当作 scene_description
            if isinstance(proc_out, str):
                scene_description = proc_out.strip()

            # 情况2：返回 dict -> 支持同时给 feat 和 scene_description
            elif isinstance(proc_out, dict):
                if "scene_description" in proc_out and proc_out["scene_description"] is not None:
                    scene_description = str(proc_out["scene_description"]).strip()

                if "feat" in proc_out and proc_out["feat"] is not None:
                    feat = proc_out["feat"]

            # 情况3：其他类型 -> 默认当作 feat
            else:
                feat = proc_out

        except Exception as exc:
            scene_description = f"image_proc_fn failed: {type(exc).__name__}: {exc}"

    parts = []
    if raw_message_text:
        parts.append(f"observer_message: {raw_message_text}")
    if scene_description:
        parts.append(f"scene_description: {scene_description}")
    merged_text = "\n".join(parts)

    return {
        "feat_dim": feature_size,       # TODO: 暂时没用
        "has_image": has_image,
        "img_emb": None, #feat,
        "text": merged_text,
        "scene_description": scene_description,
    }
    
def payload_fn_cnn(sender, obs, feature_size, image_embed_fn):
    img = obs.get("camera", None)
    raw_message = obs.get("message", "")
    raw_message_text = _safe_to_text(raw_message)

    has_image = bool(img is not None)

    if has_image and image_embed_fn is not None:
        print("Extracting CNN features from image...")
        feat = image_embed_fn(img) # feature_size 就是token size
    else:
        feat = img

    return {"feat": feat, "feat_dim": feature_size, "has_image": True, "text": raw_message}