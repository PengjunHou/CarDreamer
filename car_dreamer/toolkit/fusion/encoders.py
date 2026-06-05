"""Frozen encoders for Route-L3 (convenience for offline embedding + inference).

The L3 fusion model itself is backbone-agnostic and consumes *precomputed*
embeddings. These wrappers exist so the data-prep / inference scripts can produce
those embeddings consistently. Heavy imports are lazy so importing this module (and
unit-testing the fusion head) never requires CLIP / SigLIP / sentence-transformers.

Recommended setup:
  * ego image  -> ``ImageEmbedder`` (CLIP image encoder)
  * neighbor captions, target scene descriptions, and action statements ->
    the SAME ``TextEmbedder`` (so they share one *output space*; pick a strong
    text-text encoder such as SigLIP-text or sentence-transformers, NOT raw CLIP
    text, whose text-text similarity is weak).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


class ImageEmbedder:
    """CLIP image encoder -> image embeddings. Lazy, cached, frozen."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()

    @property
    def dim(self) -> int:
        self._ensure()
        return int(self._model.config.projection_dim)

    @torch.no_grad()
    def embed(self, images: Sequence) -> torch.Tensor:
        """``images``: list of PIL.Image / np.ndarray -> ``[len(images), dim]`` (CPU)."""
        self._ensure()
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.device)
        feats = self._model.get_image_features(**inputs)
        return feats.detach().cpu().float()


class TextEmbedder:
    """Pluggable text encoder for the L3 *output space*.

    ``backbone``:
      * ``"siglip"``  -> transformers SiglipModel (recommended; good text-text).
      * ``"sentence"`` -> sentence-transformers (strong text-text).
      * ``"clip"``    -> transformers CLIPModel text tower (weak text-text; for ablation).
    """

    def __init__(self, backbone: str = "siglip", model_name: str = None, device: str = None) -> None:
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name or {
            "siglip": "google/siglip-base-patch16-224",
            "sentence": "sentence-transformers/all-mpnet-base-v2",
            "clip": "openai/clip-vit-large-patch14",
        }[backbone]
        self._model = None
        self._processor = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if self.backbone == "sentence":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        elif self.backbone == "siglip":
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        elif self.backbone == "clip":
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        else:
            raise ValueError(f"Unknown text backbone: {self.backbone!r}")

    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """``texts`` -> ``[len(texts), dim]`` (CPU float)."""
        self._ensure()
        texts = list(texts)
        if self.backbone == "sentence":
            arr = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            return torch.from_numpy(np.asarray(arr)).float()
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = self._model.get_text_features(**inputs)
        return feats.detach().cpu().float()


def embed_action_statements(
    text_embedder: TextEmbedder,
    statements: Dict[str, List[str]],
) -> Tuple[List[str], torch.Tensor]:
    """Embed each action's phrasings and mean-pool them into one vector per action.

    :return: ``(actions, embeddings)`` where ``embeddings`` is ``[len(actions), dim]``
        aligned with ``actions``. Pass ``embeddings`` to ``L3FusionModel.score_actions``.
    """
    actions = list(statements.keys())
    vectors = []
    for action in actions:
        phr = statements[action]
        emb = text_embedder.embed(phr)               # [P, dim]
        emb = torch.nn.functional.normalize(emb, dim=-1).mean(dim=0)
        vectors.append(emb)
    return actions, torch.stack(vectors, dim=0)
