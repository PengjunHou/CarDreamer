"""Route-L3 cooperative scene-embedding fusion.

Self-supervised fusion of ego image embedding + neighbor caption embeddings (each
tagged with its ego-relative pose) into a single scene embedding, trained to match
the embedding of a privileged templated scene description, and used at inference to
score accelerate / decelerate / maintain action language.

See ``l3_model.py`` for the model and ``templates.py`` for the target/action text.
"""

from __future__ import annotations

from .l3_model import L3FusionConfig, L3FusionModel
from .pose_encoding import RelativePoseEncoder
from .set_fusion import PMA, SetTransformerFusion
from .templates import (
    ACTIONS,
    REGION_ORDER,
    ActionFactor,
    build_scene_description,
    factors_from_support_json,
)

__all__ = [
    "L3FusionConfig",
    "L3FusionModel",
    "RelativePoseEncoder",
    "SetTransformerFusion",
    "PMA",
    "ACTIONS",
    "ActionFactor",
    "factors_from_support_json",
    "REGION_ORDER",
    "build_scene_description",
]
