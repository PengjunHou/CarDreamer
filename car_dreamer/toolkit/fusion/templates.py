"""Templated language for Route-L3.

Two pieces of language live here:

1. ``build_scene_description`` -- turns a *privileged, full-visibility* structured
   scene layout (from CARLA ground truth) into a templated ego-frame description.
   Its text embedding is the L3 reconstruction target ``z_target``. Crucially, it
   uses the same vocabulary as the action statements below, so text-text cosine is
   high for matching cases.

2. ``ActionFactor`` -- the *schema* for a scene-grounded accelerate / decelerate /
   maintain "safety factor". Factors are NOT hardcoded here: they are generated per
   scene by a VLM/LLM (built in the downstream inference / semantic-confidence
   stage), because scene content varies (some scenes have no intersection or traffic
   light). At inference, each factor's positive/negative is embedded and scored
   against ``z_fused``.

The scene description (piece 1) is template-from-ground-truth on purpose -- the
reconstruction target must be deterministic. Its expressiveness is the ceiling of
what the fused representation can encode, so keep it covering exactly the facts the
action decision cares about (per-region occupancy + distance, and -- only when
present -- traffic light / pedestrians). Generated factors should reuse this same
vocabulary so their cosine against ``z_fused`` stays calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# Ego-frame regions, in a fixed reporting order.
REGION_ORDER: Sequence[str] = (
    "front",
    "left-front",
    "right-front",
    "rear",
    "left-rear",
    "right-rear",
)


def distance_phrase(distance_m: float) -> str:
    """Coarse distance bucket used in both the target and action language."""
    d = float(distance_m)
    if d < 6.0:
        return "very close"
    if d < 12.0:
        return "close"
    if d < 25.0:
        return "nearby"
    return "far"


def build_scene_description(
    objects: Sequence[Dict[str, Any]],
    traffic_light: Optional[str] = None,
    extra_facts: Optional[Sequence[str]] = None,
) -> str:
    """Build the privileged, full-visibility templated scene description.

    :param objects: each item is ``{"region": <one of REGION_ORDER>,
        "distance_m": float, "kind": "vehicle"|"pedestrian"|...}``. Build this from
        CARLA ground truth: project every relevant actor into the ego frame, bucket
        it into a region, and record its distance.
    :param traffic_light: e.g. ``"red"`` / ``"green"`` / ``"yellow"`` / ``None``.
    :param extra_facts: optional scene-level sentences (occlusion, congestion, ...).
    :return: a single-line templated description string.
    """
    by_region: Dict[str, List[Dict[str, Any]]] = {r: [] for r in REGION_ORDER}
    for obj in objects:
        region = str(obj.get("region", "front"))
        if region in by_region:
            by_region[region].append(obj)

    lines: List[str] = []
    for region in REGION_ORDER:
        items = by_region[region]
        label = region.capitalize()
        if not items:
            lines.append(f"{label}: no vehicle.")
            continue
        nearest = min(items, key=lambda o: float(o.get("distance_m", 1e9)))
        kind = str(nearest.get("kind", "vehicle"))
        dist = float(nearest.get("distance_m", 0.0))
        lines.append(f"{label}: a {kind} {distance_phrase(dist)} ({dist:.0f} m).")

    if traffic_light:
        lines.append(f"Traffic light: {traffic_light}.")
    for fact in (extra_facts or []):
        lines.append(str(fact))
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Action factors are NOT hardcoded here: scene content varies (some scenes have no
# intersection or traffic light), so the factor set must be *generated from the
# current scene* by a VLM/LLM. This is only the structural schema. The actual
# generator is built in the inference / semantic-confidence stage and emits a list
# of ``ActionFactor`` per scene.
#
# Why dynamic generation is safe in L3: ``z_fused`` is trained toward the templated
# GT description, so it *grounds* the factors -- an irrelevant / over-proposed
# factor simply gets low cosine and washes out (VLM proposes = recall; z_fused
# arbitrates = precision). The one real constraint is that generated factor language
# should share vocabulary with ``build_scene_description`` above, so the cosine
# against ``z_fused`` stays calibrated.
# --------------------------------------------------------------------------- #

ACTIONS: Sequence[str] = ("accelerate", "decelerate", "maintain")


@dataclass
class ActionFactor:
    """One scene-grounded driving factor, matching the PDF §7 structured-JSON format.

    Produced per scene by the (downstream) VLM/LLM factor generator, never hardcoded.
    Field names mirror the PDF: ``evidence_language`` (l_pos), ``contrast_language``
    (l_neg), and ``expected_sensor_check`` -- the optional physical / CARLA-state
    grounding channel from PDF §6 (e.g. ``"front_vehicle_distance < 12m"``).
    ``sign`` / ``weight`` come from the PDF §11 aggregation pseudocode.

    At inference, ``evidence_language`` / ``contrast_language`` are embedded and
    scored against ``z_fused``; the per-factor confidence (optionally combined with
    the ``expected_sensor_check`` channel) is aggregated per action -- this is where
    the deferred semantic-confidence belief/ambiguity formula plugs in.
    """

    factor: str                      # short name, e.g. "vehicle close ahead"
    action: str                      # one of ACTIONS (PDF's "<action>_support" grouping)
    evidence_language: str           # PDF: l_pos -- consistent with `action` being right
    contrast_language: str           # PDF: l_neg -- contrastive statement opposing it
    expected_sensor_check: str = ""  # PDF §6: optional physical check, e.g. "front_vehicle_distance < 12m"
    sign: float = 1.0                # PDF §11: +1 supports `action`, -1 opposes it
    weight: float = 1.0


def factors_from_support_json(payload: Dict[str, Any]) -> List["ActionFactor"]:
    """Parse the PDF §7 grouped JSON into a flat list of ``ActionFactor``.

    Expects ``{"decelerate_support": [ {...}, ... ], "accelerate_support": [...], ...}``
    -- exactly what the VLM/LLM factor generator is prompted to emit. Missing/unknown
    fields are tolerated so a partial generation still yields usable factors.
    """
    factors: List[ActionFactor] = []
    for key, items in (payload or {}).items():
        if not key.endswith("_support") or not isinstance(items, list):
            continue
        action = key[: -len("_support")]
        for item in items:
            if not isinstance(item, dict):
                continue
            factors.append(
                ActionFactor(
                    factor=str(item.get("factor", "")),
                    action=action,
                    evidence_language=str(item.get("evidence_language", "")),
                    contrast_language=str(item.get("contrast_language", "")),
                    expected_sensor_check=str(item.get("expected_sensor_check", "")),
                    sign=float(item.get("sign", 1.0)),
                    weight=float(item.get("weight", 1.0)),
                )
            )
    return factors
