"""Utilities for converting ego-centered spatial questions into member-view prompts.

This version follows the user's simulation coordinate convention exactly:

Global coordinates
------------------
- +X points to the right
- +Y points downward
- yaw / heading is measured from global +X
- clockwise angles are positive
- counterclockwise angles are negative

Vehicle-relative semantic directions
------------------------------------
"Front / rear / left / right" are defined by vehicle heading, not by the
fixed global axes.

For convenience inside this module, each queried ego-region is represented in a
vehicle-local semantic frame with:
- local +x = front
- local -x = rear
- local +y = left
- local -y = right

Those local semantic offsets are then converted into global coordinates using
the simulation's angle convention above.

Core idea
---------
Given:
- ego vehicle pose
- observer vehicle or observer sensor pose
- a question id such as ``clg_left_rear_vehicle``

we compute the center of the queried region in the ego semantic local frame,
project that region center into the observer semantic local frame, and then map
the resulting local angle to a discrete directional label such as:
    front, front-left, left, rear-left, rear, rear-right, right, front-right

This lets the caller rewrite a question like
    "Is there a vehicle in the left-rear region of the ego vehicle?"
into a member-viewed prompt such as
    "The question is still about the ego vehicle. From this camera's viewpoint,
     the queried region around the ego vehicle lies approximately in the
     front-left direction. Is there a vehicle in that queried region around the
     ego vehicle?"

The original anchor remains the ego vehicle. Only the directional wording is
converted into the observer's local view.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Basic math helpers
# -----------------------------------------------------------------------------


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi



def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0



def deg2rad(angle_deg: float) -> float:
    return math.radians(float(angle_deg))



def rad2deg(angle_rad: float) -> float:
    return math.degrees(float(angle_rad))



def semantic_local_to_global(dx_local: float, dy_local: float, yaw_rad: float) -> Tuple[float, float]:
    """Convert semantic local offset to global offset.

    Local semantic frame:
    - +x = front
    - +y = left

    Global simulation frame:
    - +X = right
    - +Y = down
    - yaw measured from +X, clockwise positive

    If heading is yaw_rad, then:
    - front unit vector = (cos(yaw), sin(yaw))
    - left unit vector  = (sin(yaw), -cos(yaw))
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    dx_global = c * dx_local + s * dy_local
    dy_global = s * dx_local - c * dy_local
    return dx_global, dy_global



def global_to_semantic_local(
    global_xy: Tuple[float, float],
    origin_xy: Tuple[float, float],
    local_yaw_rad: float,
) -> Tuple[float, float]:
    """Convert a global point into the observer semantic local frame.

    Returned local coordinates follow:
    - +x = observer front
    - +y = observer left
    """
    dx = float(global_xy[0]) - float(origin_xy[0])
    dy = float(global_xy[1]) - float(origin_xy[1])

    c = math.cos(local_yaw_rad)
    s = math.sin(local_yaw_rad)

    # Project onto the observer's front and left unit vectors.
    local_x = dx * c + dy * s
    local_y = dx * s - dy * c
    return local_x, local_y



def semantic_local_angle_deg(dx_local: float, dy_local: float) -> float:
    """Angle in semantic local frame, measured from local +x (front).

    Convention used here:
    - 0 deg   = front
    - +90 deg = right
    - -90 deg = left
    - clockwise positive
    - counterclockwise negative

    Because the semantic local frame uses +y=left, we negate atan2 so that the
    final angle matches the simulation's clockwise-positive convention.
    """
    return wrap_angle_deg(-rad2deg(math.atan2(dy_local, dx_local)))


# -----------------------------------------------------------------------------
# Pose parsing
# -----------------------------------------------------------------------------


def _maybe_get(mapping: Mapping[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            try:
                return float(mapping[k])
            except Exception:
                continue
    return default



def parse_pose_xy_yaw_rad(pose: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Parse a flexible pose mapping into (x, y, yaw_rad).

    Accepted keys include common variants such as:
    - x / y / yaw
    - location_x / location_y / yaw_deg
    - sensor_yaw_rad
    - yaw_rad

    Yaw is interpreted as degrees unless a *_rad key is present.

    Important: yaw follows the simulation convention:
    - measured from global +X
    - clockwise positive
    """
    x = _maybe_get(pose, "x", "location_x", "pos_x", default=0.0)
    y = _maybe_get(pose, "y", "location_y", "pos_y", default=0.0)

    yaw_rad = _maybe_get(pose, "yaw_rad", "sensor_yaw_rad", default=None)
    if yaw_rad is None:
        yaw_deg = _maybe_get(pose, "yaw", "yaw_deg", "sensor_yaw", default=0.0)
        yaw_rad = deg2rad(yaw_deg)

    return float(x), float(y), float(yaw_rad)


# -----------------------------------------------------------------------------
# Query definitions
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryRegionSpec:
    question_id: str
    offset_x_m: float
    offset_y_m: float
    canonical_label: str
    query_template: str


DEFAULT_QUERY_SPECS: Dict[str, QueryRegionSpec] = {
    "clg_left_front_vehicle": QueryRegionSpec(
        question_id="clg_left_front_vehicle",
        offset_x_m=8.0,
        offset_y_m=3.5,
        canonical_label="left-front",
        query_template="Is there a vehicle in the left-front region of the vehicle?",
    ),
    "clg_right_front_vehicle": QueryRegionSpec(
        question_id="clg_right_front_vehicle",
        offset_x_m=8.0,
        offset_y_m=-3.5,
        canonical_label="right-front",
        query_template="Is there a vehicle in the right-front region of the vehicle?",
    ),
    "clg_left_rear_vehicle": QueryRegionSpec(
        question_id="clg_left_rear_vehicle",
        offset_x_m=-8.0,
        offset_y_m=3.5,
        canonical_label="left-rear",
        query_template="Is there a vehicle in the left-rear region of the vehicle?",
    ),
    "clg_right_rear_vehicle": QueryRegionSpec(
        question_id="clg_right_rear_vehicle",
        offset_x_m=-8.0,
        offset_y_m=-3.5,
        canonical_label="right-rear",
        query_template="Is there a vehicle in the right-rear region of the vehicle?",
    ),
    "clg_front_vehicle": QueryRegionSpec(
        question_id="clg_front_vehicle",
        offset_x_m=8.0,
        offset_y_m=0.0,
        canonical_label="front",
        query_template="Is there a vehicle in the front region of the vehicle?",
    ),
    "clg_rear_vehicle": QueryRegionSpec(
        question_id="clg_rear_vehicle",
        offset_x_m=-8.0,
        offset_y_m=0.0,
        canonical_label="rear",
        query_template="Is there a vehicle in the rear region of the vehicle?",
    ),
    "clg_left_vehicle": QueryRegionSpec(
        question_id="clg_left_vehicle",
        offset_x_m=0.0,
        offset_y_m=3.5,
        canonical_label="left",
        query_template="Is there a vehicle in the left region of the vehicle?",
    ),
    "clg_right_vehicle": QueryRegionSpec(
        question_id="clg_right_vehicle",
        offset_x_m=0.0,
        offset_y_m=-3.5,
        canonical_label="right",
        query_template="Is there a vehicle in the right region of the vehicle?",
    ),
}


# Local-angle bins under the simulation-style clockwise-positive convention.
# 0 = front, +90 = right, -90 = left.
DIRECTION_BINS: Sequence[Tuple[float, float, str]] = (
    (-22.5, 22.5, "front"),
    (22.5, 67.5, "front-right"),
    (67.5, 112.5, "right"),
    (112.5, 157.5, "rear-right"),
    (157.5, 180.0, "rear"),
    (-180.0, -157.5, "rear"),
    (-157.5, -112.5, "rear-left"),
    (-112.5, -67.5, "left"),
    (-67.5, -22.5, "front-left"),
)


# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class ConvertedQuery:
    question_id: str
    original_label: str
    original_query: str
    converted_direction: str
    target_global_xy: Tuple[float, float]
    target_local_xy: Tuple[float, float]
    target_local_angle_deg: float
    ego_global_xy: Tuple[float, float]
    observer_global_xy: Tuple[float, float]
    ego_heading_deg: float
    observer_heading_deg: float
    ego_to_target_distance_m: float
    observer_to_target_distance_m: float
    observer_to_ego_distance_m: float
    target_offset_from_ego_local_xy: Tuple[float, float]
    target_offset_from_ego_view_label: str
    query: str
    positive: str
    negative: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Geometry and mapping
# -----------------------------------------------------------------------------


def angle_deg_to_direction_label(angle_deg: float) -> str:
    angle_deg = wrap_angle_deg(angle_deg)
    for low, high, label in DIRECTION_BINS:
        if low <= angle_deg < high:
            return label
    return "front"



def ego_local_offset_for_question(
    question_id: str,
    query_specs: Optional[Mapping[str, QueryRegionSpec]] = None,
) -> Tuple[float, float]:
    specs = query_specs or DEFAULT_QUERY_SPECS
    if question_id not in specs:
        supported = ", ".join(sorted(specs.keys()))
        raise KeyError(f"Unsupported question_id={question_id!r}. Supported ids: {supported}")
    spec = specs[question_id]
    return spec.offset_x_m, spec.offset_y_m



def compute_query_target_global_xy(
    ego_pose: Mapping[str, Any],
    question_id: str,
    query_specs: Optional[Mapping[str, QueryRegionSpec]] = None,
) -> Tuple[float, float]:
    ego_x, ego_y, ego_yaw_rad = parse_pose_xy_yaw_rad(ego_pose)
    off_x, off_y = ego_local_offset_for_question(question_id, query_specs=query_specs)
    dx_g, dy_g = semantic_local_to_global(off_x, off_y, ego_yaw_rad)
    return ego_x + dx_g, ego_y + dy_g



def compute_query_direction_from_observer(
    ego_pose: Mapping[str, Any],
    observer_pose: Mapping[str, Any],
    question_id: str,
    query_specs: Optional[Mapping[str, QueryRegionSpec]] = None,
) -> ConvertedQuery:
    specs = query_specs or DEFAULT_QUERY_SPECS
    if question_id not in specs:
        supported = ", ".join(sorted(specs.keys()))
        raise KeyError(f"Unsupported question_id={question_id!r}. Supported ids: {supported}")

    spec = specs[question_id]
    ego_x, ego_y, ego_yaw_rad = parse_pose_xy_yaw_rad(ego_pose)
    obs_x, obs_y, obs_yaw_rad = parse_pose_xy_yaw_rad(observer_pose)

    target_global_xy = compute_query_target_global_xy(ego_pose, question_id, query_specs=specs)
    target_local_xy = global_to_semantic_local(target_global_xy, (obs_x, obs_y), obs_yaw_rad)
    local_angle_deg = semantic_local_angle_deg(target_local_xy[0], target_local_xy[1])
    converted_direction = angle_deg_to_direction_label(local_angle_deg)

    off_x, off_y = spec.offset_x_m, spec.offset_y_m
    offset_angle_deg = semantic_local_angle_deg(off_x, off_y)
    offset_label = angle_deg_to_direction_label(offset_angle_deg)

    ego_to_target_distance_m = math.hypot(off_x, off_y)
    observer_to_target_distance_m = math.hypot(target_local_xy[0], target_local_xy[1])
    observer_to_ego_distance_m = math.hypot(obs_x - ego_x, obs_y - ego_y)

    # prompt = build_member_view_prompt(
    #     original_query=spec.query_template,
    #     converted_direction=converted_direction,
    #     canonical_label=spec.canonical_label,
    # )
    converted_ques = build_member_view_short_query(converted_direction)

    return ConvertedQuery(
        question_id=question_id,
        original_label=spec.canonical_label,
        original_query=spec.query_template,
        converted_direction=converted_direction,
        target_global_xy=(float(target_global_xy[0]), float(target_global_xy[1])),
        target_local_xy=(float(target_local_xy[0]), float(target_local_xy[1])),
        target_local_angle_deg=float(local_angle_deg),
        ego_global_xy=(float(ego_x), float(ego_y)),
        observer_global_xy=(float(obs_x), float(obs_y)),
        ego_heading_deg=float(wrap_angle_deg(rad2deg(ego_yaw_rad))),
        observer_heading_deg=float(wrap_angle_deg(rad2deg(obs_yaw_rad))),
        ego_to_target_distance_m=float(ego_to_target_distance_m),
        observer_to_target_distance_m=float(observer_to_target_distance_m),
        observer_to_ego_distance_m=float(observer_to_ego_distance_m),
        target_offset_from_ego_local_xy=(float(off_x), float(off_y)),
        target_offset_from_ego_view_label=offset_label,
        **converted_ques
    )


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------


DIRECTION_PHRASE = {
    "front": "front",
    "front-left": "front-left",
    "left": "left",
    "rear-left": "rear-left",
    "rear": "rear",
    "rear-right": "rear-right",
    "right": "right",
    "front-right": "front-right",
}



def build_member_view_prompt(
    original_query: str,
    converted_direction: str,
    canonical_label: str,
) -> str:
    dir_phrase = DIRECTION_PHRASE.get(converted_direction, converted_direction)
    return (
        f"Is there a vehicle in the {dir_phrase} region of the vehicle?"
    )
    
    # return (
    #     "This image is captured by a neighboring vehicle, not by the ego vehicle. "
    #     "The question is still about the ego vehicle. "
    #     f"From this camera's viewpoint, the queried region around the ego vehicle lies approximately in the {dir_phrase} direction. "
    #     f"The original ego-centered region is the {canonical_label} region of the ego vehicle. "
    #     "Answer only about that queried region around the ego vehicle. "
    #     "If the ego vehicle or the queried region cannot be determined from this image, answer unknown. "
    #     f"Question: {original_query}"
    # )



def build_member_view_short_query(converted_direction: str) -> str:
    dir_phrase = DIRECTION_PHRASE.get(converted_direction, converted_direction)
    return {
        "query": f"Is there a vehicle in the {dir_phrase} region of the vehicle?",
        "negative": f"There is no vehicle in the {dir_phrase} region of the vehicle.",
        "positive": f"There is a vehicle in the {dir_phrase} region of the vehicle."
    }



def rewrite_question_cfg_for_member(
    question_cfg: Mapping[str, Any],
    ego_pose: Mapping[str, Any],
    observer_pose: Mapping[str, Any],
    query_specs: Optional[Mapping[str, QueryRegionSpec]] = None,
    prompt_field: str = "member_view_query",
    short_field: str = "member_view_short_query",
    metadata_field: str = "member_view_metadata",
) -> Dict[str, Any]:
    """Return a copied question_cfg with converted member-view prompt fields added.

    This does not destroy the original query/positive/negative fields, so the caller
    can choose to use the converted prompt only for shared/member sensors.
    """
    result = dict(question_cfg)
    converted = compute_query_direction_from_observer(
        ego_pose=ego_pose,
        observer_pose=observer_pose,
        question_id=str(question_cfg["id"]),
        query_specs=query_specs,
    )
    result[prompt_field] = converted.prompt
    result[short_field] = converted.short_prompt
    result[metadata_field] = converted.to_dict()
    return result


# -----------------------------------------------------------------------------
# Optional visibility / gating helpers
# -----------------------------------------------------------------------------


@dataclass
class QueryVisibilityHint:
    should_answer: bool
    reason: str
    observer_to_target_distance_m: float
    target_local_angle_deg: float
    converted_direction: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def build_visibility_hint(
    converted: ConvertedQuery,
    *,
    max_distance_m: float = 80.0,
    max_abs_angle_deg_for_hint: float = 140.0,
) -> QueryVisibilityHint:
    """Cheap geometric hint for whether this observer is plausibly useful.

    This is intentionally lightweight and does not replace full image-based checks.
    Angle is interpreted under the simulation convention:
    - 0 = front
    - +90 = right
    - -90 = left
    """
    dist = converted.observer_to_target_distance_m
    ang = abs(converted.target_local_angle_deg)

    if dist > max_distance_m:
        return QueryVisibilityHint(
            should_answer=False,
            reason=f"target too far ({dist:.1f}m > {max_distance_m:.1f}m)",
            observer_to_target_distance_m=dist,
            target_local_angle_deg=converted.target_local_angle_deg,
            converted_direction=converted.converted_direction,
        )
    if ang > max_abs_angle_deg_for_hint:
        return QueryVisibilityHint(
            should_answer=False,
            reason=f"target too far outside nominal view angle (|{converted.target_local_angle_deg:.1f}| > {max_abs_angle_deg_for_hint:.1f} deg)",
            observer_to_target_distance_m=dist,
            target_local_angle_deg=converted.target_local_angle_deg,
            converted_direction=converted.converted_direction,
        )

    return QueryVisibilityHint(
        should_answer=True,
        reason="geometrically plausible",
        observer_to_target_distance_m=dist,
        target_local_angle_deg=converted.target_local_angle_deg,
        converted_direction=converted.converted_direction,
    )


# -----------------------------------------------------------------------------
# Convenience wrappers for common CARLA-style dicts
# -----------------------------------------------------------------------------



def observer_pose_from_sensor_instance(sensor_instance: Mapping[str, Any]) -> Dict[str, float]:
    """Extract a minimal observer pose dict from a CARLA-like sensor instance mapping.

    Supported input keys include:
    - sensor_pose: {x, y, yaw / yaw_rad / sensor_yaw_rad}
    - pose: {x, y, yaw}
    - top-level x/y/yaw-like keys
    """
    nested_pose = sensor_instance.get("sensor_pose")
    if isinstance(nested_pose, Mapping):
        x, y, yaw_rad = parse_pose_xy_yaw_rad(nested_pose)
        return {"x": x, "y": y, "yaw_rad": yaw_rad}

    nested_pose = sensor_instance.get("pose")
    if isinstance(nested_pose, Mapping):
        x, y, yaw_rad = parse_pose_xy_yaw_rad(nested_pose)
        return {"x": x, "y": y, "yaw_rad": yaw_rad}

    x, y, yaw_rad = parse_pose_xy_yaw_rad(sensor_instance)
    return {"x": x, "y": y, "yaw_rad": yaw_rad}


# -----------------------------------------------------------------------------
# Demo helpers
# -----------------------------------------------------------------------------


def json_pretty(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ego_pose = {"x": -6.8896684646606445, "y": -120.94688415527344, "yaw": 89.64091491699219}
    # ego_pose = {"x": 15.404886245727539, "y": -140.19998168945312, "yaw": 180}
    member_sensor_pose = {"x": 15.404886245727539, "y": -140.19998168945312, "yaw": 180}
    # member_sensor_pose = {"x": 7.400001049041748, "y": -124.199951171875, "yaw": -90}

    converted = compute_query_direction_from_observer(
        ego_pose=ego_pose,
        observer_pose=member_sensor_pose,
        question_id="clg_left_rear_vehicle",
    )
    print(json_pretty(converted.to_dict()))



