from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

from .schema import QueryRecord, RegionBox


_DEFAULT_QUERY_IDS: Dict[str, List[str]] = {
    "right_turn": [
        "clg_left_rear_vehicle",
        "clg_right_rear_vehicle",
        "clg_right_front_vehicle",
        "clg_left_front_vehicle",
        "clg_front_vehicle",
        "clg_rear_vehicle",
    ],
    "left_turn": [
        "clg_left_rear_vehicle",
        "clg_right_rear_vehicle",
        "clg_right_front_vehicle",
        "clg_left_front_vehicle",
        "clg_front_vehicle",
        "clg_rear_vehicle",
    ],
    "lane_change": [
        "target_lane_rear_vehicle",
        "target_lane_front_vehicle",
        "current_lane_front_vehicle",
        "target_lane_gap",
    ],
    "car_following": [
        "forward_lane_vehicle",
        "lead_vehicle_distance",
        "left_adjacent_vehicle",
        "right_adjacent_vehicle",
    ],
}


def get_default_query_ids(scene_type: str) -> List[str]:
    return list(_DEFAULT_QUERY_IDS.get(scene_type, _DEFAULT_QUERY_IDS["right_turn"]))


def make_query_records(scene_type: str, query_ids: Sequence[str] | None = None) -> List[QueryRecord]:
    ordered = list(query_ids or get_default_query_ids(scene_type))
    return [
        QueryRecord(
            query_id=query_id,
            query_embedding_input=_encode_query_embedding(scene_type, query_id),
            required_region=make_required_region(scene_type, query_id),
            metadata={"scene_type": scene_type},
        )
        for query_id in ordered
    ]


def make_required_region(scene_type: str, query_id: str) -> RegionBox:
    qid = query_id.lower()
    if scene_type == "lane_change":
        if "target_lane_rear" in qid:
            return RegionBox(center=(-8.0, -3.5), size=(10.0, 4.0), yaw=0.0)
        if "target_lane_front" in qid:
            return RegionBox(center=(10.0, -3.5), size=(12.0, 4.0), yaw=0.0)
        if "gap" in qid:
            return RegionBox(center=(2.0, -3.5), size=(14.0, 4.0), yaw=0.0)
        return RegionBox(center=(12.0, 0.0), size=(16.0, 4.5), yaw=0.0)
    if scene_type == "car_following":
        if "left" in qid:
            return RegionBox(center=(2.0, -3.5), size=(12.0, 4.0), yaw=0.0)
        if "right" in qid:
            return RegionBox(center=(2.0, 3.5), size=(12.0, 4.0), yaw=0.0)
        return RegionBox(center=(14.0, 0.0), size=(18.0, 5.0), yaw=0.0)
    if "left_rear" in qid:
        return RegionBox(center=(-8.0, -3.0), size=(10.0, 4.0), yaw=0.0)
    if "right_rear" in qid:
        return RegionBox(center=(-8.0, 3.0), size=(10.0, 4.0), yaw=0.0)
    if "right_front" in qid:
        return RegionBox(center=(10.0, 3.0), size=(12.0, 4.0), yaw=0.0)
    if "left_front" in qid:
        return RegionBox(center=(10.0, -3.0), size=(12.0, 4.0), yaw=0.0)
    if "rear" in qid:
        return RegionBox(center=(-8.0, 0.0), size=(10.0, 4.5), yaw=0.0)
    if "front" in qid or "forward" in qid:
        return RegionBox(center=(14.0, 0.0), size=(18.0, 5.0), yaw=0.0)
    return RegionBox(center=(8.0, 0.0), size=(12.0, 4.0), yaw=0.0)


def _encode_query_embedding(scene_type: str, query_id: str) -> List[float]:
    qid = query_id.lower()
    is_left = 1.0 if "left" in qid else 0.0
    is_right = 1.0 if "right" in qid else 0.0
    is_front = 1.0 if "front" in qid or "forward" in qid else 0.0
    is_rear = 1.0 if "rear" in qid else 0.0
    is_gap = 1.0 if "gap" in qid else 0.0
    is_turn = 1.0 if scene_type in ("right_turn", "left_turn") else 0.0
    is_lane_change = 1.0 if scene_type == "lane_change" else 0.0
    is_following = 1.0 if scene_type == "car_following" else 0.0
    region = make_required_region(scene_type, query_id)
    return [
        is_left,
        is_right,
        is_front,
        is_rear,
        is_gap,
        is_turn,
        is_lane_change,
        is_following,
        float(region.center[0]),
        float(region.center[1]),
        float(region.size[0]),
        float(region.size[1]),
        math.cos(float(region.yaw)),
        math.sin(float(region.yaw)),
    ]
