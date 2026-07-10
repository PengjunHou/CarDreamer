"""Policy-conditioned heterogeneous graph construction for the WAM pipeline.

This module realizes WAM Design §4-§7: given a Base-Station collaboration policy
``π_t = (S_t, B_t, f_t, d_t)`` it assembles the policy-conditioned cooperative graph
``G_t^{e,π}`` as a :class:`torch_geometric.data.HeteroData` with three node types and
three edge relations.

Node types (§5):
    * ``vehicle``      -- ego + policy-selected member vehicles (§5.1)
    * ``object``       -- vehicles / pedestrians / ... observed by ego or a member (§5.2)
    * ``observation``  -- modality-specific observation node (§5.3); the modality
                          (``objlist`` / ``bev``) is carried as a node feature + type id
                          rather than a separate node type, so the three edge relations
                          below match §9 exactly.

Edge relations (§6 + Edge Representation Update):
    * ``(vehicle, veh_obs, observation)``  -- vehicle v provides modality-r observation (§6.1); structural, no attr
    * ``(observation, obs_obj, object)``   -- observation detects object o (§6.2); ``edge_attr = [det_confidence]``
    * ``(vehicle, veh_veh, vehicle)``      -- collaborator m transmits to request vehicle q under the policy (§6.3);
                                             ``edge_attr = [latency_s]`` (policy-conditioned). Only present when m
                                             cooperates (the "V2V perception graph"); absent in the local graph.

The builder is intentionally CARLA-agnostic: it consumes plain dataclasses
(:class:`VehicleNodeInput`, :class:`ObservationNodeInput`) and :class:`ObjectState`
(from :mod:`car_dreamer.toolkit.wam.runtime`), so it is unit-testable without a simulator.
The learnable modality feature ``z^r`` and the initial node embeddings ``h^0`` are produced
by :mod:`car_dreamer.toolkit.wam.graph_model`; this module only lays out the node *state*
vectors and the edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .runtime import ObjectState, WAMPolicy

Point2D = Tuple[float, float]


# =====================================================================
# Canonical hetero-graph schema (must match the encoder metadata)
# =====================================================================

VEHICLE = "vehicle"
OBJECT = "object"
OBSERVATION = "observation"
NODE_TYPES: Tuple[str, ...] = (VEHICLE, OBJECT, OBSERVATION)

VEH_OBS = (VEHICLE, "veh_obs", OBSERVATION)
OBS_OBJ = (OBSERVATION, "obs_obj", OBJECT)
VEH_VEH = (VEHICLE, "veh_veh", VEHICLE)
EDGE_TYPES: Tuple[Tuple[str, str, str], ...] = (VEH_OBS, OBS_OBJ, VEH_VEH)

# Per-edge attribute widths (Edge Representation Update). ``veh_obs`` stays structural.
#   obs_obj: [det_confidence]   -- detection confidence s_det ∈ [0, 1] of the (obs, object) pair
#   veh_veh: [latency_s]        -- policy-conditioned transmission latency L^π in seconds
OBS_OBJ_EDGE_DIM = 1
VEH_VEH_EDGE_DIM = 1
EDGE_ATTR_DIMS: Dict[Tuple[str, str, str], int] = {OBS_OBJ: OBS_OBJ_EDGE_DIM, VEH_VEH: VEH_VEH_EDGE_DIM}

# torch_geometric metadata tuple ``(node_types, edge_types)``.
WAM_METADATA = (list(NODE_TYPES), list(EDGE_TYPES))

MODALITIES: Tuple[str, ...] = ("objlist", "bev")
MODALITY_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(MODALITIES)}

OBJECT_CLASSES: Tuple[str, ...] = ("vehicle", "pedestrian", "bicycle", "other")
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(OBJECT_CLASSES)}

# Numeric state-vector layouts (the learnable z^r / class / type / agent / time
# embeddings are added on top inside graph_model.py and are NOT counted here).
#   object: [x, y, z, vx, vy, cos_yaw, sin_yaw, l, w, h, dt]
# Detection confidence s_det is NOT an object property (Edge Representation Update): it is a
# per-(observation, object) property carried on the obs_obj edge attr, so it is no longer here.
OBJECT_STATE_DIM = 11
#   observation scalars: [payload_kb, latency_s, freshness, quality, sample_age_s]
OBS_SCALAR_DIM = 5
# bytes used to estimate an object-list payload size per object (state floats).
OBJLIST_BYTES_PER_OBJECT = OBJECT_STATE_DIM * 4


def vehicle_state_dim(route_waypoints: int) -> int:
    """``[x, y, z, vx, vy, cos_yaw, sin_yaw, q_comm, q_comp] + route(2 * K)``."""
    return 9 + 2 * int(route_waypoints)


# =====================================================================
# Inputs
# =====================================================================


@dataclass
class VehicleNodeInput:
    """State of one vehicle node (§5.1). ``yaw`` is in degrees (CARLA convention)."""

    actor_id: int
    is_ego: bool
    agent_slot: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    yaw: float
    q_comm: float = 1.0
    q_comp: float = 1.0
    route_xy: Tuple[Point2D, ...] = ()


@dataclass
class ObservationNodeInput:
    """A modality-specific observation a vehicle produces / shares this step (§5.3)."""

    vehicle_id: int
    modality: str
    observed_object_ids: Tuple[int, ...] = ()
    payload_bytes: float = 0.0
    latency_s: float = 0.0
    freshness: float = 1.0
    quality: float = 1.0
    sample_age_s: float = 0.0
    # Per-detected-object confidence s_det ∈ [0, 1] (obs_obj edge attr). Missing ids default to 1.0
    # (ground-truth perception; a real detector fills this later).
    det_confidence_by_object: Dict[int, float] = field(default_factory=dict)
    # Visibility-aware BEV semantic raster ``B^sem [C,H,W]`` for ``bev`` modality (§5.3); ``None`` otherwise.
    bev_raster: Optional["np.ndarray"] = None


@dataclass
class GraphBuildSpec:
    route_waypoints: int = 6
    max_object_nodes: int = 32
    ego_frame: bool = True


# =====================================================================
# Helpers
# =====================================================================


def _normalize_modalities(modality_by_vehicle: Dict[int, object]) -> Dict[int, Set[str]]:
    """Coerce ``policy.modality_by_vehicle`` (str or collection) to a set per vehicle."""
    out: Dict[int, Set[str]] = {}
    for vid, value in (modality_by_vehicle or {}).items():
        if isinstance(value, str):
            out[int(vid)] = {value}
        else:
            out[int(vid)] = {str(v) for v in value}
    return out


class _EgoFrame:
    """World -> ego-centric rotation/translation (mirrors graph_build._rel_pose_ego_frame)."""

    def __init__(self, ego: VehicleNodeInput, enabled: bool):
        self.enabled = bool(enabled)
        self.ex, self.ey = float(ego.x), float(ego.y)
        yaw = math.radians(float(ego.yaw))
        self.cos, self.sin = math.cos(-yaw), math.sin(-yaw)
        self.ego_yaw = yaw

    def xy(self, x: float, y: float) -> Point2D:
        if not self.enabled:
            return float(x), float(y)
        dx, dy = float(x) - self.ex, float(y) - self.ey
        return self.cos * dx - self.sin * dy, self.sin * dx + self.cos * dy

    def vec(self, vx: float, vy: float) -> Point2D:
        if not self.enabled:
            return float(vx), float(vy)
        return self.cos * vx - self.sin * vy, self.sin * vx + self.cos * vy

    def yaw_cos_sin(self, yaw_deg: float) -> Point2D:
        dyaw = math.radians(float(yaw_deg)) - (self.ego_yaw if self.enabled else 0.0)
        return math.cos(dyaw), math.sin(dyaw)


def _edge_index(src: Sequence[int], dst: Sequence[int]) -> torch.Tensor:
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


def _dist2_to_ego(ego: VehicleNodeInput, state: ObjectState) -> float:
    return (float(state.x) - float(ego.x)) ** 2 + (float(state.y) - float(ego.y)) ** 2


# =====================================================================
# Injection (early) fusion
# =====================================================================


@dataclass
class FusedObjects:
    """Ego-centric hard-fused object set for the injection graph.

    ``object_states`` are the deduped objects (ego-visible use ego's fresh state; collaborator-only
    objects are injected with their snapshot state, keeping the highest-confidence detection when
    several collaborators saw the same object). ``ego_visible_ids`` marks which are ego-visible (the
    rest are injected -> ``invisible``). ``det_confidence_by_object`` feeds the obs_obj edge attr.
    """

    object_states: List[ObjectState]
    ego_visible_ids: Set[int]
    det_confidence_by_object: Dict[int, float]


def detection_confidence(observer_xy: Point2D, obj: ObjectState) -> float:
    """Distance-based detection-confidence proxy ``1/(1+d)`` (nearer observer -> more reliable)."""
    d = math.hypot(float(obj.x) - float(observer_xy[0]), float(obj.y) - float(observer_xy[1]))
    return 1.0 / (1.0 + float(d))


def fuse_injected_objects(
    ego_visible: Sequence[ObjectState],
    collaborator_detections: Sequence[Tuple[ObjectState, float]],
) -> FusedObjects:
    """Fuse ego-visible objects with collaborator detections into one ego-centric object set.

    ``collaborator_detections`` is a list of ``(object_state_at_t_sense, confidence)``. Ego's fresh
    state always wins for objects ego can see; collaborator-only objects are injected, keeping the
    highest-confidence detection per object id.
    """
    fused: Dict[int, ObjectState] = {}
    conf: Dict[int, float] = {}
    ego_ids: Set[int] = set()
    for s in ego_visible:
        oid = int(s.actor_id)
        fused[oid] = s
        conf[oid] = 1.0
        ego_ids.add(oid)
    for s, c in collaborator_detections:
        oid = int(s.actor_id)
        if oid in ego_ids:
            continue  # ego's fresh state overrides stale collaborator snapshots
        if oid not in fused or float(c) > conf[oid]:
            fused[oid] = s
            conf[oid] = float(c)
    return FusedObjects(list(fused.values()), ego_ids, conf)


# =====================================================================
# Builder
# =====================================================================


def build_wam_hetero_graph(
    *,
    ego: VehicleNodeInput,
    collaborators: Sequence[VehicleNodeInput],
    objects: Sequence[ObjectState],
    observations: Sequence[ObservationNodeInput],
    policy: WAMPolicy,
    spec: GraphBuildSpec = GraphBuildSpec(),
    notable_ids: Optional[Set[int]] = None,
    latency_by_vehicle: Optional[Dict[int, float]] = None,
    object_visibility: Optional[Dict[int, bool]] = None,
) -> HeteroData:
    """Assemble the policy-conditioned heterogeneous graph ``G_t^{e,π}`` (§7).

    The ego vehicle node and its own local observations are always present.
    Only collaborators in ``policy.selected_vehicle_ids`` (and the modalities in
    ``policy.modality_by_vehicle``) are added, so an empty/no-coop policy yields a valid
    ego-only graph.

    Edge attributes (Edge Representation Update):
      * ``obs_obj.edge_attr = [det_confidence]`` -- from each observation's
        ``det_confidence_by_object`` (default 1.0).
      * ``veh_veh.edge_attr = [latency_s]`` -- policy-conditioned transmission latency in seconds,
        from ``latency_by_vehicle`` (default 0.0; ego→ego style self-latency is 0).
    """
    latency_by_vehicle = {int(k): float(v) for k, v in (latency_by_vehicle or {}).items()}
    notable_ids = {int(i) for i in (notable_ids or set())}
    # Injection (early-fusion) mode passes an explicit per-object ego-visibility map: an object is
    # ``visible`` iff ego saw it directly, ``invisible`` iff it was injected from a collaborator's
    # detection. When None, fall back to deriving visibility from ego's own observation node.
    visibility_override = None if object_visibility is None else {int(k): bool(v) for k, v in object_visibility.items()}
    selected = {int(v) for v in policy.selected_vehicle_ids}
    modality_by_vehicle = _normalize_modalities(policy.modality_by_vehicle)
    K = max(int(spec.route_waypoints), 0)
    frame = _EgoFrame(ego, spec.ego_frame)

    # ---- 1) vehicle nodes: ego + selected collaborators (§7) ----
    veh_inputs: List[VehicleNodeInput] = [ego]
    for c in collaborators:
        if int(c.actor_id) in selected and int(c.actor_id) != int(ego.actor_id):
            veh_inputs.append(c)
    veh_id_to_idx = {int(v.actor_id): i for i, v in enumerate(veh_inputs)}
    included_veh_ids = set(veh_id_to_idx)

    # ---- 2) observation nodes: ego's own + selected members' activated modalities ----
    obs_inputs: List[ObservationNodeInput] = []
    for obs in observations:
        vid = int(obs.vehicle_id)
        if vid not in included_veh_ids:
            continue
        if vid == int(ego.actor_id):
            obs_inputs.append(obs)
        elif obs.modality in modality_by_vehicle.get(vid, set()):
            obs_inputs.append(obs)

    # ---- 3) object nodes: union of objects referenced by included observations ----
    referenced: Set[int] = set()
    ego_observed: Set[int] = set()
    for obs in obs_inputs:
        ids = {int(i) for i in obs.observed_object_ids}
        referenced |= ids
        if int(obs.vehicle_id) == int(ego.actor_id):
            ego_observed |= ids
    obj_by_id = {int(s.actor_id): s for s in objects}
    obj_states = [obj_by_id[i] for i in referenced if i in obj_by_id]
    obj_states.sort(key=lambda s: _dist2_to_ego(ego, s))
    if spec.max_object_nodes >= 0:
        obj_states = obj_states[: int(spec.max_object_nodes)]
    obj_id_to_idx = {int(s.actor_id): i for i, s in enumerate(obj_states)}

    data = HeteroData()

    # ---- vehicle node features ----
    veh_dim = vehicle_state_dim(K)
    veh_x = np.zeros((len(veh_inputs), veh_dim), dtype=np.float32)
    veh_slot = np.zeros((len(veh_inputs),), dtype=np.int64)
    veh_is_ego = np.zeros((len(veh_inputs),), dtype=np.float32)
    veh_node_id = np.zeros((len(veh_inputs),), dtype=np.int64)
    for i, v in enumerate(veh_inputs):
        px, py = frame.xy(v.x, v.y)
        vx, vy = frame.vec(v.vx, v.vy)
        cyaw, syaw = frame.yaw_cos_sin(v.yaw)
        route = list(v.route_xy)[:K]
        route_feat: List[float] = []
        for j in range(K):
            if j < len(route):
                rx, ry = frame.xy(route[j][0], route[j][1])
            else:
                rx, ry = 0.0, 0.0
            route_feat.extend((rx, ry))
        veh_x[i] = [px, py, float(v.z), vx, vy, cyaw, syaw, float(v.q_comm), float(v.q_comp)] + route_feat
        veh_slot[i] = int(v.agent_slot)
        veh_is_ego[i] = 1.0 if v.is_ego else 0.0
        veh_node_id[i] = int(v.actor_id)

    data[VEHICLE].x = torch.from_numpy(veh_x)
    data[VEHICLE].agent_slot = torch.from_numpy(veh_slot)
    data[VEHICLE].is_ego = torch.from_numpy(veh_is_ego)
    data[VEHICLE].node_id = torch.from_numpy(veh_node_id)
    data[VEHICLE].node_mask = torch.ones((len(veh_inputs),), dtype=torch.float32)

    # ---- object node features (pad to >=1 with a masked dummy) ----
    n_obj = len(obj_states)
    n_obj_pad = max(n_obj, 1)
    obj_x = np.zeros((n_obj_pad, OBJECT_STATE_DIM), dtype=np.float32)
    obj_class = np.zeros((n_obj_pad,), dtype=np.int64)
    obj_node_id = -np.ones((n_obj_pad,), dtype=np.int64)
    obj_notable = np.zeros((n_obj_pad,), dtype=np.float32)
    obj_visible = np.zeros((n_obj_pad,), dtype=np.float32)
    obj_invisible = np.zeros((n_obj_pad,), dtype=np.float32)
    obj_mask = np.zeros((n_obj_pad,), dtype=np.float32)
    for i, s in enumerate(obj_states):
        px, py = frame.xy(s.x, s.y)
        vx, vy = frame.vec(s.vx, s.vy)
        cyaw, syaw = frame.yaw_cos_sin(s.yaw)
        obj_x[i] = [px, py, float(s.z), vx, vy, cyaw, syaw, float(s.length), float(s.width), float(s.height), 0.0]
        obj_class[i] = CLASS_TO_ID.get(str(s.object_class), CLASS_TO_ID["other"])
        obj_node_id[i] = int(s.actor_id)
        oid = int(s.actor_id)
        if visibility_override is not None:
            visible = bool(visibility_override.get(oid, False))
        else:
            visible = oid in ego_observed
        invisible = (not visible) and (oid in referenced)
        obj_visible[i] = 1.0 if visible else 0.0
        obj_invisible[i] = 1.0 if invisible else 0.0
        obj_notable[i] = 1.0 if oid in notable_ids else 0.0
        obj_mask[i] = 1.0

    data[OBJECT].x = torch.from_numpy(obj_x)
    data[OBJECT].class_id = torch.from_numpy(obj_class)
    data[OBJECT].node_id = torch.from_numpy(obj_node_id)
    data[OBJECT].notable = torch.from_numpy(obj_notable)
    data[OBJECT].visible = torch.from_numpy(obj_visible)
    data[OBJECT].invisible = torch.from_numpy(obj_invisible)
    data[OBJECT].node_mask = torch.from_numpy(obj_mask)

    # ---- observation node features ----
    n_obs = len(obs_inputs)
    obs_x = np.zeros((n_obs, OBS_SCALAR_DIM), dtype=np.float32)
    obs_modality = np.zeros((n_obs,), dtype=np.int64)
    obs_slot = np.zeros((n_obs,), dtype=np.int64)
    obs_veh_id = np.zeros((n_obs,), dtype=np.int64)
    for i, obs in enumerate(obs_inputs):
        obs_x[i] = [
            float(obs.payload_bytes) / 1024.0,
            float(obs.latency_s),
            float(obs.freshness),
            float(obs.quality),
            float(obs.sample_age_s),
        ]
        obs_modality[i] = MODALITY_TO_ID.get(str(obs.modality), 0)
        obs_slot[i] = int(veh_inputs[veh_id_to_idx[int(obs.vehicle_id)]].agent_slot)
        obs_veh_id[i] = int(obs.vehicle_id)

    data[OBSERVATION].x = torch.from_numpy(obs_x)
    data[OBSERVATION].modality_id = torch.from_numpy(obs_modality)
    data[OBSERVATION].agent_slot = torch.from_numpy(obs_slot)
    data[OBSERVATION].node_id = torch.from_numpy(obs_veh_id)
    data[OBSERVATION].node_mask = torch.ones((n_obs,), dtype=torch.float32)

    # ---- per-node BEV raster (§5.3): attach only when at least one observation carries B^sem ----
    bev_shape = next((o.bev_raster.shape for o in obs_inputs if o.bev_raster is not None), None)
    if bev_shape is not None:
        bev_stack = np.zeros((n_obs,) + tuple(bev_shape), dtype=np.float32)
        for i, obs in enumerate(obs_inputs):
            if obs.bev_raster is not None:
                bev_stack[i] = np.asarray(obs.bev_raster, dtype=np.float32)
        data[OBSERVATION].bev_raster = torch.from_numpy(bev_stack)

    # ---- edges ----
    vo_src, vo_dst = [], []
    oo_src, oo_dst, oo_attr = [], [], []
    for i, obs in enumerate(obs_inputs):
        vo_src.append(veh_id_to_idx[int(obs.vehicle_id)])
        vo_dst.append(i)
        for oid in obs.observed_object_ids:
            j = obj_id_to_idx.get(int(oid))
            if j is not None:
                oo_src.append(i)
                oo_dst.append(j)
                # obs_obj edge attr: detection confidence of this (observation, object) pair (§11).
                oo_attr.append(float(obs.det_confidence_by_object.get(int(oid), 1.0)))

    ego_idx = veh_id_to_idx[int(ego.actor_id)]
    vv_src, vv_dst, vv_attr = [], [], []
    for v in veh_inputs:
        if int(v.actor_id) in selected and not v.is_ego:
            vv_src.append(veh_id_to_idx[int(v.actor_id)])
            vv_dst.append(ego_idx)
            # veh_veh edge attr: policy-conditioned transmission latency m -> ego (§12).
            vv_attr.append(latency_by_vehicle.get(int(v.actor_id), 0.0))

    data[VEH_OBS].edge_index = _edge_index(vo_src, vo_dst)
    data[OBS_OBJ].edge_index = _edge_index(oo_src, oo_dst)
    data[OBS_OBJ].edge_attr = torch.tensor(oo_attr, dtype=torch.float32).reshape(-1, OBS_OBJ_EDGE_DIM)
    data[VEH_VEH].edge_index = _edge_index(vv_src, vv_dst)
    data[VEH_VEH].edge_attr = torch.tensor(vv_attr, dtype=torch.float32).reshape(-1, VEH_VEH_EDGE_DIM)

    return data


def hetero_graph_stats(data: HeteroData) -> Dict[str, int]:
    """Plain-int node/edge counts for logging / ``info`` (no tensors)."""

    def n_nodes(node_type: str) -> int:
        store = data[node_type]
        if hasattr(store, "node_mask"):
            return int(store.node_mask.sum().item())
        return int(store.num_nodes or 0)

    def n_edges(edge_type) -> int:
        return int(data[edge_type].edge_index.shape[1])

    return {
        "wam_graph_num_vehicle_nodes": n_nodes(VEHICLE),
        "wam_graph_num_object_nodes": n_nodes(OBJECT),
        "wam_graph_num_observation_nodes": n_nodes(OBSERVATION),
        "wam_graph_num_veh_obs_edges": n_edges(VEH_OBS),
        "wam_graph_num_obs_obj_edges": n_edges(OBS_OBJ),
        "wam_graph_num_veh_veh_edges": n_edges(VEH_VEH),
    }
