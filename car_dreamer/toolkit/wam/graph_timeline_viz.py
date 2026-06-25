"""Per-step visualization of the WAM global cooperative graph over time.

This renders the layered structure of the ego-centric cooperative hetero-graph (``env._wam_graph``,
a :class:`torch_geometric.data.HeteroData` from :func:`build_wam_hetero_graph`) so the user can watch
how it evolves across time steps -- under a single active policy (which changes over its lifetime
``Td``) or compared across several fixed policies at the same step.

Pipeline (mirrors the repo's record -> offline-viz split):

    HeteroData --hetero_graph_to_record--> plain JSON record (CARLA/torch-free)
              --layout_layered--> node positions
              --render_graph_matplotlib / _draw_on_ax--> a matplotlib figure
              --write_graph_frames_png / _gif / _html--> per-step PNG, animated GIF, slider HTML

The diagram has three rows: **vehicles** (ego + collaborators), **observations** (one Object-list /
BEV node per vehicle x modality), and **objects**; with ``veh_obs`` (vehicle->obs), ``obs_obj``
(obs->object) and ``veh_veh`` (collaborator->ego, curved, carrying the measured latency ``L_M``) edges.

Only :func:`hetero_graph_to_record` touches torch tensors (via their ``.tolist()`` methods); everything
downstream operates on plain dicts, so the rendering/HTML/GIF path is import-light and CARLA-free.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

from .graph import MODALITIES, OBJECT, OBJECT_CLASSES, OBS_OBJ, OBSERVATION, VEH_OBS, VEH_VEH, VEHICLE

PathLike = Union[str, Path]

# --- colours (loosely matching the reference sketch) ---
COLOR_EGO = "#5B8FF9"          # ego vehicle (blue)
COLOR_COLLAB = "#C7A0E8"       # collaborator vehicle (purple)
COLOR_OBS_EGO_OBJLIST = "#BDE3A6"   # ego object-list (green)
COLOR_OBS_EGO_BEV = "#F6BD7B"       # ego BEV (orange)
COLOR_OBS_COLLAB = "#F5E6A0"        # collaborator observation (buff/yellow)
COLOR_OBJECT = "#E8A0A0"            # object notable to ego (red)
COLOR_OBJECT_UNIMPORTANT = "#C9C9C9"  # non-notable object (gray)
COLOR_EDGE = "#333333"

# --- BEV overlay palette (matplotlib hex; matches BirdeyeRenderer Color.* fill colours) ---
BEV_EGO_COLOR = "#F57900"          # Color.ORANGE_1
BEV_VEHICLE_COLOR = "#00FF00"      # Color.GREEN
BEV_NOTABLE_COLOR = "#EF2929"      # Color.SCARLET_RED_0 (notable object)
BEV_PEDESTRIAN_COLOR = "#FCAF3E"   # Color.ORANGE_0
BEV_BICYCLE_COLOR = "#AD7FA8"      # Color.PLUM_0
BEV_VISIBLE_STYLE = "solid"        # ego can see the object
BEV_COLLAB_ONLY_STYLE = (0, (4, 2))  # only a collaborator sees it (dashed)
# black halo keeps fill-coloured labels legible on both road and vehicle boxes
_LABEL_HALO = [patheffects.withStroke(linewidth=2.2, foreground="black", alpha=0.9)]
_DEFAULT_OBJ_BOX_M = (4.6, 2.0)    # fallback length, width (m) when an object carries no extent
_MAP_CROP_HALF_SHRINK = 0.82       # tighten fixed-map BEV window (~18% smaller than bbox+margin)

# --- fixed figure geometry (constant across frames so the slider doesn't jitter) ---
FIG_HEIGHT = 5.6
TOPO_WIDTH = 8.0
BEV_WIDTH = 5.6
BEV_FALLBACK_RANGE_M = 50.0  # scatter-BEV half-range (m) when no birdeye image is available
UNC_HEIGHT = 2.0             # height (in) of the optional bottom uncertainty-vs-step panel
UNC_MOTION_COLOR = "#1f77b4"
UNC_COVERAGE_COLOR = "#2ca02c"
UNC_TOTAL_COLOR = "#d62728"


@dataclass
class BevOptions:
    """How to draw the BEV panel.

    ``birdeye_dir`` points at the CARLA birdeye dumps (``data/birdeye_frames``); when an image is
    found for the frame's ego/step it is used as the map background and the policy graph's nodes are
    overlaid on top. ``obs_range`` / ``ego_offset`` are the birdeye's calibration (defaults match the
    ``birdeye_wpt`` observation: 64 m range, ego 12 m below center) used to project ego-frame meters
    to birdeye pixels: ``ppm = W / obs_range``, ego at ``(W/2, H/2 + (obs_range/2 - ego_offset)*ppm)``,
    forward → up, right → +x.
    """

    birdeye_dir: Optional[PathLike] = None
    obs_range: float = 64.0
    ego_offset: float = 12.0
    mode: str = "auto"  # generated | auto | birdeye-dir | scatter
    frame: str = "map"  # map | birdeye | world | episode_start
    margin_m: float = 10.0
    show_candidates: bool = True
    contexts: Optional[Mapping[int, Mapping[str, Any]]] = None

# --- layout constants (data coordinates) ---
ROW_Y = {"vehicle": 2.0, "observation": 1.0, "object": 0.0}
VEH_SPACING = 2.0
OBS_SPACING = 1.0
OBJ_SPACING = 1.2
VEH_RADIUS = 0.20
OBJ_RADIUS = 0.16
OBS_W, OBS_H = 1.1, 0.40


# =====================================================================
# 1) Extraction: HeteroData -> plain-JSON record
# =====================================================================


def _as_list(store, attr) -> List:
    value = getattr(store, attr, None)
    if value is None:
        return []
    return value.tolist()


def _edge_records(graph, edge_type, *, with_attr: bool) -> List[Dict[str, float]]:
    store = graph[edge_type]
    edge_index = getattr(store, "edge_index", None)
    if edge_index is None:
        return []
    src, dst = edge_index.tolist() if edge_index.numel() else ([], [])
    attrs = None
    if with_attr:
        edge_attr = getattr(store, "edge_attr", None)
        if edge_attr is not None and edge_attr.numel():
            attrs = edge_attr.tolist()
    out: List[Dict[str, float]] = []
    for j in range(len(src)):
        edge: Dict[str, float] = {"src": int(src[j]), "dst": int(dst[j])}
        if attrs is not None:
            value = attrs[j]
            edge["attr"] = float(value[0] if isinstance(value, (list, tuple)) else value)
        out.append(edge)
    return out


def hetero_graph_to_record(
    graph,
    *,
    step: int,
    policy_label: str,
    policy_id: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract a JSON-serializable structural record from a WAM ``HeteroData`` graph."""
    veh_node_id = _as_list(graph[VEHICLE], "node_id")
    veh_is_ego = _as_list(graph[VEHICLE], "is_ego")
    veh_slot = _as_list(graph[VEHICLE], "agent_slot")
    veh_feat = _as_list(graph[VEHICLE], "x")  # ego-frame state: [x, y, z, vx, vy, cos_yaw, sin_yaw, ...]
    vehicles = []
    for i in range(len(veh_node_id)):
        feat = veh_feat[i] if i < len(veh_feat) else []
        vehicles.append(
            {
                "idx": i,
                "node_id": int(veh_node_id[i]),
                "is_ego": bool(veh_is_ego[i] >= 0.5),
                "slot": int(veh_slot[i]) if i < len(veh_slot) else i,
                # ego-frame spatial position (x=forward, y=right) + heading, for the BEV panel
                "x": float(feat[0]) if len(feat) > 1 else 0.0,
                "y": float(feat[1]) if len(feat) > 1 else 0.0,
                "heading_cos": float(feat[5]) if len(feat) > 6 else 1.0,
                "heading_sin": float(feat[6]) if len(feat) > 6 else 0.0,
            }
        )

    obs_vid = _as_list(graph[OBSERVATION], "node_id")
    obs_mod = _as_list(graph[OBSERVATION], "modality_id")
    obs_x = _as_list(graph[OBSERVATION], "x")
    observations = []
    for i in range(len(obs_vid)):
        scalars = list(obs_x[i]) if i < len(obs_x) else []
        scalars = (scalars + [0.0] * 5)[:5]
        mod_id = int(obs_mod[i]) if i < len(obs_mod) else 0
        observations.append(
            {
                "idx": i,
                "vehicle_id": int(obs_vid[i]),
                "modality": MODALITIES[mod_id] if 0 <= mod_id < len(MODALITIES) else str(mod_id),
                "payload_kb": float(scalars[0]),
                "latency_s": float(scalars[1]),
                "freshness": float(scalars[2]),
                "quality": float(scalars[3]),
                "sample_age_s": float(scalars[4]),
            }
        )

    obj_node_id = _as_list(graph[OBJECT], "node_id")
    obj_mask = _as_list(graph[OBJECT], "node_mask") or [1.0] * len(obj_node_id)
    obj_notable = _as_list(graph[OBJECT], "notable") or [0.0] * len(obj_node_id)
    obj_visible = _as_list(graph[OBJECT], "visible") or [0.0] * len(obj_node_id)
    obj_invisible = _as_list(graph[OBJECT], "invisible") or [0.0] * len(obj_node_id)
    obj_class = _as_list(graph[OBJECT], "class_id") or [0] * len(obj_node_id)
    obj_feat = _as_list(graph[OBJECT], "x")  # ego-frame state: [x, y, z, vx, vy, cos, sin, len, wid, hgt, 0]
    objects = []
    for i in range(len(obj_node_id)):
        feat = obj_feat[i] if i < len(obj_feat) else []
        cid = int(obj_class[i]) if i < len(obj_class) else 0
        objects.append(
            {
                "idx": i,
                "node_id": int(obj_node_id[i]),
                "valid": bool(obj_mask[i] >= 0.5),
                "notable": bool(obj_notable[i] >= 0.5),
                "visible": bool(obj_visible[i] >= 0.5),
                "invisible": bool(obj_invisible[i] >= 0.5),
                "class_id": cid,
                "object_class": OBJECT_CLASSES[cid] if 0 <= cid < len(OBJECT_CLASSES) else "other",
                "x": float(feat[0]) if len(feat) > 1 else 0.0,  # ego-frame forward
                "y": float(feat[1]) if len(feat) > 1 else 0.0,  # ego-frame right
                "heading_cos": float(feat[5]) if len(feat) > 6 else 1.0,
                "heading_sin": float(feat[6]) if len(feat) > 6 else 0.0,
                "length": float(feat[7]) if len(feat) > 7 else 0.0,
                "width": float(feat[8]) if len(feat) > 8 else 0.0,
            }
        )

    edges = {
        "veh_obs": _edge_records(graph, VEH_OBS, with_attr=False),
        "obs_obj": _edge_records(graph, OBS_OBJ, with_attr=True),
        "veh_veh": _edge_records(graph, VEH_VEH, with_attr=True),
    }
    counts = {
        "vehicles": len(vehicles),
        "observations": len(observations),
        "objects": sum(1 for o in objects if o["valid"]),
        "veh_veh": len(edges["veh_veh"]),
    }
    record: Dict[str, Any] = {
        "step": int(step),
        "policy_label": str(policy_label),
        "policy_id": None if policy_id is None else int(policy_id),
        "is_v2v": counts["veh_veh"] > 0,
        "vehicles": vehicles,
        "observations": observations,
        "objects": objects,
        "edges": edges,
        "counts": counts,
    }
    if extra:
        record["extra"] = dict(extra)
    return record


# =====================================================================
# 2) Layout: assign (x, y) to every node
# =====================================================================


def _collab_spots(n: int, spacing: float) -> List[float]:
    """Alternating non-zero x slots ``[-1, +1, -2, +2, ...] * spacing`` (ego keeps x=0)."""
    spots: List[float] = []
    k = 1
    while len(spots) < n:
        spots.append(-k * spacing)
        if len(spots) < n:
            spots.append(k * spacing)
        k += 1
    return spots[:n]


def _centered_spots(m: int, spacing: float) -> List[float]:
    if m <= 1:
        return [0.0]
    start = -(m - 1) / 2.0
    return [(start + i) * spacing for i in range(m)]


def layout_layered(record: Mapping[str, Any]) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Three-row layout: vehicles on top, observations in the middle, objects at the bottom."""
    vehicles = list(record["vehicles"])
    observations = list(record["observations"])
    objects = [o for o in record["objects"] if o.get("valid", True)]

    pos: Dict[Tuple[str, int], Tuple[float, float]] = {}
    veh_x: Dict[int, float] = {}
    collabs = sorted((v for v in vehicles if not v["is_ego"]), key=lambda v: v["slot"])
    for v in vehicles:
        if v["is_ego"]:
            veh_x[v["idx"]] = 0.0
    for v, sx in zip(collabs, _collab_spots(len(collabs), VEH_SPACING)):
        veh_x[v["idx"]] = sx
    for v in vehicles:
        pos[("vehicle", v["idx"])] = (veh_x.get(v["idx"], 0.0), ROW_Y["vehicle"])

    vid_to_x = {v["node_id"]: veh_x.get(v["idx"], 0.0) for v in vehicles}
    by_vehicle: Dict[int, List[dict]] = {}
    for o in observations:
        by_vehicle.setdefault(int(o["vehicle_id"]), []).append(o)
    for vid, obs_list in by_vehicle.items():
        base_x = vid_to_x.get(vid, 0.0)
        for o, off in zip(obs_list, _centered_spots(len(obs_list), OBS_SPACING)):
            pos[("observation", o["idx"])] = (base_x + off, ROW_Y["observation"])

    obj_xs = _centered_spots(len(objects), OBJ_SPACING) if objects else []
    for o, ox in zip(objects, obj_xs):
        pos[("object", o["idx"])] = (ox, ROW_Y["object"])
    return pos


# =====================================================================
# 3) Rendering (matplotlib)
# =====================================================================


def _observation_color(is_ego: bool, modality: str) -> str:
    if is_ego:
        return COLOR_OBS_EGO_BEV if modality == "bev" else COLOR_OBS_EGO_OBJLIST
    return COLOR_OBS_COLLAB


def _arrow(ax, p0, p1, *, rad: float = 0.0, lw: float = 1.2, color: str = COLOR_EDGE) -> None:
    style = "arc3,rad=%.3f" % rad
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", connectionstyle=style, mutation_scale=12,
            lw=lw, color=color, shrinkA=13, shrinkB=13, zorder=1,
        )
    )


def _draw_on_ax(ax, record: Mapping[str, Any]) -> None:
    """Draw one cooperative-graph record onto ``ax``."""
    pos = layout_layered(record)
    veh_by_idx = {v["idx"]: v for v in record["vehicles"]}
    ego_idx = next((v["idx"] for v in record["vehicles"] if v["is_ego"]), None)

    # enumerate collaborators 1..n for friendly short labels (V1, V2, ...)
    collab_label: Dict[int, str] = {}
    for n, v in enumerate(sorted((v for v in record["vehicles"] if not v["is_ego"]), key=lambda v: v["slot"]), start=1):
        collab_label[v["idx"]] = f"V{n}"

    # ---- edges first (under the nodes) ----
    for e in record["edges"]["veh_obs"]:
        p0 = pos.get(("vehicle", e["src"]))
        p1 = pos.get(("observation", e["dst"]))
        if p0 and p1:
            _arrow(ax, p0, p1)
    for e in record["edges"]["obs_obj"]:
        p0 = pos.get(("observation", e["src"]))
        p1 = pos.get(("object", e["dst"]))
        if p0 and p1:
            _arrow(ax, p0, p1)
    for e in record["edges"]["veh_veh"]:
        p0 = pos.get(("vehicle", e["src"]))
        p1 = pos.get(("vehicle", e["dst"]))
        if not (p0 and p1):
            continue
        rad = -0.35 if p0[0] <= p1[0] else 0.35  # arc outward from ego
        _arrow(ax, p0, p1, rad=rad, lw=1.5)
        mx, my = (p0[0] + p1[0]) / 2.0, max(p0[1], p1[1]) + 0.28
        ax.text(mx, my, f"L={float(e.get('attr', 0.0)):.2f}s", fontsize=7, ha="center", color=COLOR_EDGE)

    # ---- vehicle nodes ----
    for v in record["vehicles"]:
        x, y = pos[("vehicle", v["idx"])]
        is_ego = bool(v["is_ego"])
        ax.add_patch(Circle((x, y), VEH_RADIUS, fc=COLOR_EGO if is_ego else COLOR_COLLAB, ec="black", lw=1.5, zorder=3))
        label = "ego" if is_ego else collab_label.get(v["idx"], f"V{v['node_id']}")
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, weight="bold", zorder=4)
        if not is_ego:
            ax.text(x, y - VEH_RADIUS - 0.14, f"id={v['node_id']}", ha="center", va="top", fontsize=6, color="#666")

    # ---- observation nodes ----
    for o in record["observations"]:
        key = ("observation", o["idx"])
        if key not in pos:
            continue
        x, y = pos[key]
        is_ego = bool(veh_by_idx.get(_idx_of_vehicle(record, o["vehicle_id"]), {}).get("is_ego", False))
        color = _observation_color(is_ego, o["modality"])
        ax.add_patch(
            FancyBboxPatch(
                (x - OBS_W / 2, y - OBS_H / 2), OBS_W, OBS_H,
                boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="black", lw=1.2, zorder=3,
            )
        )
        label = "BEV" if o["modality"] == "bev" else "Object-list"
        ax.text(x, y + 0.04, label, ha="center", va="center", fontsize=7.5, weight="bold", zorder=4)
        ax.text(
            x, y - 0.13, f"L={o['latency_s']:.2f} f={o['freshness']:.2f}",
            ha="center", va="center", fontsize=5.8, color="#444", zorder=4,
        )

    # ---- object nodes ----
    for o in record["objects"]:
        key = ("object", o["idx"])
        if key not in pos:
            continue
        x, y = pos[key]
        # fill encodes ego-notability: red = notable to ego, gray = not important
        fill = COLOR_OBJECT if o["notable"] else COLOR_OBJECT_UNIMPORTANT
        ax.add_patch(Circle((x, y), OBJ_RADIUS, fc=fill, ec="black", lw=1.2, zorder=3))
        ax.text(x, y, f"O{o['node_id']}", ha="center", va="center", fontsize=6.5, zorder=4)

    # ---- framing ----
    xs = [p[0] for p in pos.values()] or [0.0]
    title = (
        f"step {record['step']}  |  policy: {record['policy_label']}"
        f"  |  {'V2V' if record.get('is_v2v') else 'local'}"
        f"  |  veh={record['counts']['vehicles']} obs={record['counts']['observations']}"
        f" obj={record['counts']['objects']}"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(-0.8, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")


def _idx_of_vehicle(record: Mapping[str, Any], vehicle_node_id: int) -> int:
    for v in record["vehicles"]:
        if int(v["node_id"]) == int(vehicle_node_id):
            return int(v["idx"])
    return -1


def _resolve_birdeye_path(record: Mapping[str, Any], birdeye_dir: Optional[PathLike]):
    """Locate the real CARLA birdeye PNG for this frame: ``<dir>/vehicle_<egoid>/birdeye_<step>.png``."""
    if not birdeye_dir:
        return None
    ego = next((v for v in record["vehicles"] if v.get("is_ego")), None)
    if ego is None:
        return None
    path = Path(birdeye_dir) / f"vehicle_{int(ego['node_id'])}" / f"birdeye_{int(record.get('step', 0)):06d}.png"
    return path if path.exists() else None


def _draw_bev_scatter(ax, record: Mapping[str, Any]) -> None:
    """Labelled ego-centric scatter BEV (fallback when no birdeye image is available)."""
    vehicles = list(record["vehicles"])
    objects = [o for o in record["objects"] if o.get("valid", True)]
    if not (any("x" in v for v in vehicles) or any("x" in o for o in objects)):
        ax.text(0.5, 0.5, "no birdeye image and no positions\n(pass --birdeye-dir or re-record)",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#999")
        ax.axis("off")
        return

    collab_label: Dict[int, str] = {}
    for n, v in enumerate(sorted((v for v in vehicles if not v["is_ego"]), key=lambda v: v["slot"]), start=1):
        collab_label[v["idx"]] = f"V{n}"

    # screen mapping: screen_x = y (right), screen_y = x (forward, up)
    pts = [(0.0, 0.0)]
    pts += [(float(o.get("y", 0.0)), float(o.get("x", 0.0))) for o in objects]
    pts += [(float(v.get("y", 0.0)), float(v.get("x", 0.0))) for v in vehicles]
    span = max((abs(px) for px, _ in pts), default=0.0)
    span = max(span, max((abs(py) for _, py in pts), default=0.0))
    r = max(BEV_FALLBACK_RANGE_M, span + 5.0)  # bigger range so distant objects aren't clipped
    arrow_len = r * 0.06

    for o in objects:
        sx, sy = float(o.get("y", 0.0)), float(o.get("x", 0.0))
        fill = COLOR_OBJECT if o["notable"] else COLOR_OBJECT_UNIMPORTANT
        ax.scatter([sx], [sy], s=110, c=fill, edgecolors="black", linewidths=0.8, zorder=3)
        ax.text(sx, sy, f"O{o['node_id']}", fontsize=5.0, ha="center", va="center", zorder=4)

    for v in vehicles:
        sx, sy = float(v.get("y", 0.0)), float(v.get("x", 0.0))
        is_ego = bool(v["is_ego"])
        ax.scatter([sx], [sy], s=200, marker="s", c=COLOR_EGO if is_ego else COLOR_COLLAB,
                   edgecolors="black", linewidths=1.2, zorder=5)
        label = "ego" if is_ego else collab_label.get(v["idx"], f"V{v['node_id']}")
        ax.text(sx, sy + r * 0.04, label, fontsize=6.5, weight="bold", ha="center", va="bottom", zorder=6)
        hc, hs = float(v.get("heading_cos", 1.0)), float(v.get("heading_sin", 0.0))
        ax.annotate("", xy=(sx + arrow_len * hs, sy + arrow_len * hc), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0), zorder=4)

    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.axhline(0, color="#ccc", lw=0.6, zorder=0)
    ax.axvline(0, color="#ccc", lw=0.6, zorder=0)
    ax.set_xlabel("lateral (m) →right", fontsize=7)
    ax.set_ylabel("forward (m) ↑", fontsize=7)
    ax.tick_params(labelsize=6)


def _world_pose(item: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    try:
        return float(item["x"]), float(item["y"]), float(item.get("yaw", 0.0))
    except Exception:
        return None


def _record_extra(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = record.get("extra", {})
    return extra if isinstance(extra, Mapping) else {}


def _episode_id(record: Mapping[str, Any]) -> int:
    return int(_record_extra(record).get("episode", 0))


def _iter_world_items(record: Mapping[str, Any], *, include_objects: bool = True):
    extra = _record_extra(record)
    ego = extra.get("ego_world")
    if isinstance(ego, Mapping):
        yield ego
    for candidate in extra.get("candidate_world", ()) or ():
        if isinstance(candidate, Mapping):
            yield candidate
    if include_objects:
        for obj in extra.get("graph_object_world", ()) or ():
            if isinstance(obj, Mapping):
                yield obj


def _frame_origin(record: Mapping[str, Any]) -> Tuple[float, float, float]:
    ego = _record_extra(record).get("ego_world")
    pose = _world_pose(ego) if isinstance(ego, Mapping) else None
    return pose or (0.0, 0.0, 0.0)


def _relative_to_pose(x: float, y: float, origin: Tuple[float, float, float]) -> Tuple[float, float]:
    ox, oy, oyaw = origin
    yaw = math.radians(float(oyaw))
    dx, dy = float(x) - float(ox), float(y) - float(oy)
    fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
    right = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return fwd, right


def _to_bev_xy(x: float, y: float, ctx: Mapping[str, Any]) -> Tuple[float, float]:
    frame = str(ctx.get("frame", "episode_start"))
    if frame == "world":
        return float(x), float(y)
    ox, oy, oyaw = float(ctx.get("origin_x", 0.0)), float(ctx.get("origin_y", 0.0)), float(ctx.get("origin_yaw", 0.0))
    yaw = math.radians(oyaw)
    dx, dy = float(x) - ox, float(y) - oy
    # episode-start frame: screen x = right, screen y = forward, fixed to the first ego pose.
    fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
    right = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return right, fwd


def _heading_in_context(yaw_deg: float, ctx: Mapping[str, Any]) -> float:
    frame = str(ctx.get("frame", "episode_start"))
    return float(yaw_deg) if frame == "world" else float(yaw_deg) - float(ctx.get("origin_yaw", 0.0))


def _all_episode_points(records: Sequence[Mapping[str, Any]], ctx: Mapping[str, Any]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for rec in records:
        for item in _iter_world_items(rec, include_objects=True):
            pose = _world_pose(item)
            if pose is not None:
                points.append(_to_bev_xy(pose[0], pose[1], ctx))
    return points


def _episode_birdeye_range(records: Sequence[Mapping[str, Any]], opts: BevOptions) -> Tuple[float, float]:
    base_range = max(float(getattr(opts, "obs_range", 64.0)), 1.0)
    base_offset = float(getattr(opts, "ego_offset", 12.0))
    offset_ratio = min(max(base_offset / base_range, 0.05), 0.5)
    margin = max(float(getattr(opts, "margin_m", 10.0)), 0.0)
    required = base_range
    for rec in records:
        ego_pose = _frame_origin(rec)
        for item in _iter_world_items(rec, include_objects=True):
            pose = _world_pose(item)
            if pose is None:
                continue
            fwd, right = _relative_to_pose(pose[0], pose[1], ego_pose)
            required = max(required, 2.0 * (abs(right) + margin))
            if fwd >= 0.0:
                required = max(required, (fwd + margin) / max(1.0 - offset_ratio, 1e-6))
            else:
                required = max(required, (-fwd + margin) / max(offset_ratio, 1e-6))
    return float(required), float(required * offset_ratio)


def _episode_map_crop(records: Sequence[Mapping[str, Any]], bg: Mapping[str, Any], *, margin_m: float) -> Optional[Tuple[int, int, int, int]]:
    """One fixed map-pixel window (x0, y0, x1, y1) covering every episode actor + margin.

    Computed once per episode so the recorded CARLA map background stays put while the ego box moves
    across it (background does NOT follow the ego).
    """
    eppm = float(bg.get("pixels_per_meter", 1.0)) * float(bg.get("scale", 1.0))
    width = int(bg.get("width_px", 0))
    height = int(bg.get("height_px", 0))
    if eppm <= 0.0 or width <= 0 or height <= 0:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for rec in records:
        for item in _iter_world_items(rec, include_objects=True):
            for corner in item.get("bbox") or ():
                if len(corner) >= 2:
                    px, py = _world_to_map_pixel(float(corner[0]), float(corner[1]), bg)
                    xs.append(px)
                    ys.append(py)
            pose = _world_pose(item)
            if pose is not None:
                px, py = _world_to_map_pixel(pose[0], pose[1], bg)
                xs.append(px)
                ys.append(py)
    if not xs:
        return None
    margin_px = max(float(margin_m), 0.0) * eppm
    x0, x1 = min(xs) - margin_px, max(xs) + margin_px
    y0, y1 = min(ys) - margin_px, max(ys) + margin_px
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max((x1 - x0) / 2.0, (y1 - y0) / 2.0, 6.0 * eppm) * _MAP_CROP_HALF_SHRINK
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    x0 = int(max(0, math.floor(x0)))
    y0 = int(max(0, math.floor(y0)))
    x1 = int(min(width, math.ceil(x1)))
    y1 = int(min(height, math.ceil(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def build_episode_bev_contexts(records: Sequence[Mapping[str, Any]], bev: Optional[BevOptions] = None) -> Dict[int, Dict[str, Any]]:
    """Compute one fixed generated-BEV extent per episode, never per step."""
    opts = bev or BevOptions()
    by_ep: Dict[int, List[Mapping[str, Any]]] = {}
    for rec in records:
        by_ep.setdefault(_episode_id(rec), []).append(rec)
    contexts: Dict[int, Dict[str, Any]] = {}
    for ep, recs in by_ep.items():
        first = recs[0] if recs else {}
        ox, oy, oyaw = _frame_origin(first)
        ctx: Dict[str, Any] = {
            "episode": int(ep),
            "frame": str(getattr(opts, "frame", "episode_start")),
            "origin_x": float(ox),
            "origin_y": float(oy),
            "origin_yaw": float(oyaw),
        }
        for rec in recs:
            bg = _record_extra(rec).get("map_background")
            if isinstance(bg, Mapping):
                ctx["map_background"] = dict(bg)
                break
        if str(ctx["frame"]) == "birdeye":
            obs_range, ego_offset = _episode_birdeye_range(recs, opts)
            ctx.update({"obs_range": obs_range, "ego_offset": ego_offset, "image_size": 512})
        if str(ctx["frame"]) == "map" and isinstance(ctx.get("map_background"), Mapping):
            crop = _episode_map_crop(recs, ctx["map_background"], margin_m=float(getattr(opts, "margin_m", 10.0)))
            if crop is not None:
                ctx["crop"] = crop
        pts = _all_episode_points(recs, ctx)
        if not pts:
            r = BEV_FALLBACK_RANGE_M
            ctx.update({"xmin": -r, "xmax": r, "ymin": -r, "ymax": r})
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            margin = max(float(getattr(opts, "margin_m", 10.0)), 0.0)
            xmin, xmax = min(xs) - margin, max(xs) + margin
            ymin, ymax = min(ys) - margin, max(ys) + margin
            span = max(xmax - xmin, ymax - ymin, 2.0 * BEV_FALLBACK_RANGE_M)
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            half = span / 2.0
            ctx.update({"xmin": cx - half, "xmax": cx + half, "ymin": cy - half, "ymax": cy + half})
        contexts[int(ep)] = ctx
    return contexts


def _bev_context_for_record(record: Mapping[str, Any], bev: Optional[BevOptions]) -> Mapping[str, Any]:
    if bev is not None and bev.contexts:
        ctx = bev.contexts.get(_episode_id(record))
        if ctx is not None:
            return ctx
    return build_episode_bev_contexts([record], bev).get(_episode_id(record), {})


def _item_bbox_xy(item: Mapping[str, Any], ctx: Mapping[str, Any], *, default_length: float = 4.6, default_width: float = 2.0):
    bbox = item.get("bbox") or ()
    pts = []
    for p in bbox:
        if len(p) >= 2:
            pts.append(_to_bev_xy(float(p[0]), float(p[1]), ctx))
    if len(pts) >= 3:
        return pts
    pose = _world_pose(item)
    if pose is None:
        return []
    sx, sy = _to_bev_xy(pose[0], pose[1], ctx)
    yaw = math.radians(_heading_in_context(pose[2], ctx))
    length = float(item.get("length", default_length) or default_length)
    width = float(item.get("width", default_width) or default_width)
    hl, hw = length / 2.0, width / 2.0
    fwd = (math.sin(yaw), math.cos(yaw))
    right = (math.cos(yaw), -math.sin(yaw))
    return [
        (sx + a * hl * fwd[0] + b * hw * right[0], sy + a * hl * fwd[1] + b * hw * right[1])
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]


def _draw_world_box(ax, item: Mapping[str, Any], ctx: Mapping[str, Any], *, edge: str, face: str = "none", lw: float = 1.4,
                    alpha: float = 1.0, linestyle="solid", zorder: int = 4) -> Optional[Tuple[float, float]]:
    pts = _item_bbox_xy(item, ctx)
    if pts:
        ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor=edge, linewidth=lw,
                             alpha=alpha, linestyle=linestyle, zorder=zorder, clip_on=True))
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    pose = _world_pose(item)
    if pose is None:
        return None
    xy = _to_bev_xy(pose[0], pose[1], ctx)
    ax.scatter([xy[0]], [xy[1]], s=45, facecolors=face, edgecolors=edge, linewidths=lw, zorder=zorder, clip_on=True)
    return xy


def _draw_generated_bev_on_ax(ax, record: Mapping[str, Any], *, bev: Optional[BevOptions] = None) -> None:
    if not any(True for _ in _iter_world_items(record, include_objects=True)):
        _draw_bev_scatter(ax, record)
        return
    ctx = _bev_context_for_record(record, bev)
    if not ctx:
        _draw_bev_scatter(ax, record)
        return
    frame = str(ctx.get("frame", "map"))
    if frame == "map" and _draw_fixed_birdeye_on_ax(ax, record, ctx, bev=bev):
        return
    if frame == "birdeye" and _draw_generated_birdeye_on_ax(ax, record, ctx, bev=bev):
        return
    extra = _record_extra(record)
    graph_vehicle_ids = {int(v["node_id"]) for v in record.get("vehicles", ())}
    collab_label: Dict[int, str] = {}
    for n, v in enumerate(sorted((v for v in record["vehicles"] if not v["is_ego"]), key=lambda v: v["slot"]), start=1):
        collab_label[int(v["node_id"])] = f"V{n}"

    ax.set_facecolor("#050505")
    _draw_generated_map_background(ax, ctx)
    ax.grid(True, color="#36424E", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.axhline(0, color="#65717D", lw=0.8, alpha=0.7, zorder=0)
    ax.axvline(0, color="#65717D", lw=0.8, alpha=0.7, zorder=0)

    if bev is None or bool(getattr(bev, "show_candidates", True)):
        for cand in extra.get("candidate_world", ()) or ():
            if not isinstance(cand, Mapping):
                continue
            vid = int(cand.get("actor_id", -1))
            selected = vid in graph_vehicle_ids
            if selected:
                continue
            center = _draw_world_box(ax, cand, ctx, edge=BEV_VEHICLE_COLOR, face=BEV_VEHICLE_COLOR, lw=1.0, alpha=0.55,
                                     linestyle=(0, (3, 2)), zorder=2)
            if center is not None:
                _bev_label(ax, center[0], center[1], f"C{vid}", color=BEV_VEHICLE_COLOR, fontsize=5.5, zorder=3)

    for obj in extra.get("graph_object_world", ()) or ():
        if not isinstance(obj, Mapping):
            continue
        oid = int(obj.get("actor_id", -1))
        graph_obj = next((o for o in record.get("objects", ()) if int(o.get("node_id", -2)) == oid), {})
        notable = bool(graph_obj.get("notable", False))
        visible = bool(graph_obj.get("visible", False))
        obj_hex = _object_birdeye_hex(str(obj.get("object_class", "vehicle")), notable=notable)
        style = BEV_VISIBLE_STYLE if visible else BEV_COLLAB_ONLY_STYLE
        center = _draw_world_box(ax, obj, ctx, edge=obj_hex, face=obj_hex, lw=2.0, linestyle=style, zorder=5)
        if center is not None:
            _bev_label(ax, center[0], center[1], f"O{oid}", color=obj_hex, fontsize=6.0, zorder=6)

    ego = extra.get("ego_world")
    if isinstance(ego, Mapping):
        center = _draw_world_box(ax, ego, ctx, edge=BEV_EGO_COLOR, face=BEV_EGO_COLOR, lw=2.3, zorder=7)
        if center is not None:
            _bev_label(ax, center[0], center[1], "EGO", color=BEV_EGO_COLOR, fontsize=8.0, weight="bold", va="bottom", zorder=9)

    for cand in extra.get("candidate_world", ()) or ():
        if not isinstance(cand, Mapping):
            continue
        vid = int(cand.get("actor_id", -1))
        if vid not in graph_vehicle_ids:
            continue
        center = _draw_world_box(ax, cand, ctx, edge=BEV_VEHICLE_COLOR, face=BEV_VEHICLE_COLOR, lw=2.0, zorder=7)
        if center is not None:
            _bev_label(ax, center[0], center[1], collab_label.get(vid, f"V{vid}"),
                       color=BEV_VEHICLE_COLOR, fontsize=8.0, weight="bold", va="bottom", zorder=9)

    ax.set_xlim(float(ctx["xmin"]), float(ctx["xmax"]))
    ax.set_ylim(float(ctx["ymin"]), float(ctx["ymax"]))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6, colors="#AAB4BE")
    for spine in ax.spines.values():
        spine.set_color("#6C7782")
    ax.set_xlabel("right from episode start (m)" if str(ctx.get("frame")) != "world" else "world x (m)", fontsize=7)
    ax.set_ylabel("forward from episode start (m)" if str(ctx.get("frame")) != "world" else "world y (m)", fontsize=7)


def _birdeye_color_hex(color: Tuple[int, int, int]) -> str:
    """``Color.*`` constants are RGB-named; matplotlib needs ``#RRGGBB``."""
    r, g, b = (int(color[0]), int(color[1]), int(color[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def _object_birdeye_color(object_class: str, *, notable: bool = False) -> Tuple[int, int, int]:
    from car_dreamer.toolkit.observer.handlers.renderer.constants import Color

    if notable:
        return Color.SCARLET_RED_0
    return {
        "vehicle": Color.GREEN,
        "pedestrian": Color.ORANGE_0,
        "bicycle": Color.PLUM_0,
    }.get(str(object_class), Color.GREEN)


def _object_birdeye_hex(object_class: str, *, notable: bool = False) -> str:
    return _birdeye_color_hex(_object_birdeye_color(object_class, notable=notable))


def _birdeye_display_rotation_deg(reference_yaw_deg: float) -> float:
    """Match :func:`_ego_centric_warp` so CARLA forward points up on screen."""
    return float(reference_yaw_deg) + 90.0


def _rotate_points_2d(points: Sequence[Tuple[float, float]], matrix) -> List[Tuple[float, float]]:
    return [_apply_affine((float(px), float(py)), matrix) for px, py in points]


def _world_to_map_pixel(x: float, y: float, bg: Mapping[str, Any]) -> Tuple[float, float]:
    ppm = float(bg.get("pixels_per_meter", 1.0)) * float(bg.get("scale", 1.0))
    ox, oy = bg.get("world_offset", (0.0, 0.0))
    return ppm * (float(x) - float(ox)), ppm * (float(y) - float(oy))


def _draw_fixed_birdeye_on_ax(ax, record: Mapping[str, Any], ctx: Mapping[str, Any], *, bev: Optional[BevOptions] = None) -> bool:
    """Fixed-window BEV that reuses the *exact* birdeye look (data/birdeye_frames).

    The recorded CARLA map surface is cropped to one fixed per-episode window (background never
    follows the ego). The crop is then rotated once per episode with the same ``yaw + 90°`` rule as
    :class:`BirdeyeRenderer`, so ego forward points **up** and motion reads bottom→top like
    ``data/birdeye_frames``. Vehicles/objects use cv2 ``fillPoly`` + white outline; ego is orange
    (``Color.ORANGE_1``) so it does not blend with sky-blue lane centre lines.
    """
    bg = ctx.get("map_background")
    crop = ctx.get("crop")
    if not isinstance(bg, Mapping) or not crop:
        return False
    path = bg.get("path")
    if not path:
        return False
    try:
        import cv2
        import numpy as np

        from car_dreamer.toolkit.observer.handlers.renderer.constants import Color

        img = cv2.imread(str(path))  # BGR-on-disk == renderer surface (RGB-named) space
        if img is None:
            return False
        height, width = img.shape[:2]
        x0, y0, x1, y1 = (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))
        canvas = img[y0:y1, x0:x1].copy()
        crop_h, crop_w = canvas.shape[:2]
    except Exception:
        return False

    eppm = float(bg.get("pixels_per_meter", 1.0)) * float(bg.get("scale", 1.0))
    ox, oy = bg.get("world_offset", (0.0, 0.0))

    def w2l(x: float, y: float) -> Tuple[float, float]:
        return eppm * (float(x) - float(ox)) - x0, eppm * (float(y) - float(oy)) - y0

    def local_corners(item: Mapping[str, Any]) -> List[Tuple[float, float]]:
        return [w2l(p[0], p[1]) for p in (item.get("bbox") or ()) if len(p) >= 2]

    def fill_box(item: Mapping[str, Any], color, *, border_w: int = 1) -> Optional[Tuple[float, float]]:
        pts = local_corners(item)
        if len(pts) >= 3:
            arr = np.array([[int(round(a)), int(round(b))] for a, b in pts], dtype=np.int32)
            cv2.fillPoly(canvas, [arr], color)
            cv2.polylines(canvas, [arr], True, color, int(border_w), cv2.LINE_AA)
            return sum(a for a, _ in pts) / len(pts), sum(b for _, b in pts) / len(pts)
        pose = _world_pose(item)
        if pose is None:
            return None
        lx, ly = w2l(pose[0], pose[1])
        cv2.circle(canvas, (int(round(lx)), int(round(ly))), 5, color, -1, cv2.LINE_AA)
        return lx, ly

    extra = _record_extra(record)
    graph_vehicle_ids = {int(v["node_id"]) for v in record.get("vehicles", ())}
    collab_label: Dict[int, str] = {}
    for n, v in enumerate(sorted((v for v in record["vehicles"] if not v["is_ego"]), key=lambda v: v["slot"]), start=1):
        collab_label[int(v["node_id"])] = f"V{n}"

    vehicle_hex = _birdeye_color_hex(Color.GREEN)
    ego_hex = _birdeye_color_hex(Color.ORANGE_1)

    cand_colors: Dict[int, str] = {}
    cand_centers: Dict[int, Optional[Tuple[float, float]]] = {}
    for cand in extra.get("candidate_world", ()) or ():
        if not isinstance(cand, Mapping):
            continue
        vid = int(cand.get("actor_id", -1))
        cand_colors[vid] = vehicle_hex
        cand_centers[vid] = fill_box(cand, Color.GREEN)

    obj_colors: Dict[int, str] = {}
    obj_centers: Dict[int, Optional[Tuple[float, float]]] = {}
    for obj in extra.get("graph_object_world", ()) or ():
        if not isinstance(obj, Mapping):
            continue
        oid = int(obj.get("actor_id", -1))
        graph_obj = next((o for o in record.get("objects", ()) if int(o.get("node_id", -2)) == oid), {})
        notable = bool(graph_obj.get("notable", False))
        fill = _object_birdeye_color(str(obj.get("object_class", "vehicle")), notable=notable)
        obj_colors[oid] = _birdeye_color_hex(fill)
        border_w = 2 if notable else 1
        obj_centers[oid] = fill_box(obj, fill, border_w=border_w)

    ego = extra.get("ego_world")
    ego_center = fill_box(ego, Color.ORANGE_1, border_w=2) if isinstance(ego, Mapping) else None

    # One fixed rotation per episode: align episode-start ego heading to screen-up (birdeye convention).
    rot_center = (crop_w / 2.0, crop_h / 2.0)
    rot_deg = _birdeye_display_rotation_deg(float(ctx.get("origin_yaw", 0.0)))
    rot_m = cv2.getRotationMatrix2D(rot_center, rot_deg, 1.0)
    canvas = cv2.warpAffine(
        canvas,
        rot_m,
        (crop_w, crop_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    def to_display(px: float, py: float) -> Tuple[float, float]:
        return _apply_affine((px, py), rot_m)

    ax.imshow(canvas[:, :, ::-1], zorder=0)

    show_candidates = bev is None or bool(getattr(bev, "show_candidates", True))
    for vid, center in cand_centers.items():
        if center is None:
            continue
        center = to_display(*center)
        label = collab_label.get(vid, f"V{vid}") if vid in graph_vehicle_ids else f"C{vid}"
        if vid in graph_vehicle_ids or show_candidates:
            _bev_label(ax, center[0], center[1], label, color=cand_colors[vid],
                       fontsize=7.5 if vid in graph_vehicle_ids else 5.0,
                       weight="bold" if vid in graph_vehicle_ids else "normal",
                       va="bottom" if vid in graph_vehicle_ids else "center", zorder=8)

    for oid, center in obj_centers.items():
        if center is None:
            continue
        center = to_display(*center)
        _bev_label(ax, center[0], center[1], f"O{oid}", color=obj_colors[oid], fontsize=6.0, zorder=7)

    if ego_center is not None:
        ego_center = to_display(*ego_center)
        _bev_label(ax, ego_center[0], ego_center[1], "EGO", color=ego_hex, fontsize=8.0, weight="bold",
                   va="bottom", zorder=9)

    ax.set_aspect("equal")
    ax.axis("off")
    return True


def _birdeye_affine(record: Mapping[str, Any], ctx: Mapping[str, Any], bg: Mapping[str, Any]):
    import cv2
    import numpy as np

    ego = _record_extra(record).get("ego_world")
    pose = _world_pose(ego) if isinstance(ego, Mapping) else None
    if pose is None:
        return None
    ego_px = _world_to_map_pixel(pose[0], pose[1], bg)
    source_ppm = float(bg.get("pixels_per_meter", 1.0)) * float(bg.get("scale", 1.0))
    image_size = int(ctx.get("image_size", 512))
    obs_range = float(ctx.get("obs_range", 64.0))
    ego_offset = float(ctx.get("ego_offset", 12.0))
    output_ppm = float(image_size) / max(obs_range, 1e-6)
    scale = output_ppm / max(source_ppm, 1e-6)
    pixels_ahead_vehicle = (obs_range / 2.0 - ego_offset) * output_ppm
    matrix = cv2.getRotationMatrix2D(ego_px, float(pose[2]) + 90.0, scale)
    matrix[0][2] -= ego_px[0] - float(image_size) / 2.0
    matrix[1][2] -= ego_px[1] - float(image_size) / 2.0 - pixels_ahead_vehicle
    return np.asarray(matrix, dtype=float)


def _apply_affine(point: Tuple[float, float], matrix) -> Tuple[float, float]:
    return (
        float(matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2]),
        float(matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2]),
    )


def _item_bbox_pixels(item: Mapping[str, Any], bg: Mapping[str, Any], matrix) -> List[Tuple[float, float]]:
    bbox = item.get("bbox") or ()
    pts = []
    for p in bbox:
        if len(p) >= 2:
            pts.append(_apply_affine(_world_to_map_pixel(float(p[0]), float(p[1]), bg), matrix))
    if len(pts) >= 3:
        return pts
    pose = _world_pose(item)
    if pose is None:
        return []
    center = _world_to_map_pixel(pose[0], pose[1], bg)
    ppm = float(bg.get("pixels_per_meter", 1.0)) * float(bg.get("scale", 1.0))
    yaw = math.radians(float(pose[2]))
    length = float(item.get("length", _DEFAULT_OBJ_BOX_M[0]) or _DEFAULT_OBJ_BOX_M[0])
    width = float(item.get("width", _DEFAULT_OBJ_BOX_M[1]) or _DEFAULT_OBJ_BOX_M[1])
    hl, hw = length * ppm / 2.0, width * ppm / 2.0
    fwd = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    raw = [
        (center[0] + a * hl * fwd[0] + b * hw * right[0], center[1] + a * hl * fwd[1] + b * hw * right[1])
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
    return [_apply_affine(p, matrix) for p in raw]


def _draw_pixel_box(ax, item: Mapping[str, Any], bg: Mapping[str, Any], matrix, *, edge: str, face: str = "none",
                    lw: float = 1.4, alpha: float = 1.0, linestyle="solid", zorder: int = 4) -> Optional[Tuple[float, float]]:
    pts = _item_bbox_pixels(item, bg, matrix)
    if pts:
        ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor=edge, linewidth=lw,
                             alpha=alpha, linestyle=linestyle, zorder=zorder, clip_on=True))
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    pose = _world_pose(item)
    if pose is None:
        return None
    xy = _apply_affine(_world_to_map_pixel(pose[0], pose[1], bg), matrix)
    ax.scatter([xy[0]], [xy[1]], s=45, facecolors=face, edgecolors=edge, linewidths=lw, zorder=zorder, clip_on=True)
    return xy


def _draw_generated_birdeye_on_ax(ax, record: Mapping[str, Any], ctx: Mapping[str, Any], *, bev: Optional[BevOptions] = None) -> bool:
    bg = ctx.get("map_background")
    if not isinstance(bg, Mapping):
        return False
    matrix = _birdeye_affine(record, ctx, bg)
    if matrix is None:
        return False
    path = bg.get("path")
    if not path:
        return False
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return False
        image_size = int(ctx.get("image_size", 512))
        warped = cv2.warpAffine(
            img,
            matrix,
            (image_size, image_size),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        ax.imshow(warped[:, :, ::-1], extent=[0, image_size, image_size, 0], zorder=0)
    except Exception:
        return False

    extra = _record_extra(record)
    graph_vehicle_ids = {int(v["node_id"]) for v in record.get("vehicles", ())}
    collab_label = {
        int(v["node_id"]): f"V{n}"
        for n, v in enumerate(sorted((v for v in record["vehicles"] if not v["is_ego"]), key=lambda v: v["slot"]), start=1)
    }
    if bev is None or bool(getattr(bev, "show_candidates", True)):
        for cand in extra.get("candidate_world", ()) or ():
            if not isinstance(cand, Mapping):
                continue
            vid = int(cand.get("actor_id", -1))
            if vid in graph_vehicle_ids:
                continue
            center = _draw_pixel_box(ax, cand, bg, matrix, edge=BEV_VEHICLE_COLOR, face=BEV_VEHICLE_COLOR, lw=1.0, alpha=0.65,
                                     linestyle=(0, (3, 2)), zorder=2)
            if center is not None:
                _bev_label(ax, center[0], center[1], f"C{vid}", color=BEV_VEHICLE_COLOR, fontsize=5.5, zorder=3)

    for obj in extra.get("graph_object_world", ()) or ():
        if not isinstance(obj, Mapping):
            continue
        oid = int(obj.get("actor_id", -1))
        graph_obj = next((o for o in record.get("objects", ()) if int(o.get("node_id", -2)) == oid), {})
        obj_hex = _object_birdeye_hex(str(obj.get("object_class", "vehicle")),
                                      notable=bool(graph_obj.get("notable", False)))
        style = BEV_VISIBLE_STYLE if bool(graph_obj.get("visible", False)) else BEV_COLLAB_ONLY_STYLE
        center = _draw_pixel_box(ax, obj, bg, matrix, edge=obj_hex, face=obj_hex, lw=2.0, linestyle=style, zorder=5)
        if center is not None:
            _bev_label(ax, center[0], center[1], f"O{oid}", color=obj_hex, fontsize=6.0, zorder=6)

    ego = extra.get("ego_world")
    if isinstance(ego, Mapping):
        center = _draw_pixel_box(ax, ego, bg, matrix, edge=BEV_EGO_COLOR, face=BEV_EGO_COLOR, lw=2.2, zorder=7)
        if center is not None:
            _bev_label(ax, center[0], center[1], "EGO", color=BEV_EGO_COLOR, fontsize=8.0, weight="bold", va="bottom", zorder=9)

    for cand in extra.get("candidate_world", ()) or ():
        if not isinstance(cand, Mapping):
            continue
        vid = int(cand.get("actor_id", -1))
        if vid not in graph_vehicle_ids:
            continue
        center = _draw_pixel_box(ax, cand, bg, matrix, edge=BEV_VEHICLE_COLOR, face=BEV_VEHICLE_COLOR, lw=2.0, zorder=7)
        if center is not None:
            _bev_label(ax, center[0], center[1], collab_label.get(vid, f"V{vid}"),
                       color=BEV_VEHICLE_COLOR, fontsize=8.0, weight="bold", va="bottom", zorder=9)
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    return True


def _draw_generated_map_background(ax, ctx: Mapping[str, Any]) -> bool:
    bg = ctx.get("map_background")
    if not isinstance(bg, Mapping) or str(ctx.get("frame", "world")) != "world":
        return False
    path = bg.get("path")
    if not path:
        return False
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return False
        rgb = img[:, :, ::-1]
        ppm = float(bg.get("pixels_per_meter", 1.0)) * float(bg.get("scale", 1.0))
        ox, oy = bg.get("world_offset", (0.0, 0.0))
        h, w = rgb.shape[:2]
        xmin = float(ox)
        xmax = xmin + float(w) / max(ppm, 1e-6)
        ymin = float(oy)
        ymax = ymin + float(h) / max(ppm, 1e-6)
        ax.imshow(rgb, extent=[xmin, xmax, ymin, ymax], origin="lower", zorder=0)
        return True
    except Exception:
        return False


def _bev_label(ax, x: float, y: float, text: str, *, color: str, fontsize: float, weight: str = "normal",
               va: str = "center", ha: str = "center", zorder: int = 9) -> None:
    """A BEV label legible on birdeye backgrounds: fill-coloured text with a black halo."""
    txt = ax.text(x, y, text, fontsize=fontsize, weight=weight, ha=ha, va=va, color=color,
                  zorder=zorder, clip_on=True)  # clip with the axes so off-view labels vanish too
    txt.set_path_effects(_LABEL_HALO)


def _oriented_box_pixels(o: Mapping[str, Any], proj) -> List[Tuple[float, float]]:
    """Corner pixels of an object's footprint box (ego-frame length/width/heading -> birdeye pixels)."""
    fwd0, right0 = float(o.get("x", 0.0)), float(o.get("y", 0.0))
    cyaw, syaw = float(o.get("heading_cos", 1.0)), float(o.get("heading_sin", 0.0))
    length = float(o.get("length", 0.0)) or _DEFAULT_OBJ_BOX_M[0]
    width = float(o.get("width", 0.0)) or _DEFAULT_OBJ_BOX_M[1]
    hl, hw = length / 2.0, width / 2.0
    fwd_unit = (cyaw, syaw)        # heading forward in (forward, right)
    right_unit = (-syaw, cyaw)     # perpendicular (left/right) in (forward, right)
    corners = []
    for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        cf = fwd0 + a * hl * fwd_unit[0] + b * hw * right_unit[0]
        cr = right0 + a * hl * fwd_unit[1] + b * hw * right_unit[1]
        corners.append(proj(cf, cr))
    return corners


def _overlay_graph_nodes(ax, record: Mapping[str, Any], *, cx: float, cy: float, ppm: float) -> List[Tuple[float, float]]:
    """Overlay this policy graph's vehicles/objects onto a birdeye image (pixel coords).

    Only the graph's nodes are drawn, so the overlay is exactly the policy-conditioned, visibility-
    filtered set: ego-visible objects + objects seen by the *cooperating* (selected) collaborators.
    Objects only a non-cooperating vehicle sees are not in the graph, so they are not drawn.

    Encoding (chosen so it reads clearly on the colourful birdeye):
      * vehicle-class objects -> an oriented **rectangle** matching the birdeye car box (no extra circle);
        other classes (pedestrian/bicycle) -> a small triangle marker.
      * outline colour = ego-notability (red = notable, blue = not); line style = visibility
        (solid = ego can see it, dashed = only a collaborator sees it).
      * ego = bright cyan ``*`` star; collaborators = purple diamonds (``V1/V2``).
      * every label is coloured text with a white halo, drawn at the node (no pixel offset, so it can't
        drift off its shape).
    """
    vehicles = list(record["vehicles"])
    objects = [o for o in record["objects"] if o.get("valid", True)]
    collab_label: Dict[int, str] = {}
    for n, v in enumerate(sorted((v for v in vehicles if not v["is_ego"]), key=lambda v: v["slot"]), start=1):
        collab_label[v["idx"]] = f"V{n}"

    def proj(fwd: float, right: float) -> Tuple[float, float]:
        return cx + right * ppm, cy - fwd * ppm  # forward → up, right → +x

    pts: List[Tuple[float, float]] = []
    for o in objects:
        px, py = proj(float(o.get("x", 0.0)), float(o.get("y", 0.0)))
        pts.append((px, py))
        notable = bool(o.get("notable", False))
        ego_seen = bool(o.get("visible", False))
        obj_hex = _object_birdeye_hex(str(o.get("object_class", "vehicle")), notable=notable)
        style = BEV_VISIBLE_STYLE if ego_seen else BEV_COLLAB_ONLY_STYLE
        if o.get("object_class", "vehicle") == "vehicle":
            corners = _oriented_box_pixels(o, proj)
            ax.add_patch(Polygon(corners, closed=True, facecolor=obj_hex, edgecolor=obj_hex,
                                 linewidth=2.0, linestyle=style, zorder=5, clip_on=True, alpha=0.85))
        else:
            ax.scatter([px], [py], s=70, marker="^", facecolors=obj_hex, edgecolors=obj_hex,
                       linewidths=2.0, linestyle=style, zorder=5, clip_on=True, alpha=0.85)
        _bev_label(ax, px, py, f"O{o['node_id']}", color=obj_hex, fontsize=6.0, zorder=6)

    for v in vehicles:
        px, py = proj(float(v.get("x", 0.0)), float(v.get("y", 0.0)))
        pts.append((px, py))
        is_ego = bool(v["is_ego"])
        veh_hex = BEV_EGO_COLOR if is_ego else BEV_VEHICLE_COLOR
        label = "EGO" if is_ego else collab_label.get(v["idx"], f"V{v['node_id']}")
        corners = _oriented_box_pixels(v, proj)
        if len(corners) >= 3:
            ax.add_patch(Polygon(corners, closed=True, facecolor=veh_hex, edgecolor=veh_hex,
                                 linewidth=2.0, zorder=7, clip_on=True, alpha=0.85))
        _bev_label(ax, px, py + (12 if is_ego else 11), label, color=veh_hex, fontsize=8.0, weight="bold",
                   va="bottom", zorder=8)
    return pts


def _draw_bev_on_ax(ax, record: Mapping[str, Any], *, bev: Optional[BevOptions] = None) -> None:
    """BEV panel: the real CARLA birdeye with the policy graph overlaid; scatter fallback otherwise."""
    mode = str(getattr(bev, "mode", "generated") if bev is not None else "generated").lower()
    if mode == "scatter":
        _draw_bev_scatter(ax, record)
        return
    bpath = _resolve_birdeye_path(record, bev.birdeye_dir if bev else None)
    if bpath is not None and mode in {"auto", "birdeye-dir"}:
        import cv2

        img = cv2.imread(str(bpath))  # BGR
        if img is not None:
            rgb = img[:, :, ::-1]
            h, w = rgb.shape[:2]
            obs_range = float(bev.obs_range) if bev else 64.0
            ego_offset = float(bev.ego_offset) if bev else 12.0
            ppm = float(w) / max(obs_range, 1e-6)
            cx, cy = w / 2.0, h / 2.0 + (obs_range / 2.0 - ego_offset) * ppm
            ax.imshow(rgb, extent=[0, w, h, 0], zorder=0)
            _overlay_graph_nodes(ax, record, cx=cx, cy=cy, ppm=ppm)
            # FIXED limits = the birdeye image bounds, identical every frame, so the BEV never grows
            # or shrinks with the object positions. Nodes outside the view are clipped (clip_on=True).
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # image y points down
            ax.set_aspect("equal")
            ax.axis("off")
            return
    if mode == "birdeye-dir":
        _draw_bev_scatter(ax, record)
        return
    _draw_generated_bev_on_ax(ax, record, bev=bev)


def build_episode_uncertainty_series(
    records: Sequence[Mapping[str, Any]], *, sigma_scale: float = 4.0, alpha: float = 0.5,
) -> Dict[int, List[Dict[str, float]]]:
    """Per-episode ``{episode: [{step, motion, coverage, total, *_norm}, ...]}`` from ``extra.uncertainty``.

    The uncertainty breakdown is the ego's runtime value for the active policy, so the first record
    seen per ``(episode, step)`` wins. Episodes whose records carry no ``uncertainty`` are absent. The
    ``_norm`` ([0, 1]) values are taken from the record if present, else derived from the raw values via
    the same saturation as the evaluator: ``motion_norm = 1 - exp(-motion / sigma_scale)`` and
    ``total_norm = alpha * motion_norm + (1 - alpha) * clamp(coverage, 0, 1)``.
    """
    tau = max(float(sigma_scale), 1e-6)
    a = float(min(max(alpha, 0.0), 1.0))
    by_ep: Dict[int, Dict[int, Dict[str, float]]] = {}
    for rec in records:
        extra = rec.get("extra") or {}
        unc = extra.get("uncertainty") if isinstance(extra, Mapping) else None
        if not isinstance(unc, Mapping):
            continue
        ep = int(extra.get("episode", 0))
        step = int(rec.get("step", 0))
        per_step = by_ep.setdefault(ep, {})
        if step in per_step:
            continue
        motion = float(unc.get("motion_uncertainty", 0.0))
        coverage = float(unc.get("coverage_uncertainty", 0.0))
        total = float(unc.get("total_uncertainty", motion + coverage))
        cov01 = min(max(coverage, 0.0), 1.0)
        motion_norm = float(unc.get("motion_uncertainty_norm", 1.0 - math.exp(-max(motion, 0.0) / tau)))
        total_norm = float(unc.get("total_uncertainty_norm", a * motion_norm + (1.0 - a) * cov01))
        per_step[step] = {
            "step": step, "motion": motion, "coverage": coverage, "total": total,
            "motion_norm": motion_norm, "coverage_norm": cov01, "total_norm": total_norm,
        }
    return {ep: [per_step[s] for s in sorted(per_step)] for ep, per_step in by_ep.items()}


def _draw_uncertainty_panel(
    ax, series: Sequence[Mapping[str, float]], cur_step: Optional[int] = None, *, mode: str = "raw",
) -> None:
    """Plot motion / coverage / total uncertainty vs step with a marker at ``cur_step``.

    ``mode="norm"`` plots the [0, 1]-saturated ``*_norm`` values (coverage is already in [0, 1]).
    """
    if not series:
        ax.axis("off")
        return
    norm = str(mode) == "norm"
    keys = ("motion_norm", "coverage_norm", "total_norm") if norm else ("motion", "coverage", "total")
    colors = (UNC_MOTION_COLOR, UNC_COVERAGE_COLOR, UNC_TOTAL_COLOR)
    steps = [d["step"] for d in series]
    for key, color, lw, label in zip(keys, colors, (1.4, 1.4, 1.8), ("motion", "coverage", "total")):
        ax.plot(steps, [d[key] for d in series], color=color, lw=lw, label=label)
    cur = None
    if cur_step is not None:
        ax.axvline(float(cur_step), color="#888888", ls="--", lw=1.0)
        cur = next((d for d in series if int(d["step"]) == int(cur_step)), None)
        if cur is not None:
            for key, color in zip(keys, colors):
                ax.plot([cur_step], [cur[key]], marker="o", ms=4.5, color=color)
    if cur is not None:
        ax.set_title(
            f"uncertainty{' (norm)' if norm else ''} @ step {int(cur_step)}:   "
            f"motion={cur[keys[0]]:.3f}    coverage={cur[keys[1]]:.3f}    total={cur[keys[2]]:.3f}",
            fontsize=8,
        )
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("uncertainty (norm)" if norm else "uncertainty", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0) if norm else ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.6)


def render_graph_matplotlib(
    record: Mapping[str, Any], *, with_bev: bool = True, bev: Optional[BevOptions] = None,
    figsize: Optional[Tuple[float, float]] = None,
    unc_series: Optional[Sequence[Mapping[str, float]]] = None,
    cur_step: Optional[int] = None,
    unc_mode: str = "raw",
):
    """Render a record at a FIXED size: topology (left) + (optional) BEV (right).

    The size is constant across frames (independent of object count) so the HTML slider does not
    jitter. The BEV uses the real CARLA birdeye + policy overlay when ``bev`` resolves an image for
    this ego/step, otherwise a labelled ego-centric scatter. When ``unc_series`` (the episode's
    motion/coverage/total uncertainty series) is given, a bottom panel plots it vs step with a marker
    at ``cur_step``; without it the layout is unchanged (no extra axis).
    """
    has_unc = bool(unc_series)
    if not with_bev:
        if has_unc:
            fig = plt.figure(figsize=figsize or (TOPO_WIDTH, FIG_HEIGHT + UNC_HEIGHT))
            gs = fig.add_gridspec(2, 1, height_ratios=[FIG_HEIGHT, UNC_HEIGHT], hspace=0.4)
            ax = fig.add_subplot(gs[0, 0])
            ax_unc = fig.add_subplot(gs[1, 0])
            _draw_on_ax(ax, record)
            _draw_uncertainty_panel(ax_unc, unc_series, cur_step, mode=unc_mode)
            fig.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.10)
            return fig
        fig, ax = plt.subplots(figsize=figsize or (TOPO_WIDTH, FIG_HEIGHT))
        _draw_on_ax(ax, record)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
        return fig
    if has_unc:
        fig = plt.figure(figsize=figsize or (TOPO_WIDTH + BEV_WIDTH, FIG_HEIGHT + UNC_HEIGHT))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[FIG_HEIGHT, UNC_HEIGHT], width_ratios=[TOPO_WIDTH, BEV_WIDTH],
            hspace=0.4, wspace=0.08,
        )
        ax_topo = fig.add_subplot(gs[0, 0])
        ax_bev = fig.add_subplot(gs[0, 1])
        ax_unc = fig.add_subplot(gs[1, :])
        _draw_on_ax(ax_topo, record)
        _draw_bev_on_ax(ax_bev, record, bev=bev)
        _draw_uncertainty_panel(ax_unc, unc_series, cur_step, mode=unc_mode)
        fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.09)
        return fig
    fig, (ax_topo, ax_bev) = plt.subplots(
        1, 2, figsize=figsize or (TOPO_WIDTH + BEV_WIDTH, FIG_HEIGHT),
        gridspec_kw={"width_ratios": [TOPO_WIDTH, BEV_WIDTH]},
    )
    _draw_on_ax(ax_topo, record)
    _draw_bev_on_ax(ax_bev, record, bev=bev)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06, wspace=0.08)
    return fig


# =====================================================================
# 4) Frames + writers (PNG / GIF / interactive HTML)
# =====================================================================


def group_records_to_frames(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Group a flat record list into ordered per-step frames ``{episode, step, panels{label: record}}``.

    Frames are keyed by ``(episode, step)`` -- the episode comes from ``record.extra.episode``
    (default 0) -- so records from different episodes that share a step number do NOT collide into
    the same frame. Each frame's ``panels`` map holds one record per policy label.
    """
    by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for rec in records:
        step = int(rec["step"])
        episode = int(rec.get("extra", {}).get("episode", 0)) if isinstance(rec.get("extra"), dict) else 0
        frame = by_key.setdefault((episode, step), {"episode": episode, "step": step, "panels": {}})
        label = str(rec.get("policy_label", "policy"))
        unique = label
        k = 2
        while unique in frame["panels"]:
            unique = f"{label}#{k}"
            k += 1
        frame["panels"][unique] = rec
    return [by_key[key] for key in sorted(by_key)]


def _fig_to_png_bytes(fig, *, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    # No bbox_inches="tight": that crops to content and makes the output size vary per frame.
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def _policy_png_list(
    frame: Mapping[str, Any], *, dpi: int = 110, bev: Optional[BevOptions] = None,
    unc_series: Optional[Sequence[Mapping[str, float]]] = None, unc_mode: str = "raw",
) -> List[bytes]:
    """Render each policy of a frame to its own fixed-size PNG (independent images)."""
    out: List[bytes] = []
    for label in frame["panels"]:
        fig = render_graph_matplotlib(
            frame["panels"][label], bev=bev, unc_series=unc_series, cur_step=frame.get("step"),
            unc_mode=unc_mode,
        )
        out.append(_fig_to_png_bytes(fig, dpi=dpi))
    return out


def _stack_vertical(images: List) -> "Any":
    """Vertically concatenate PIL images (width-padded, centered) into one tall image."""
    from PIL import Image

    width = max(im.width for im in images)
    height = sum(im.height for im in images)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, ((width - im.width) // 2, y))
        y += im.height
    return canvas


def _frame_png_bytes(
    frame: Mapping[str, Any], *, dpi: int = 110, bev: Optional[BevOptions] = None,
    unc_series: Optional[Sequence[Mapping[str, float]]] = None, unc_mode: str = "raw",
) -> bytes:
    """Render a frame as ONE image: each policy its own figure, stacked vertically (for PNG/GIF)."""
    from PIL import Image

    pngs = _policy_png_list(frame, dpi=dpi, bev=bev, unc_series=unc_series, unc_mode=unc_mode)
    images = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    composite = images[0] if len(images) == 1 else _stack_vertical(images)
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


def _bev_with_context(records: Sequence[Mapping[str, Any]], bev: Optional[BevOptions]) -> BevOptions:
    opts = bev or BevOptions()
    if opts.contexts is not None or str(getattr(opts, "mode", "generated")).lower() in {"scatter", "birdeye-dir"}:
        return opts
    return BevOptions(
        birdeye_dir=opts.birdeye_dir,
        obs_range=opts.obs_range,
        ego_offset=opts.ego_offset,
        mode=opts.mode,
        frame=opts.frame,
        margin_m=opts.margin_m,
        show_candidates=opts.show_candidates,
        contexts=build_episode_bev_contexts(records, opts),
    )


def write_graph_frames_png(
    records: Sequence[Mapping[str, Any]], out_dir: PathLike, *, dpi: int = 110, bev: Optional[BevOptions] = None,
    show_uncertainty: bool = True, unc_mode: str = "raw", unc_sigma_scale: float = 4.0, unc_alpha: float = 0.5,
) -> List[Path]:
    """Write one PNG per time-step frame; returns the written paths."""
    bev = _bev_with_context(records, bev)
    unc_by_ep = build_episode_uncertainty_series(records, sigma_scale=unc_sigma_scale, alpha=unc_alpha) if show_uncertainty else {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, frame in enumerate(group_records_to_frames(records)):
        path = out_dir / f"frame_{i:04d}_step{frame['step']:04d}.png"
        path.write_bytes(_frame_png_bytes(frame, dpi=dpi, bev=bev, unc_series=unc_by_ep.get(frame["episode"]), unc_mode=unc_mode))
        paths.append(path)
    return paths


def write_graph_timeline_gif(
    records: Sequence[Mapping[str, Any]], out_gif: PathLike, *, fps: float = 2.0, dpi: int = 110,
    bev: Optional[BevOptions] = None, show_uncertainty: bool = True, unc_mode: str = "raw",
    unc_sigma_scale: float = 4.0, unc_alpha: float = 0.5,
) -> Path:
    """Render frames and assemble an animated GIF (via PIL)."""
    from PIL import Image

    bev = _bev_with_context(records, bev)
    unc_by_ep = build_episode_uncertainty_series(records, sigma_scale=unc_sigma_scale, alpha=unc_alpha) if show_uncertainty else {}
    frames = group_records_to_frames(records)
    images: List["Image.Image"] = []
    for frame in frames:
        png = _frame_png_bytes(frame, dpi=dpi, bev=bev, unc_series=unc_by_ep.get(frame["episode"]), unc_mode=unc_mode)
        images.append(Image.open(io.BytesIO(png)).convert("RGB"))
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        raise ValueError("no records to render into a GIF")
    duration_ms = int(1000.0 / max(float(fps), 1e-3))
    images[0].save(out_gif, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return out_gif


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; color: #222; }}
 #caption {{ margin: 8px 0; font-weight: 600; }}
 .controls {{ display: flex; align-items: center; gap: 12px; margin: 10px 0; }}
 #slider {{ flex: 1; }}
 /* each policy is its OWN independent image; stacked one per row */
 #panels {{ display: flex; flex-direction: column; gap: 10px; }}
 #panels img {{ max-width: 100%; border: 1px solid #ddd; }}
</style></head>
<body>
 <h2>{title}</h2>
 <div id="caption"></div>
 <div class="controls">
   <button id="play">▶ play</button>
   <input id="slider" type="range" min="0" max="{maxidx}" value="0" step="1">
   <span id="idx">0 / {maxidx}</span>
 </div>
 <div id="panels"></div>
 <script>
  const FRAMES = {frames_json};      // FRAMES[i] = array of per-policy base64 PNGs
  const CAPTIONS = {captions_json};
  const panels = document.getElementById('panels');
  const slider = document.getElementById('slider');
  const idx = document.getElementById('idx');
  const cap = document.getElementById('caption');
  let timer = null;
  function show(i) {{
    panels.innerHTML = '';
    for (const b64 of FRAMES[i]) {{
      const im = document.createElement('img');
      im.src = 'data:image/png;base64,' + b64;
      panels.appendChild(im);
    }}
    cap.textContent = CAPTIONS[i];
    idx.textContent = i + ' / ' + (FRAMES.length - 1);
    slider.value = i;
  }}
  slider.addEventListener('input', e => show(parseInt(e.target.value)));
  document.getElementById('play').addEventListener('click', function() {{
    if (timer) {{ clearInterval(timer); timer = null; this.textContent = '▶ play'; return; }}
    this.textContent = '⏸ pause';
    timer = setInterval(() => {{
      let i = (parseInt(slider.value) + 1) % FRAMES.length;
      show(i);
      if (i === FRAMES.length - 1) {{ clearInterval(timer); timer = null; document.getElementById('play').textContent = '▶ play'; }}
    }}, {interval_ms});
  }});
  show(0);
 </script>
</body></html>
"""


def write_graph_timeline_html(
    records: Sequence[Mapping[str, Any]],
    out_html: PathLike,
    *,
    title: str = "WAM cooperative graph timeline",
    fps: float = 2.0,
    dpi: int = 110,
    bev: Optional[BevOptions] = None,
    show_uncertainty: bool = True,
    unc_mode: str = "raw",
    unc_sigma_scale: float = 4.0,
    unc_alpha: float = 0.5,
) -> Path:
    """Write a self-contained interactive HTML: a time-step slider over per-step frames.

    Each frame embeds its policies as **independent images** (one per policy, stacked one per row),
    not a single composite image, so policy panels stay separate and fixed-size.
    """
    bev = _bev_with_context(records, bev)
    unc_by_ep = build_episode_uncertainty_series(records, sigma_scale=unc_sigma_scale, alpha=unc_alpha) if show_uncertainty else {}
    frames = group_records_to_frames(records)
    if not frames:
        raise ValueError("no records to render into HTML")
    frame_images: List[List[str]] = []
    captions: List[str] = []
    for frame in frames:
        pngs = _policy_png_list(frame, dpi=dpi, bev=bev, unc_series=unc_by_ep.get(frame["episode"]), unc_mode=unc_mode)
        frame_images.append([base64.b64encode(p).decode("ascii") for p in pngs])
        panels = frame["panels"]
        policy_text = " | ".join(
            f"{label}: {'V2V' if rec.get('is_v2v') else 'local'} "
            f"(veh {rec['counts']['vehicles']}, obs {rec['counts']['observations']}, obj {rec['counts']['objects']})"
            for label, rec in panels.items()
        )
        prefix = f"ep {frame['episode']} " if frame.get("episode") else ""
        captions.append(f"{prefix}step {frame['step']} — {policy_text}")
    html = _HTML_TEMPLATE.format(
        title=title,
        maxidx=len(frames) - 1,
        frames_json=json.dumps(frame_images),
        captions_json=json.dumps(captions),
        interval_ms=int(1000.0 / max(float(fps), 1e-3)),
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


# =====================================================================
# 5) JSONL persistence
# =====================================================================


def append_record_jsonl(record: Mapping[str, Any], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_records_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
