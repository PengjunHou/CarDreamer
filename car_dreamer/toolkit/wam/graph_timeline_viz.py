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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .graph import MODALITIES, OBJECT, OBS_OBJ, OBSERVATION, VEH_OBS, VEH_VEH, VEHICLE

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

# --- fixed figure geometry (constant across frames so the slider doesn't jitter) ---
FIG_HEIGHT = 5.6
TOPO_WIDTH = 8.0
BEV_WIDTH = 5.6
BEV_FALLBACK_RANGE_M = 50.0  # scatter-BEV half-range (m) when no birdeye image is available


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
    obj_feat = _as_list(graph[OBJECT], "x")  # ego-frame state: [x, y, z, ...]
    objects = []
    for i in range(len(obj_node_id)):
        feat = obj_feat[i] if i < len(obj_feat) else []
        objects.append(
            {
                "idx": i,
                "node_id": int(obj_node_id[i]),
                "valid": bool(obj_mask[i] >= 0.5),
                "notable": bool(obj_notable[i] >= 0.5),
                "visible": bool(obj_visible[i] >= 0.5),
                "invisible": bool(obj_invisible[i] >= 0.5),
                "x": float(feat[0]) if len(feat) > 1 else 0.0,  # ego-frame forward
                "y": float(feat[1]) if len(feat) > 1 else 0.0,  # ego-frame right
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
        ax.set_title("BEV (no data)", fontsize=9)
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
    ax.set_title("BEV (ego-frame, forward ↑)", fontsize=9)


def _overlay_graph_nodes(ax, record: Mapping[str, Any], *, cx: float, cy: float, ppm: float) -> List[Tuple[float, float]]:
    """Overlay this policy graph's vehicles/objects onto a birdeye image (pixel coords).

    Only the graph's nodes are drawn, so the overlay is exactly the policy-conditioned, visibility-
    filtered set: ego-visible objects + objects seen by the *cooperating* (selected) collaborators.
    Objects only a non-cooperating vehicle sees are not in the graph, so they are not drawn. Object
    fill = red(notable)/gray; **black edge = ego sees it, white edge = only a collaborator sees it**.
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
        fill = COLOR_OBJECT if o["notable"] else COLOR_OBJECT_UNIMPORTANT
        ego_seen = bool(o.get("visible", False))
        edge = "black" if ego_seen else "white"  # white edge = collaborator-only (ego can't see it)
        ax.scatter([px], [py], s=80, c=fill, edgecolors=edge, linewidths=1.6, zorder=5)
        ax.text(px, py - 7, f"O{o['node_id']}", fontsize=5.5, ha="center", va="top", color="white",
                zorder=6, bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))

    for v in vehicles:
        px, py = proj(float(v.get("x", 0.0)), float(v.get("y", 0.0)))
        pts.append((px, py))
        is_ego = bool(v["is_ego"])
        label = "ego" if is_ego else collab_label.get(v["idx"], f"V{v['node_id']}")
        ax.scatter([px], [py], s=120, marker="s", facecolors="none",
                   edgecolors=COLOR_EGO if is_ego else COLOR_COLLAB, linewidths=2.2, zorder=7)
        ax.text(px, py + 8, label, fontsize=7, weight="bold", ha="center", va="bottom",
                color=COLOR_EGO if is_ego else "#6A0DAD", zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    return pts


def _draw_bev_on_ax(ax, record: Mapping[str, Any], *, bev: Optional[BevOptions] = None) -> None:
    """BEV panel: the real CARLA birdeye with the policy graph overlaid; scatter fallback otherwise."""
    bpath = _resolve_birdeye_path(record, bev.birdeye_dir if bev else None)
    if bpath is not None:
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
            pts = _overlay_graph_nodes(ax, record, cx=cx, cy=cy, ppm=ppm)
            # expand limits so objects beyond the birdeye view are still shown (issue: clipped objects)
            xs = [0, w] + [p[0] for p in pts]
            ys = [0, h] + [p[1] for p in pts]
            m = 12.0
            ax.set_xlim(min(xs) - m, max(xs) + m)
            ax.set_ylim(max(ys) + m, min(ys) - m)  # image y points down
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title("BEV (birdeye + policy overlay; ◻ego/collab, ●obj red=notable, white-edge=collab-only)",
                         fontsize=7.0)
            return
    _draw_bev_scatter(ax, record)


def render_graph_matplotlib(
    record: Mapping[str, Any], *, with_bev: bool = True, bev: Optional[BevOptions] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Render a record at a FIXED size: topology (left) + (optional) BEV (right).

    The size is constant across frames (independent of object count) so the HTML slider does not
    jitter. The BEV uses the real CARLA birdeye + policy overlay when ``bev`` resolves an image for
    this ego/step, otherwise a labelled ego-centric scatter.
    """
    if not with_bev:
        fig, ax = plt.subplots(figsize=figsize or (TOPO_WIDTH, FIG_HEIGHT))
        _draw_on_ax(ax, record)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
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
    the same frame. Each frame's ``panels`` map holds one record per policy label (mode-A: a single
    panel; mode-B counterfactual: one panel per fixed policy).
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


def _policy_png_list(frame: Mapping[str, Any], *, dpi: int = 110, bev: Optional[BevOptions] = None) -> List[bytes]:
    """Render each policy of a frame to its own fixed-size PNG (independent images)."""
    out: List[bytes] = []
    for label in frame["panels"]:
        fig = render_graph_matplotlib(frame["panels"][label], bev=bev)
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


def _frame_png_bytes(frame: Mapping[str, Any], *, dpi: int = 110, bev: Optional[BevOptions] = None) -> bytes:
    """Render a frame as ONE image: each policy its own figure, stacked vertically (for PNG/GIF)."""
    from PIL import Image

    pngs = _policy_png_list(frame, dpi=dpi, bev=bev)
    images = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    composite = images[0] if len(images) == 1 else _stack_vertical(images)
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


def write_graph_frames_png(
    records: Sequence[Mapping[str, Any]], out_dir: PathLike, *, dpi: int = 110, bev: Optional[BevOptions] = None
) -> List[Path]:
    """Write one PNG per time-step frame; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, frame in enumerate(group_records_to_frames(records)):
        path = out_dir / f"frame_{i:04d}_step{frame['step']:04d}.png"
        path.write_bytes(_frame_png_bytes(frame, dpi=dpi, bev=bev))
        paths.append(path)
    return paths


def write_graph_timeline_gif(
    records: Sequence[Mapping[str, Any]], out_gif: PathLike, *, fps: float = 2.0, dpi: int = 110,
    bev: Optional[BevOptions] = None,
) -> Path:
    """Render frames and assemble an animated GIF (via PIL)."""
    from PIL import Image

    frames = group_records_to_frames(records)
    images: List["Image.Image"] = []
    for frame in frames:
        images.append(Image.open(io.BytesIO(_frame_png_bytes(frame, dpi=dpi, bev=bev))).convert("RGB"))
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
) -> Path:
    """Write a self-contained interactive HTML: a time-step slider over per-step frames.

    Each frame embeds its policies as **independent images** (one per policy, stacked one per row),
    not a single composite image, so mode-B policy panels stay separate and fixed-size.
    """
    frames = group_records_to_frames(records)
    if not frames:
        raise ValueError("no records to render into HTML")
    frame_images: List[List[str]] = []
    captions: List[str] = []
    for frame in frames:
        pngs = _policy_png_list(frame, dpi=dpi, bev=bev)
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
