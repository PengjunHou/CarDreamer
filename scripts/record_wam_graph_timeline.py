#!/usr/bin/env python3
"""Record the per-step WAM global cooperative graph of a live run to a JSONL timeline.

Mirrors ``scripts/run_env.py`` but, after every env step, extracts the structural record of the
ego's cooperative graph (``env._wam_graph``) plus the currently active Base-Station policy and
appends it to a JSONL file. Render it offline with ``scripts/visualize_wam_graph_timeline.py``.

This is the **mode-A (live timeline)** recorder: the active policy evolves over its lifetime ``Td``,
so the recorded graph naturally shows local <-> V2V switching across time. Needs a running CARLA.

Examples
--------
    python scripts/record_wam_graph_timeline.py --steps 300 --out outputs/graph_timeline.jsonl
    python scripts/visualize_wam_graph_timeline.py --jsonl outputs/graph_timeline.jsonl \
        --html outputs/graph_timeline.html --gif outputs/graph_timeline.gif
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_carla_pythonapi() -> None:
    carla_root = os.environ.get("CARLA_ROOT", "/home/peh324/carla_simulator")
    for sub in ("PythonAPI", "PythonAPI/carla"):
        path = os.path.join(carla_root, sub)
        if path not in sys.path:
            sys.path.append(path)


def _enable_wide_bev(config):
    """Append the wide ego-centered birdeye (birdeye_wam100) to the ego observation (record-only).

    Enabled last so its dump wins (data/birdeye_frames/vehicle_<id>/birdeye_<step>.png becomes the
    100 m view). Does not touch the model's birdeye_wpt input. Render with the viz flags
    ``--bev-obs-range 100 --bev-ego-offset 50``.

    Config is immutable, so this returns the updated config instead of mutating in place.
    """
    enabled = list(config.env.observation.enabled)
    if "birdeye_wam100" not in enabled:
        config = config.update({"env.observation.enabled": enabled + ["birdeye_wam100"]})
    return config


def build_env(task: str, env_args: List[str], wide_bev: bool = False):
    import gymnasium as gym

    import car_dreamer
    from car_dreamer import toolkit

    config = car_dreamer.load_task_configs(task)
    config, _ = toolkit.Flags(config).parse_known(env_args)
    if wide_bev:
        config = _enable_wide_bev(config)
    # This script's viz pipeline reads the per-step birdeye dumps, so recording must be on.
    config = config.update({"env.observation.dump_frames": True})
    env = gym.make(config.env.name, config=config.env)
    return env, config


def _write_map_background(sim, out_path: Path, *, pixels_per_meter: float) -> Optional[Dict[str, object]]:
    """Write one global CARLA map surface for offline generated-BEV rendering."""
    try:
        import cv2

        from car_dreamer.toolkit.observer.handlers.renderer.map_renderer import MapRenderer

        world_manager = getattr(sim, "_world", None)
        carla_world = getattr(world_manager, "_world", None)
        carla_map = getattr(world_manager, "_map", None)
        if carla_world is None or carla_map is None:
            return None
        renderer = MapRenderer(carla_world, carla_map, float(pixels_per_meter))
        surface = getattr(renderer, "_surface", None)
        if surface is None:
            return None
        bg_path = out_path.with_suffix("")
        bg_path = bg_path.parent / f"{bg_path.name}_map.png"
        bg_path.parent.mkdir(parents=True, exist_ok=True)
        # Match BirdeyeHandler's dump path: it writes the renderer surface directly.
        cv2.imwrite(str(bg_path), surface)
        return {
            "path": str(bg_path),
            "pixels_per_meter": float(getattr(renderer, "_pixels_per_meter")),
            "scale": float(getattr(renderer, "_scale", 1.0)),
            "world_offset": [float(v) for v in getattr(renderer, "_world_offset_in_meter")],
            "width_px": int(surface.shape[1]),
            "height_px": int(surface.shape[0]),
        }
    except Exception as exc:
        print(f"warning: could not write map background: {exc}", flush=True)
        return None


def active_policy_label(sim) -> Tuple[str, Optional[int]]:
    """Readable label + id for the currently active Base-Station policy."""
    proc = getattr(sim, "_comm_process", None)
    policy = getattr(proc, "policy", None) if proc is not None else None
    if policy is None:
        return "none", None
    if getattr(policy, "is_local_only", False):
        return "local", int(policy.policy_id)
    members = ",".join(str(v) for v in sorted(policy.selected_collaborators))
    return f"coop[{members}]", int(policy.policy_id)


def _actor_world_record(actor, *, role: str, selected: bool = False) -> Dict[str, object]:
    tf = actor.get_transform()
    bb = actor.bounding_box
    x, y, z = float(tf.location.x), float(tf.location.y), float(tf.location.z)
    yaw = float(tf.rotation.yaw)
    import math

    cos_y, sin_y = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    hl, hw = float(bb.extent.x), float(bb.extent.y)
    bbox = []
    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        bbox.append((x + lx * cos_y - ly * sin_y, y + lx * sin_y + ly * cos_y))
    return {
        "actor_id": int(actor.id),
        "role": str(role),
        "selected_by_policy": bool(selected),
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "length": float(2.0 * bb.extent.x),
        "width": float(2.0 * bb.extent.y),
        "bbox": bbox,
    }


def _object_world_record(state) -> Dict[str, object]:
    return {
        "actor_id": int(state.actor_id),
        "object_class": str(getattr(state, "object_class", "other")),
        "x": float(state.x),
        "y": float(state.y),
        "z": float(state.z),
        "yaw": float(state.yaw),
        "length": float(state.length),
        "width": float(state.width),
        "bbox": [tuple(p) for p in (getattr(state, "bbox", None) or ())],
        "visible_to_ego": bool(getattr(state, "visible_to_ego", False)),
        "visible_to_collaborators": [int(v) for v in getattr(state, "visible_to_collaborators", ())],
    }


def _graph_object_ids(graph) -> set:
    try:
        node_id = graph["object"].node_id
        return {int(v) for v in node_id.tolist() if int(v) >= 0}
    except Exception:
        return set()


def _timeline_world_extra(sim, graph, *, map_background: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    proc = getattr(sim, "_comm_process", None)
    policy = getattr(proc, "policy", None) if proc is not None else None
    selected = set(int(v) for v in getattr(policy, "selected_collaborators", ()))
    out: Dict[str, object] = {}
    ego = getattr(sim, "ego", None)
    if ego is not None:
        out["ego_world"] = _actor_world_record(ego, role="ego", selected=True)
    candidates = []
    for actor in getattr(sim, "group_vehs", []):
        if actor is None:
            continue
        try:
            candidates.append(_actor_world_record(actor, role="candidate", selected=int(actor.id) in selected))
        except RuntimeError:
            pass
    out["candidate_world"] = candidates
    if map_background is not None:
        out["map_background"] = dict(map_background)

    graph_object_ids = _graph_object_ids(graph)
    objects = []
    for state in getattr(sim, "_wam_object_states", []):
        if graph_object_ids and int(state.actor_id) not in graph_object_ids:
            continue
        objects.append(_object_world_record(state))
    out["graph_object_world"] = objects
    return out


def record_step(sim, out_path: Path, *, map_background: Optional[Dict[str, object]] = None) -> bool:
    """Extract + append one step's graph record. Returns True if a graph was recorded."""
    from car_dreamer.toolkit.wam import append_record_jsonl, hetero_graph_to_record

    graph = getattr(sim, "_wam_graph", None)
    if graph is None:
        return False
    label, policy_id = active_policy_label(sim)
    extra = {"episode": int(getattr(sim, "_episode_count", 0) or 0)}
    breakdown = getattr(sim, "_wam_uncertainty_breakdown", None)
    if isinstance(breakdown, dict):
        # Per-step motion / coverage / total uncertainty for the timeline's bottom panel.
        extra["uncertainty"] = {k: float(v) for k, v in breakdown.items() if isinstance(v, (int, float))}
    extra.update(_timeline_world_extra(sim, graph, map_background=map_background))
    record = hetero_graph_to_record(
        graph,
        step=int(getattr(sim, "_time_step", 0)),
        policy_label=label,
        policy_id=policy_id,
        extra=extra,
    )
    append_record_jsonl(record, out_path)
    return True


def _cf_policy_specs(state, tokens):
    """Return ``[(label, WAMPolicy), ...]`` for the requested counterfactual policies."""
    from car_dreamer.toolkit.wam import WAMPolicy

    ego = state["ego"]
    collabs = list(state.get("collaborators", ()))
    specs = []
    for tok in tokens:
        tok = str(tok).strip()
        if tok == "ego_only":
            specs.append(("ego_only", WAMPolicy((), {}, {}, 5, "cf")))
        elif tok in ("nearest_single", "single_nearest"):
            if collabs:
                nearest = min(collabs, key=lambda c: (float(c.x) - float(ego.x)) ** 2 + (float(c.y) - float(ego.y)) ** 2)
                nid = int(nearest.actor_id)
                specs.append((f"single_nearest[{nid}]", WAMPolicy((nid,), {nid: "objlist"}, {nid: 1.0}, 5, "cf")))
        elif tok == "all_candidates":
            ids = tuple(int(c.actor_id) for c in collabs)
            if ids:
                specs.append(("all_candidates", WAMPolicy(ids, {i: "objlist" for i in ids}, {i: 1.0 for i in ids}, 5, "cf")))
    return specs


def _cf_stage2_policy(uwm, sim, state, spec, bev_spec, device, n_candidates=1):
    """Generate a request-vehicle collaboration policy from the trained Stage-2 UWM for this scene.

    Builds the BS condition C^BS from the request vehicle's current perception graph (``sim._wam_graph``,
    matching what Stage-2 trained on; falls back to an ego-only graph) + its notable objects, runs
    ``propose_policies``, and decodes the first candidate into a :class:`WAMPolicy`. Member slots map to
    the scene's collaborator ids (so the proposal can be turned into a real policy-conditioned graph).
    Returns ``None`` if the scene has no collaborators or the graph is unavailable.
    """
    import torch

    from car_dreamer.toolkit.wam import WAMPolicy, build_stage1_policy_graph, decode_chunk_to_wampolicy

    collabs = list(state.get("collaborators", ()))
    candidate_ids = [int(c.actor_id) for c in collabs]
    if not candidate_ids:
        return None
    req_graph = getattr(sim, "_wam_graph", None)
    if req_graph is None:
        req_graph = build_stage1_policy_graph(
            ego=state["ego"], collaborators=tuple(collabs), objects=tuple(state.get("live_states", ())),
            policy=WAMPolicy((), {}, {}, 5, "cf"), spec=spec, notable_ids=state.get("notable_ids", ()),
            latency_by_vehicle={}, bev_spec=bev_spec,
        )
    notable_ids = list(state.get("notable_ids", ()))
    with torch.no_grad():
        cond, tids = uwm.condition_tokens([req_graph.to(device)], 0, notable_ids)
        cond = cond.unsqueeze(0); tids = tids.unsqueeze(0)
        mask = torch.ones(1, cond.shape[1], device=cond.device)
        m_max = int(uwm.flow.config.max_members)
        mm = torch.zeros(1, m_max, device=cond.device)
        mm[0, : min(len(candidate_ids), m_max)] = 1.0
        cand = uwm.flow.propose_policies(cond, tids, mask, n_candidates=int(n_candidates), member_mask=mm)
    return decode_chunk_to_wampolicy(cand[0, 0], candidate_ids, num_formats=int(uwm.flow.config.num_formats))


def _cf_uncertainty(model, window, sim, state, policy, device, debug=False, coverage_override=None):
    """Per-policy motion + coverage uncertainty (+ [0,1] norm) for one counterfactual graph window.

    Emits two motion口径 that share the same predicted ``TrSigma`` and coverage, differing only in the
    per-object weight: ``motion_uncertainty*`` is GT-notable hard-weighted (the original), and
    ``motion_uncertainty_prob*`` is the online ``notable_prob``-weighted one (soft-gate + mass_floor +
    sigma_scale/alpha_norm read from ``sim``), matching the live ``_predict_wam_with_checkpoint`` path.
    """
    import math

    import torch

    from car_dreamer.toolkit.wam import coverage_metrics

    motion = 0.0       # GT-notable hard-weighted (the original口径)
    motion_prob = 0.0  # online notable_prob-weighted (mirrors _predict_wam_with_checkpoint)
    per_object = []    # debug breakdown: per-object notable_prob / TrSigma / w_p / gt_notable
    if window:
        with torch.no_grad():
            out = model([g.to(device) for g in window])
        if int(out["object_node_ids"].numel()) > 0:
            trace = torch.exp(out["traj_log_var"].detach()).sum(dim=-1).mean(dim=-1)  # [Q] TrSigma_o
            qids = [int(v) for v in out["object_node_ids"].detach().cpu().tolist()]
            # GT-notable口径: weight by the t-time GT notable set (``state["notable_ids"]``) aligned to the
            # window-union query ids -- NOT the model's soft notable_prob, and NOT heads' best-effort
            # newest-frame label. An object that was notable but has since left the latest frame lingers in
            # the window union with a stale ``notable=1`` (and a GRU-inflated extrapolated variance), which
            # would keep motion non-zero (even growing) for ``history_window`` steps after it is gone.
            # Weighting by the t-time GT set drops it, so "no notable object at t -> motion 0" -- matching
            # the BEV / topology (latest frame only) and the offline ``perception_labels_at_t`` semantics.
            notable_set = {int(i) for i in (state.get("notable_ids", ()) or ())}
            w_gt = torch.tensor([1.0 if q in notable_set else 0.0 for q in qids], device=trace.device)
            if float(w_gt.sum()) > 1e-6:
                motion = float((w_gt * trace).sum() / w_gt.sum().clamp_min(1e-6))
            # online口径: same formula as the live checkpoint predictor -- soft-gate the model's notable_prob
            # then notable-weighted mean of TrSigma with a denominator mass_floor. gate_k / threshold /
            # mass_floor come from the env (``--env.wam.coverage.*``) so this matches the online run.
            p = out["notable_prob"].detach()
            gate_k = float(getattr(sim, "_wam_uncertainty_notable_gate_k", 8.0))
            gate_thr = float(getattr(sim, "_wam_uncertainty_notable_gate_threshold", 0.5))
            mass_floor = float(getattr(sim, "_wam_uncertainty_notable_mass_floor", 1.0))
            w_p = torch.sigmoid(gate_k * (p - gate_thr)) if gate_k > 0 else p
            motion_prob = float((w_p * trace).sum() / (w_p.sum() + mass_floor).clamp_min(1e-6))
            if debug:
                p_l = [float(x) for x in p.detach().cpu().reshape(-1).tolist()]
                tr_l = [float(x) for x in trace.detach().cpu().reshape(-1).tolist()]
                wp_l = [float(x) for x in w_p.detach().cpu().reshape(-1).tolist()]
                per_object = [
                    {"node_id": qids[i], "gt_notable": int(qids[i] in notable_set),
                     "notable_prob": p_l[i], "trace": tr_l[i], "w_p": wp_l[i]}
                    for i in range(len(qids))
                ]
    coverage = 0.0
    if coverage_override is not None:
        # comm-replay passes the real-latency coverage (built from delivered messages); otherwise fall
        # back to the 0-latency policy coverage from current visibility.
        coverage = float(coverage_override)
    else:
        cov_fn = getattr(sim, "_build_wam_coverage_for_stage1_policy", None)
        if cov_fn is not None:
            raster = cov_fn(state, policy)
            if raster is not None:
                coverage = float(coverage_metrics(raster).get("coverage_uncertainty", 0.0))
    cov01 = min(max(coverage, 0.0), 1.0)
    # GT-notable norm keeps the original fixed tau/alpha; the online口径 uses the env's sigma_scale /
    # alpha_norm so its [0,1] view matches the live ``_update_wam_uncertainty_breakdown``.
    tau_gt, a_gt = 4.0, 0.5
    tau_p = max(float(getattr(sim, "_wam_uncertainty_sigma_scale", 4.0)), 1e-6)
    a_p = float(min(max(getattr(sim, "_wam_uncertainty_alpha_norm", 0.5), 0.0), 1.0))
    motion_norm = 1.0 - math.exp(-max(motion, 0.0) / tau_gt)
    motion_prob_norm = 1.0 - math.exp(-max(motion_prob, 0.0) / tau_p)
    result = {
        "motion_uncertainty": motion,
        "coverage_uncertainty": coverage,
        "total_uncertainty": motion + coverage,
        "motion_uncertainty_norm": motion_norm,
        "total_uncertainty_norm": a_gt * motion_norm + (1.0 - a_gt) * cov01,
        # online notable_prob口径 (same coverage; differs only in the per-object motion weight)
        "motion_uncertainty_prob": motion_prob,
        "total_uncertainty_prob": motion_prob + coverage,
        "motion_uncertainty_prob_norm": motion_prob_norm,
        "total_uncertainty_prob_norm": a_p * motion_prob_norm + (1.0 - a_p) * cov01,
    }
    if debug:
        # per-object breakdown so A (mis-classified -> high notable_prob) vs B (low prob + large TrSigma,
        # passed by mass_floor) can be told apart offline from the JSONL.
        result["per_object"] = per_object
    return result


def record_counterfactual_step(sim, model, windows, out_path, *, tokens, history_window, device,
                               map_background=None, debug=False, uwm=None, uwm_candidates=1):
    """One rollout step: build + record a per-policy graph and uncertainty for each counterfactual policy.

    When ``uwm`` (a trained Stage-2 model) is given, also generates a ``stage2`` policy from it and scores
    it with the same Stage-1 ``_cf_uncertainty`` as the fixed policies -- a direct apples-to-apples
    "generator vs ego_only / nearest / all_candidates" comparison on the same scene (0-latency)."""
    from collections import deque

    from car_dreamer.toolkit.wam import (
        BevSpec,
        GraphBuildSpec,
        append_record_jsonl,
        build_stage1_policy_graph,
        hetero_graph_to_record,
    )

    state_fn = getattr(sim, "_wam_stage1_slot_state", None)
    if state_fn is None:
        return False
    step = int(getattr(sim, "_time_step", 0))
    state = state_fn(step)
    spec = GraphBuildSpec(
        route_waypoints=int(getattr(sim, "_wam_graph_route_waypoints", 16)),
        max_object_nodes=int(getattr(sim, "_wam_graph_max_object_nodes", 32)),
    )
    bev_spec = getattr(sim, "_wam_bev_spec", BevSpec())
    world_extra = _timeline_world_extra(sim, getattr(sim, "_wam_graph", None), map_background=map_background)
    episode = int(getattr(sim, "_episode_count", 0) or 0)
    specs = list(_cf_policy_specs(state, tokens))
    if uwm is not None:
        gen = _cf_stage2_policy(uwm, sim, state, spec, bev_spec, device, n_candidates=uwm_candidates)
        if gen is not None:
            specs.append(("stage2", gen))
    wrote = False
    for label, policy in specs:
        graph = build_stage1_policy_graph(
            ego=state["ego"], collaborators=tuple(state.get("collaborators", ())),
            objects=tuple(state.get("live_states", ())), policy=policy, spec=spec,
            notable_ids=state.get("notable_ids", ()), latency_by_vehicle={}, bev_spec=bev_spec,
            bev_payload_mode=str(getattr(sim, "_wam_bev_payload_mode", "feature")),
            bev_feature_dim=int(getattr(sim, "_wam_bev_feature_dim", 256)),
            bev_feature_dtype_bytes=int(getattr(sim, "_wam_bev_feature_dtype_bytes", 4)),
            gamma_freshness=float(getattr(sim, "_wam_graph_gamma_freshness", 5.0)),
            overhead_bytes=int(getattr(sim, "_comm_overhead_bytes", 64)),
        )
        win = windows.setdefault(label.split("[", 1)[0], deque(maxlen=int(history_window) + 1))
        win.append(graph)
        unc = _cf_uncertainty(model, list(win), sim, state, policy, device, debug=debug)
        extra = {"episode": episode, "uncertainty": unc, "counterfactual": True}
        extra.update(world_extra)
        rec = hetero_graph_to_record(graph, step=step, policy_label=label, extra=extra)
        append_record_jsonl(rec, out_path)
        wrote = True
    return wrote


def record_counterfactual_comm_step(sim, model, comm_recorder, slot_window, out_path, *,
                                    history_window, sample_period_steps, device,
                                    map_background=None, debug=False):
    """One rollout step for the COMMUNICATION-REPLAY counterfactual timeline.

    Reuses the proven comm-replay machinery (one :class:`CommunicationProcess` per policy in the stable
    family ``ego_only / single_candidate_* / all_candidates_*``) so each policy's graph window is built
    from messages that have *actually arrived* under that policy's bandwidth/latency -- matching the
    online perception conditions, unlike the 0-latency :func:`record_counterfactual_step`. Windows are
    sampled at the training/online cadence (``sample_period_steps``) and only scored once full, mirroring
    ``_checkpoint_graph_window_ready``.
    """
    from car_dreamer.toolkit.wam import append_record_jsonl, coverage_metrics, hetero_graph_to_record

    state_fn = getattr(sim, "_wam_stage1_slot_state", None)
    if state_fn is None:
        return False
    step = int(getattr(sim, "_time_step", 0))
    state = state_fn(step)
    # build per-policy comm runtimes once (stable family), advance the comm queues EVERY step
    comm_recorder._ensure_runtimes(state, start_step=0)
    comm_recorder._advance_communication(step, state)
    # sample windows at the training/online cadence; only score once the window is full (in-distribution)
    if int(step) % int(max(sample_period_steps, 1)) != 0:
        return False
    slot_window.append((int(step), dict(state)))
    if len(slot_window) < int(history_window) + 1:
        return False
    payload = {"window_steps": [s for s, _ in slot_window], "source_window": [st for _, st in slot_window]}
    world_extra = _timeline_world_extra(sim, getattr(sim, "_wam_graph", None), map_background=map_background)
    episode = int(getattr(sim, "_episode_count", 0) or 0)
    wrote = False
    for key in comm_recorder._policy_order:
        runtime = comm_recorder._runtimes[key]
        graphs, coverage_tensor = comm_recorder._build_window(runtime, step, payload)
        if not graphs:
            continue
        cov_override = None
        if coverage_tensor is not None and len(coverage_tensor) > 0:
            arr = coverage_tensor[-1]
            arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr
            cov_override = float(coverage_metrics(arr).get("coverage_uncertainty", 0.0))
        unc = _cf_uncertainty(model, graphs, sim, state, runtime.policy, device,
                              debug=debug, coverage_override=cov_override)
        sel = ",".join(str(int(v)) for v in runtime.policy.selected_vehicle_ids)
        label = runtime.policy_type if not sel else f"{runtime.policy_type}[{sel}]"
        extra = {"episode": episode, "uncertainty": unc, "counterfactual": True, "comm_replay": True}
        extra.update(world_extra)
        rec = hetero_graph_to_record(graphs[-1], step=step, policy_label=label, extra=extra)
        append_record_jsonl(rec, out_path)
        wrote = True
    return wrote


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Record the per-step WAM cooperative graph to JSONL.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", default="outputs/graph_timeline.jsonl", help="output JSONL path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--wide-bev", action="store_true", default=False,
                        help="dump a wide 100m ego-centered birdeye (birdeye_wam100) for the BEV panel; "
                             "render with visualize ... --bev-obs-range 100 --bev-ego-offset 50.")
    parser.add_argument("--map-background", action="store_true", default=True,
                        help="write a single CARLA map background PNG for generated BEV rendering.")
    parser.add_argument("--no-map-background", dest="map_background", action="store_false")
    parser.add_argument("--map-background-ppm", type=float, default=4.0,
                        help="pixels per meter for the generated global map background.")
    parser.add_argument("--no-display", dest="display", action="store_false", default=False)
    parser.add_argument("--counterfactual", action="store_true", default=False,
                        help="record, in ONE rollout, a per-step graph + uncertainty for EACH --cf-policies "
                             "on the same scene (counterfactual). Needs env.wam.predictor_mode=checkpoint.")
    parser.add_argument("--cf-policies", default="ego_only,nearest_single",
                        help="counterfactual policies to record per step: any of ego_only, nearest_single, "
                             "all_candidates (comma-separated)")
    parser.add_argument("--cf-debug", action="store_true", default=False,
                        help="dump per-object notable_prob / TrSigma / w_p / gt_notable into each "
                             "counterfactual record's extra.uncertainty.per_object (for diagnosing "
                             "why the prob口径 motion is large)")
    parser.add_argument("--cf-comm-replay", action="store_true", default=False,
                        help="counterfactual with REAL V2V latency: replay a per-policy CommunicationProcess "
                             "and build each policy's graph from delivered messages (stable family "
                             "ego_only/single_candidate_*/all_candidates_*). Default is 0-latency. "
                             "--cf-policies is ignored in this mode.")
    parser.add_argument("--cf-uwm-checkpoint", default=None,
                        help="Stage-2 UWM checkpoint; when set, also adds a 'stage2'-generated policy to the "
                             "0-latency counterfactual set, scored by the same Stage-1 uncertainty "
                             "(Phase-2 generator-vs-enumerated comparison). Ignored with --cf-comm-replay.")
    parser.add_argument("--cf-uwm-candidates", type=int, default=1,
                        help="n_candidates for propose_policies when --cf-uwm-checkpoint is set")
    known, passthrough = parser.parse_known_args()
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def main() -> int:
    known, passthrough = parse_args()
    _setup_carla_pythonapi()

    out_path = Path(known.out)
    if out_path.exists():
        out_path.unlink()  # fresh timeline per run

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        *passthrough,
    ]
    env, _ = build_env(known.task, env_args, wide_bev=known.wide_bev)
    sim = env.unwrapped

    env.reset(seed=known.seed)
    map_background = _write_map_background(sim, out_path, pixels_per_meter=known.map_background_ppm) if known.map_background else None

    cf_model = cf_windows = cf_device = cf_tokens = cf_history = None
    cf_comm_recorder = cf_slot_window = cf_sample_period = None
    cf_uwm = None
    if known.counterfactual:
        loader = getattr(sim, "_load_wam_predictor", None)
        if loader is None:
            raise SystemExit("--counterfactual needs a checkpoint predictor; pass "
                             "--env.wam.predictor_mode=checkpoint --env.wam.predictor_checkpoint=<path>")
        cf_model = loader()
        cf_device = getattr(sim, "_wam_predictor_device_resolved", None) or sim._resolve_wam_predictor_device()
        cf_history = int(getattr(sim, "_wam_predictor_history_window", 4))
        cf_tokens = [t.strip() for t in str(known.cf_policies).split(",") if t.strip()]
        cf_windows = {}
        if known.cf_comm_replay:
            from collections import deque

            from car_dreamer.toolkit.wam import BevSpec, WAMStage1CommunicationPolicyDataRecorder

            cf_sample_period = max(1, int(getattr(sim, "_wam_predictor_sample_period_steps", 1)))
            cf_comm_recorder = WAMStage1CommunicationPolicyDataRecorder(
                out_path.parent / "cf_comm_runtime",
                fixed_dt=float(sim._comm_config.dt),
                comm_config=sim._comm_config,
                link_rate_bps=sim._link_rate_bps,
                graph_builder=sim._build_wam_graph_for_stage1_slot,
                coverage_builder=getattr(sim, "_build_wam_coverage_for_stage1_slot", None),
                history_window=cf_history,
                bandwidth_ratio=float(getattr(sim, "_comm_bandwidth_ratio", 1.0)),
                bev_spec=getattr(sim, "_wam_bev_spec", None) or BevSpec(),
                bev_payload_mode=str(getattr(sim, "_wam_bev_payload_mode", "feature")),
                bev_feature_dim=int(getattr(sim, "_wam_bev_feature_dim", 256)),
                bev_feature_dtype_bytes=int(getattr(sim, "_wam_bev_feature_dtype_bytes", 4)),
                overhead_bytes=int(getattr(sim, "_comm_overhead_bytes", 64)),
            )
            cf_slot_window = deque(maxlen=cf_history + 1)
            print(f"counterfactual COMM-REPLAY: stable policy family, history_window={cf_history} "
                  f"sample_period_steps={cf_sample_period} (--cf-policies ignored)", flush=True)
        else:
            cf_uwm = None
            if known.cf_uwm_checkpoint:
                from car_dreamer.toolkit.wam import load_wam_uwm

                cf_uwm = load_wam_uwm(known.cf_uwm_checkpoint, device=cf_device)
                print(f"counterfactual + Stage-2 generator: uwm={known.cf_uwm_checkpoint} "
                      f"candidates={known.cf_uwm_candidates}", flush=True)
            print(f"counterfactual recording (0-latency): policies={cf_tokens} history_window={cf_history}", flush=True)

    recorded = 0
    try:
        # Single-episode recording: run until the episode ends (or the --steps cap), then stop.
        for step in range(known.steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if known.counterfactual and known.cf_comm_replay:
                if record_counterfactual_comm_step(sim, cf_model, cf_comm_recorder, cf_slot_window, out_path,
                                                   history_window=cf_history, sample_period_steps=cf_sample_period,
                                                   device=cf_device, map_background=map_background,
                                                   debug=known.cf_debug):
                    recorded += 1
            elif known.counterfactual:
                if record_counterfactual_step(sim, cf_model, cf_windows, out_path, tokens=cf_tokens,
                                              history_window=cf_history, device=cf_device,
                                              map_background=map_background, debug=known.cf_debug,
                                              uwm=cf_uwm, uwm_candidates=known.cf_uwm_candidates):
                    recorded += 1
            elif record_step(sim, out_path, map_background=map_background):
                recorded += 1
            if step % known.print_every == 0:
                label, pid = active_policy_label(sim)
                print(f"step {step}: policy={label} (id={pid}) recorded={recorded}", flush=True)
            if terminated or truncated:
                print(f"episode ended at step {step}; stopping (single-episode recording).", flush=True)
                break
            if known.sleep:
                time.sleep(known.sleep)
    finally:
        env.close()
    print(f"Recorded {recorded} graph frames (single episode) -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
