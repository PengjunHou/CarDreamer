#!/usr/bin/env python3
"""Compare fixed WAM policies across episodes.

Each episode installs one communication policy for the whole rollout, records
the resulting WAM graph topology, and writes a policy/uncertainty time series.
This is useful for comparing ego-only, single-collaborator, and all-collaborator
policies without the online Base-Station policy lifecycle switching underneath.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_POLICIES = (
    "ego_only",
    "single_candidate_objlist",
    "single_candidate_bev",
    "single_candidate_bev_objlist",
    "all_candidates_objlist",
    "all_candidates_bev",
    "all_candidates_bev_objlist",
)

POLICY_MODALITIES = {
    "single_candidate_objlist": ("objlist",),
    "single_candidate_bev": ("bev",),
    "single_candidate_bev_objlist": ("bev", "objlist"),
    "all_candidates_objlist": ("objlist",),
    "all_candidates_bev": ("bev",),
    "all_candidates_bev_objlist": ("bev", "objlist"),
}


def _setup_carla_pythonapi() -> None:
    carla_root = os.environ.get("CARLA_ROOT", "/home/peh324/carla_simulator")
    for sub in ("PythonAPI", "PythonAPI/carla"):
        path = os.path.join(carla_root, sub)
        if path not in sys.path:
            sys.path.append(path)


def build_env(task: str, env_args: List[str]):
    import gymnasium as gym

    import car_dreamer
    from car_dreamer import toolkit

    config = car_dreamer.load_task_configs(task)
    config, _ = toolkit.Flags(config).parse_known(env_args)
    env = gym.make(config.env.name, config=config.env)
    return env, config


def _parse_policies(text: str) -> List[str]:
    out = [item.strip() for item in str(text).split(",") if item.strip()]
    valid = set(DEFAULT_POLICIES)
    bad = [item for item in out if item not in valid]
    if bad:
        raise ValueError(f"unknown policy type(s): {bad}; valid={sorted(valid)}")
    return out or list(DEFAULT_POLICIES)


def _candidate_ids(sim) -> List[int]:
    ids = sorted(int(v) for v in getattr(sim, "coop_participant_ids", set()))
    if ids:
        return ids
    return sorted(int(actor.id) for actor in getattr(sim, "group_vehs", []) if actor is not None)


def _modalities_for_policy(policy_type: str) -> Tuple[str, ...]:
    return tuple(POLICY_MODALITIES.get(str(policy_type), ()))


def _policy_specs(policy_types: Sequence[str], candidate_ids: Sequence[int], single_candidates: str):
    candidates = tuple(sorted(int(v) for v in candidate_ids))
    for policy_type in policy_types:
        if policy_type == "ego_only":
            yield policy_type, (), ()
        elif policy_type.startswith("single_candidate_"):
            selected_ids = candidates if single_candidates == "all" else candidates[:1]
            modalities = _modalities_for_policy(policy_type)
            for vid in selected_ids:
                yield policy_type, (int(vid),), modalities
        elif policy_type.startswith("all_candidates_"):
            modalities = _modalities_for_policy(policy_type)
            yield policy_type, candidates, modalities


def _install_fixed_policy(
    sim,
    *,
    policy_type: str,
    selected: Sequence[int],
    modalities: Sequence[str],
    start_step: int,
    duration_steps: int,
    bandwidth_ratio: float,
):
    from car_dreamer.toolkit import CommPolicy, make_local_policy

    proc = sim._ensure_comm_process()
    if not selected:
        policy = make_local_policy(
            policy_id=sim._next_policy_id(),
            request_vehicle_id=int(sim.ego.id),
            start_step=int(start_step),
            duration_steps=int(duration_steps),
        )
        policy = CommPolicy(
            policy_id=int(policy.policy_id),
            request_vehicle_id=int(policy.request_vehicle_id),
            start_step=int(policy.start_step),
            duration_steps=int(policy.duration_steps),
            selected_collaborators=(),
            modalities_by_vehicle={},
            bandwidth_by_vehicle={},
            reason=f"fixed_{policy_type}",
        )
    else:
        selected_tuple = tuple(sorted(int(v) for v in selected))
        policy = CommPolicy(
            policy_id=sim._next_policy_id(),
            request_vehicle_id=int(sim.ego.id),
            start_step=int(start_step),
            duration_steps=int(duration_steps),
            selected_collaborators=selected_tuple,
            modalities_by_vehicle={int(v): tuple(str(m) for m in modalities) for v in selected_tuple},
            bandwidth_by_vehicle={int(v): float(bandwidth_ratio) for v in selected_tuple},
            reason=f"fixed_{policy_type}",
        )
    proc.set_policy(policy, int(start_step))
    sim._sync_policy_views(policy)
    return policy


def _fixed_label(policy_type: str, selected: Sequence[int]) -> str:
    if not selected:
        return "ego_only"
    return f"{policy_type}[{','.join(str(int(v)) for v in selected)}]"


def _wam_update(sim) -> None:
    update = getattr(sim, "_update_wam_runtime_state", None)
    if update is not None:
        update(force=True)


def _max_mean_uncertainty(predictions: Dict[int, object]) -> Tuple[float, float]:
    vals = [float(pred.uncertainty_score) for pred in predictions.values()]
    if not vals:
        return 0.0, 0.0
    return max(vals), sum(vals) / len(vals)


def _uncertainty_breakdown(sim, legacy_max: float, legacy_mean: float) -> Dict[str, float]:
    breakdown = dict(getattr(sim, "_wam_uncertainty_breakdown", {}) or {})
    motion = float(breakdown.get("motion_uncertainty", legacy_mean))
    coverage = float(breakdown.get("coverage_uncertainty", 0.0))
    total = float(breakdown.get("total_uncertainty", motion + coverage))
    return {
        "motion_uncertainty": motion,
        "coverage_uncertainty": coverage,
        "total_uncertainty": total,
        "route_coverage_quality_mean": float(breakdown.get("route_coverage_quality_mean", 0.0)),
        "poor_coverage_risk_mean": float(breakdown.get("poor_coverage_risk_mean", 0.0)),
        "route_coverage_ratio": float(breakdown.get("route_coverage_ratio", 0.0)),
        "legacy_uncertainty_max": float(legacy_max),
        "legacy_uncertainty_mean": float(legacy_mean),
    }


def _json_list(values: Iterable[int]) -> str:
    return json.dumps([int(v) for v in values], separators=(",", ":"))


def _json_mapping(mapping: Dict[int, object]) -> str:
    return json.dumps({str(int(k)): v for k, v in mapping.items()}, sort_keys=True, separators=(",", ":"))


def _write_uncertainty_plot(csv_path: Path, out_png: Path) -> Optional[Path]:
    import pandas as pd
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(13, 5))
    for label, group in df.groupby("policy_label", sort=False):
        ax1.plot(group["step"], group["uncertainty_max"], label=str(label), linewidth=1.8)
    ax1.set_xlabel("step")
    ax1.set_ylabel("max uncertainty")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = ax1.twinx()
    for label, group in df.groupby("policy_label", sort=False):
        ax2.step(group["step"], group["num_selected"], where="post", alpha=0.18)
    ax2.set_ylabel("# selected collaborators")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Compare fixed WAM policies and their uncertainty over time.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/wam_fixed_policy_compare"))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--single-candidates", choices=("all", "first"), default="all")
    parser.add_argument("--bandwidth-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-display", dest="display", action="store_false", default=True)
    parser.add_argument("--render-html", action="store_true", default=False,
                        help="also render topology HTML at the end; slower, normally use the offline visualizer")
    parser.add_argument("--render-plot", action="store_true", default=False,
                        help="also render uncertainty PNG at the end; normally use the offline visualizer")
    known, passthrough = parser.parse_known_args()
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def main() -> int:
    args, passthrough = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.sleep < 0:
        raise ValueError("--sleep must be non-negative")

    _setup_carla_pythonapi()

    env_args = [
        f"--env.world.carla_port={args.carla_port}",
        f"--env.display.enable={bool(args.display)}",
        *passthrough,
    ]
    env, _ = build_env(args.task, env_args)
    sim = env.unwrapped

    # The host env calls this every step; replace it so our fixed policy is not
    # overwritten by the online Base-Station lifecycle.
    sim._update_policy_lifecycle = lambda step: None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_jsonl = args.out_dir / "fixed_policy_graph_timeline.jsonl"
    uncertainty_csv = args.out_dir / "fixed_policy_uncertainty.csv"
    topology_html = args.out_dir / "fixed_policy_graph_timeline.html"
    uncertainty_png = args.out_dir / "fixed_policy_uncertainty.png"

    from car_dreamer.toolkit.wam import append_record_jsonl, hetero_graph_stats, hetero_graph_to_record

    policy_types = _parse_policies(args.policies)
    fields = [
        "episode_index",
        "step",
        "policy_label",
        "policy_type",
        "selected_vehicle_ids",
        "modality_by_vehicle",
        "num_selected",
        "uncertainty_max",
        "uncertainty_mean",
        "motion_uncertainty",
        "coverage_uncertainty",
        "total_uncertainty",
        "route_coverage_quality_mean",
        "poor_coverage_risk_mean",
        "route_coverage_ratio",
        "num_predictions",
        "notable_object_ids",
        "visible_notable_object_ids",
        "invisible_notable_object_ids",
        "graph_vehicles",
        "graph_observations",
        "graph_objects",
        "graph_veh_veh",
    ]

    written_graphs = 0
    written_rows = 0
    graph_jsonl.write_text("", encoding="utf-8")
    with uncertainty_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        episode_index = 0
        # Reset once only to learn how many single-candidate episodes to schedule.
        # Actor ids can change across reset, so later episodes select by sorted candidate index.
        env.reset(seed=args.seed)
        discovered_candidates = _candidate_ids(sim)
        if not discovered_candidates and any(p != "ego_only" for p in policy_types):
            print("warning: no cooperative candidates discovered; non-local policies will be empty", flush=True)

        plans = []
        for policy_type in policy_types:
            if policy_type == "ego_only":
                plans.append((policy_type, None, ()))
            elif policy_type.startswith("single_candidate_"):
                modalities = _modalities_for_policy(policy_type)
                count = len(discovered_candidates) if args.single_candidates == "all" else min(1, len(discovered_candidates))
                for candidate_index in range(count):
                    plans.append((policy_type, candidate_index, modalities))
            elif policy_type.startswith("all_candidates_"):
                modalities = _modalities_for_policy(policy_type)
                plans.append((policy_type, "all", modalities))
        if not plans:
            raise RuntimeError("no fixed policy plans generated")

        for policy_type, candidate_selector, modalities in plans:
            env.reset(seed=args.seed)
            candidates = _candidate_ids(sim)
            if candidate_selector == "all":
                selected = tuple(candidates)
            elif isinstance(candidate_selector, int):
                if candidate_selector >= len(candidates):
                    print(f"skip {policy_type}[candidate#{candidate_selector}]: not enough candidates after reset", flush=True)
                    continue
                selected = (int(candidates[candidate_selector]),)
            else:
                selected = ()

            start_step = int(getattr(sim, "_time_step", 0))
            policy = _install_fixed_policy(
                sim,
                policy_type=policy_type,
                selected=selected,
                modalities=modalities,
                start_step=start_step,
                duration_steps=int(args.steps) + 1000,
                bandwidth_ratio=float(args.bandwidth_ratio),
            )
            label = _fixed_label(policy_type, selected)
            print(f"[episode {episode_index}] fixed_policy={label} candidates={candidates}", flush=True)

            for local_step in range(int(args.steps)):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                _wam_update(sim)
                step = int(getattr(sim, "_time_step", local_step))
                graph = getattr(sim, "_wam_graph", None)
                if graph is not None:
                    stats = hetero_graph_stats(graph)
                    extra = {
                        "episode": int(episode_index),
                        "policy_type": str(policy_type),
                        "fixed_policy_label": str(label),
                        "selected_vehicle_ids": [int(v) for v in selected],
                        "modality_by_vehicle": {str(int(k)): list(v) for k, v in policy.modalities_by_vehicle.items()},
                    }
                    record = hetero_graph_to_record(
                        graph,
                        step=step,
                        policy_label=label,
                        policy_id=int(policy.policy_id),
                        extra=extra,
                    )
                    append_record_jsonl(record, graph_jsonl)
                    written_graphs += 1
                else:
                    stats = {}

                notable = list(getattr(sim, "_wam_notable_records", []))
                predictions = dict(getattr(sim, "_wam_motion_predictions", {}))
                unc_max, unc_mean = _max_mean_uncertainty(predictions)
                unc = _uncertainty_breakdown(sim, unc_max, unc_mean)
                modality_by_vehicle = {int(k): list(v) for k, v in policy.modalities_by_vehicle.items()}
                writer.writerow(
                    {
                        "episode_index": int(episode_index),
                        "step": int(step),
                        "policy_label": label,
                        "policy_type": policy_type,
                        "selected_vehicle_ids": _json_list(selected),
                        "modality_by_vehicle": _json_mapping(modality_by_vehicle),
                        "num_selected": len(selected),
                        "uncertainty_max": float(unc_max),
                        "uncertainty_mean": float(unc_mean),
                        "motion_uncertainty": float(unc["motion_uncertainty"]),
                        "coverage_uncertainty": float(unc["coverage_uncertainty"]),
                        "total_uncertainty": float(unc["total_uncertainty"]),
                        "route_coverage_quality_mean": float(unc["route_coverage_quality_mean"]),
                        "poor_coverage_risk_mean": float(unc["poor_coverage_risk_mean"]),
                        "route_coverage_ratio": float(unc["route_coverage_ratio"]),
                        "num_predictions": len(predictions),
                        "notable_object_ids": _json_list(
                            int(r.object_state.actor_id) for r in notable
                        ),
                        "visible_notable_object_ids": _json_list(
                            int(r.object_state.actor_id) for r in notable if bool(r.visible)
                        ),
                        "invisible_notable_object_ids": _json_list(
                            int(r.object_state.actor_id) for r in notable if bool(r.invisible)
                        ),
                        "graph_vehicles": int(stats.get("wam_graph_num_vehicle_nodes", 0)),
                        "graph_observations": int(stats.get("wam_graph_num_observation_nodes", 0)),
                        "graph_objects": int(stats.get("wam_graph_num_object_nodes", 0)),
                        "graph_veh_veh": int(stats.get("wam_graph_num_veh_veh_edges", 0)),
                    }
                )
                written_rows += 1

                if local_step % int(args.print_every) == 0:
                    print(
                        f"  step={step} total_uncertainty={unc['total_uncertainty']:.4f} "
                        f"legacy_max={unc_max:.4f} "
                        f"notable={[int(r.object_state.actor_id) for r in notable]}",
                        flush=True,
                    )
                if terminated or truncated:
                    print(
                        f"  episode ended early at local_step={local_step}: "
                        f"terminated={terminated} truncated={truncated}",
                        flush=True,
                    )
                    break
                if args.sleep:
                    import time

                    time.sleep(float(args.sleep))

            episode_index += 1

    plot_path = None
    if args.render_plot:
        plot_path = _write_uncertainty_plot(uncertainty_csv, uncertainty_png)
    if args.render_html and written_graphs:
        from car_dreamer.toolkit.wam import load_records_jsonl, write_graph_timeline_html

        records = load_records_jsonl(graph_jsonl)
        write_graph_timeline_html(records, topology_html, title="Fixed-policy WAM graph comparison")

    env.close()
    print(f"Graph timeline JSONL -> {graph_jsonl} ({written_graphs} records)", flush=True)
    print(f"Uncertainty CSV      -> {uncertainty_csv} ({written_rows} rows)", flush=True)
    if args.render_html and written_graphs:
        print(f"Topology HTML        -> {topology_html}", flush=True)
    if plot_path is not None:
        print(f"Uncertainty plot     -> {plot_path}", flush=True)
    print("Use scripts/visualize_wam_fixed_policy_compare.py to render offline.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
