#!/usr/bin/env python3
"""Record policy-augmented WAM Stage-1 samples from live CARLA.

Each recorded live step enumerates a fixed policy family and writes one graph-window
sample per policy. All policy samples for the same step share the same future
trajectory GT, but carry different policy-conditioned graphs and metadata.

Example:
    python scripts/record_wam_stage1_policy_data.py --task carla_group_right_turn_auto \
        --carla-port 2000 --steps 400 --out-dir data/wam_stage1_policy
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Record policy-augmented WAM Stage-1 samples.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("data/wam_stage1_policy"))
    parser.add_argument("--future-horizon-s", type=float, default=3.0)
    parser.add_argument("--graph-timeline-jsonl", default=None,
                        help="also emit a per-step, per-policy cooperative-graph timeline JSONL "
                             "(mode-B counterfactual) for scripts/visualize_wam_graph_timeline.py. "
                             "When set, recording stops after a SINGLE episode (one consistent "
                             "candidate set, no cross-episode frames).")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    known, passthrough = parser.parse_known_args()
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def _fixed_dt(sim, config) -> float:
    settings = getattr(getattr(sim, "_world", None), "_settings", None)
    if settings is not None and getattr(settings, "fixed_delta_seconds", None):
        return float(settings.fixed_delta_seconds)
    return float(getattr(getattr(config.env, "world", None), "fixed_delta_seconds", 0.1))


def _carla_world(sim):
    return getattr(getattr(sim, "_world", None), "_world", None)


def _actor_snapshots(sim) -> Dict[int, object]:
    from car_dreamer.toolkit.wam import snapshot_from_carla_actor

    world = _carla_world(sim)
    snapshots: Dict[int, object] = {}
    if world is None:
        return snapshots
    actors = world.get_actors()
    for actor in list(actors.filter("vehicle.*")) + list(actors.filter("walker.pedestrian.*")):
        try:
            snap = snapshot_from_carla_actor(actor)
            snapshots[int(snap.actor_id)] = snap
        except RuntimeError:
            pass
    return snapshots


def _plain_config(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(v) for v in value]
    return str(value)


def _get_candidate_actors(sim, candidate_ids: Iterable[int]):
    actors = {}
    for vid in sorted({int(v) for v in candidate_ids}):
        actor = None
        try:
            actor = sim._get_group_member_actor(vid)
        except Exception:
            actor = None
        if actor is not None:
            actors[int(vid)] = actor
    return actors


def _latency_by_vehicle(sim, policy, objects, candidate_actors) -> Dict[int, float]:
    latency: Dict[int, float] = {}
    selected = [int(v) for v in policy.selected_vehicle_ids]
    out_degree = max(len(selected), 1)
    for vid in selected:
        actor = candidate_actors.get(vid)
        if actor is None:
            continue
        modality = str(policy.modality_by_vehicle.get(vid, "objlist"))
        if modality == "bev" and hasattr(sim, "_wam_bev_payload_bytes"):
            payload = float(sim._wam_bev_payload_bytes())
        elif modality == "bev":
            payload = float(getattr(sim, "_wam_graph_bev_channels", 7)) * float(getattr(sim, "_wam_graph_bev_size", 64)) ** 2
        else:
            observed = [s for s in objects if int(vid) in s.visible_to_collaborators]
            if hasattr(sim, "_wam_objlist_payload_bytes"):
                payload = float(sim._wam_objlist_payload_bytes(len(observed)))
            else:
                from car_dreamer.toolkit.wam import objlist_payload_bytes

                payload = objlist_payload_bytes(len(observed))
        try:
            latency[vid] = float(
                sim.latency_model.compute_latency_s(
                    sender=actor,
                    receiver=sim.ego,
                    payload_size_bytes=int(payload),
                    sender_res=sim._veh_net_res.get(int(vid), sim._default_net_res),
                    receiver_res=sim._default_net_res,
                    out_degree=out_degree,
                    in_degree=out_degree,
                )
            )
        except Exception:
            latency[vid] = 0.0
    return latency


def _policy_graphs_for_step(sim, *, step: int, episode_id: int, fixed_dt: float):
    from car_dreamer.toolkit.wam import (
        BevSpec,
        GraphBuildSpec,
        build_stage1_policy_graph,
        enumerate_stage1_policies,
        make_stage1_policy_metadata,
        policy_key,
        visible_object_ids_by_vehicle,
    )

    if getattr(sim, "ego", None) is None:
        return []
    if hasattr(sim, "_update_wam_runtime_state"):
        sim._update_wam_runtime_state(force=True)

    objects = list(getattr(sim, "_wam_object_states", []))
    notable_records = list(getattr(sim, "_wam_notable_records", []))
    notable_ids = [int(record.object_state.actor_id) for record in notable_records]
    all_candidate_ids = sorted({int(v) for v in getattr(sim, "coop_participant_ids", set())})
    candidate_actors = _get_candidate_actors(sim, all_candidate_ids)
    candidate_ids = sorted(candidate_actors)

    ego = sim._wam_vehicle_node_input(sim.ego, is_ego=True, agent_slot=0, route_xy=sim._wam_route_xy())
    collaborators = [
        sim._wam_vehicle_node_input(actor, is_ego=False, agent_slot=slot)
        for slot, actor in enumerate((candidate_actors[vid] for vid in candidate_ids), start=1)
    ]
    spec = GraphBuildSpec(
        route_waypoints=int(getattr(sim, "_wam_graph_route_waypoints", 6)),
        max_object_nodes=int(getattr(sim, "_wam_graph_max_object_nodes", 32)),
    )
    bev_spec = getattr(
        sim,
        "_wam_bev_spec",
        BevSpec(size=int(getattr(sim, "_wam_graph_bev_size", 64)), range_m=float(getattr(sim, "_wam_graph_bev_range_m", 50.0))),
    )
    visible_ids = visible_object_ids_by_vehicle(int(sim.ego.id), candidate_ids, objects)
    policies = enumerate_stage1_policies(
        candidate_ids,
        uplink_bps=float(getattr(getattr(sim, "_default_net_res", None), "uplink_bps", 0.0)),
        frequency_steps=int(getattr(sim, "comm_period", 1)),
    )
    ego_pose = (float(ego.x), float(ego.y), float(ego.yaw))

    rows = []
    for policy_type, policy in policies:
        latency = _latency_by_vehicle(sim, policy, objects, candidate_actors)
        graph = build_stage1_policy_graph(
            ego=ego,
            collaborators=collaborators,
            objects=objects,
            policy=policy,
            spec=spec,
            notable_ids=notable_ids,
            latency_by_vehicle=latency,
            bev_spec=bev_spec,
            gamma_freshness=float(getattr(sim, "_wam_graph_gamma_freshness", 5.0)),
            overhead_bytes=int(getattr(getattr(sim, "latency_model", None), "overhead_bytes", 64)),
        )
        metadata = make_stage1_policy_metadata(
            step=step,
            episode_id=episode_id,
            policy_type=policy_type,
            policy=policy,
            candidate_vehicle_ids=candidate_ids,
            notable_object_ids=notable_ids,
            visible_ids_by_vehicle=visible_ids,
            ego_pose=ego_pose,
            fixed_dt=fixed_dt,
        )
        rows.append((policy_key(policy_type, policy), graph, ego_pose, metadata))
    return rows


def _emit_timeline_record(graph, step: int, metadata, path: Path) -> None:
    """Append a per-policy cooperative-graph record (mode-B counterfactual) to the timeline JSONL."""
    from car_dreamer.toolkit.wam import append_record_jsonl, hetero_graph_to_record, policy_label

    policy = metadata.get("policy", {}) if isinstance(metadata, dict) else {}
    label = policy_label(str(metadata.get("policy_type", "policy")), policy.get("selected_vehicle_ids", []))
    record = hetero_graph_to_record(
        graph,
        step=int(step),
        policy_label=label,
        extra={
            "policy_type": metadata.get("policy_type"),
            "episode": int(metadata.get("episode_id", 0)),
        },
    )
    append_record_jsonl(record, path)


def main() -> int:
    known, passthrough = parse_args()
    if known.steps <= 0:
        raise ValueError("--steps must be positive")
    if known.future_horizon_s <= 0:
        raise ValueError("--future-horizon-s must be positive")

    _setup_carla_pythonapi()

    from car_dreamer.toolkit.wam import (
        STAGE1_POLICY_TYPES,
        WAMStage1PolicyDataRecorder,
        wam_stage1_configs_from_env,
    )

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        "--env.wam.build_graph=True",
        *passthrough,
    ]
    env, config = build_env(known.task, env_args)
    sim = env.unwrapped
    perc_cfg, stage1_cfg = wam_stage1_configs_from_env(config)

    timeline_path = Path(known.graph_timeline_jsonl) if known.graph_timeline_jsonl else None
    if timeline_path is not None and timeline_path.exists():
        timeline_path.unlink()  # fresh timeline per run

    try:
        env.reset(seed=known.seed)
        fixed_dt = _fixed_dt(sim, config)
        manifest = {
            "task": known.task,
            "graph_config": _plain_config(perc_cfg.graph_config()),
            "stage1_config": _plain_config(stage1_cfg),
            "policy_types": list(STAGE1_POLICY_TYPES),
            "history_window": int(stage1_cfg.history_window),
            "trajectory_horizon": int(perc_cfg.traj_samples),
            "future_horizon_s": float(known.future_horizon_s),
        }
        recorder = WAMStage1PolicyDataRecorder(
            known.out_dir,
            fixed_dt=fixed_dt,
            horizon_s=known.future_horizon_s,
            samples=int(perc_cfg.traj_samples),
            history_window=int(stage1_cfg.history_window),
            manifest=manifest,
        )
        print(
            f"Recording {known.steps} policy-augmented steps to {known.out_dir} "
            f"(window={stage1_cfg.history_window + 1}, horizon_steps={recorder.horizon_steps})",
            flush=True,
        )

        last_record_step = int(known.steps) - 1
        last_observe_step = last_record_step + int(recorder.horizon_steps)
        episode_id = 0
        global_step = 0

        while global_step <= last_observe_step:
            recorder.observe(global_step, _actor_snapshots(sim))
            if global_step <= last_record_step:
                for key, graph, ego_pose, metadata in _policy_graphs_for_step(
                    sim, step=global_step, episode_id=episode_id, fixed_dt=fixed_dt
                ):
                    recorder.register(global_step, key=key, graph=graph, ego_pose=ego_pose, metadata=metadata)
                    if timeline_path is not None:
                        _emit_timeline_record(graph, global_step, metadata, timeline_path)
            recorder.flush_ready(global_step)

            if global_step % known.print_every == 0:
                print(f"step={global_step} episode={episode_id} written={recorder.written}", flush=True)

            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            global_step += 1
            if terminated or truncated:
                recorder.flush_all()
                if timeline_path is not None:
                    # Single-episode timeline: one consistent candidate set, no cross-episode frames.
                    print("episode ended; stopping (single-episode timeline recording).", flush=True)
                    break
                recorder.reset_episode()
                episode_id += 1
                env.reset(seed=known.seed)

        recorder.flush_all()
        recorder.write_manifest()
        print(f"Done. wrote {recorder.written} samples to {known.out_dir}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
