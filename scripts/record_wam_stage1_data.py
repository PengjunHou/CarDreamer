#!/usr/bin/env python3
"""Record WAM Stage-1 training windows from the live CARLA env (WAM Design §16.1 data side).

Steps a task, slides each step's policy-conditioned hetero graph (``sim._wam_graph``) into a window, and
once the trajectory horizon elapses pairs the window with the GT future positions of the last graph's
object nodes -- writing self-contained ``.pt`` samples consumed by ``scripts/train_wam_stage1.py`` /
``WAMStage1Dataset``.

Needs CARLA running on ``--carla-port``. Example:
    python scripts/record_wam_stage1_data.py --task carla_group_right_turn_auto --carla-port 2000 \
        --steps 400 --out-dir data/wam_stage1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Record WAM Stage-1 training windows.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("data/wam_stage1"))
    parser.add_argument(
        "--future-horizon-s",
        type=float,
        default=None,
        help="future trajectory horizon in seconds; defaults to env.wam.stage1.traj_horizon_s",
    )
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--policy-sampler",
        choices=("request_all", "random_duration"),
        default="request_all",
        help="communication policy sampler used during recording",
    )
    parser.add_argument(
        "--policy-augmented",
        action="store_true",
        help="record counterfactual samples for ego/single/all candidate policies with policy metadata",
    )
    parser.add_argument(
        "--policy-bandwidth-ratio",
        type=float,
        default=1.0,
        help="bandwidth ratio written into policy-augmented metadata",
    )
    parser.add_argument(
        "--policy-replay-mode",
        choices=("instant", "communication"),
        default="instant",
        help=(
            "policy-augmented replay mode: instant builds counterfactual graphs directly; "
            "communication replays V2V sender/receiver queues and delayed messages offline"
        ),
    )
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument(
        "--single-episode",
        action="store_true",
        help="stop at the first terminated/truncated episode instead of resetting to keep collecting steps",
    )
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


def _comm_snapshot(sim):
    proc = getattr(sim, "_comm_process", None)
    if proc is None:
        return (), None
    return tuple(getattr(proc.receive_queue, "messages", ())), proc.active_policy_id


def _register_current_slot(sim, recorder, step: int) -> bool:
    """Slide the current slot's local graph source state into the Stage-1 window."""
    state_fn = getattr(sim, "_wam_stage1_slot_state", None)
    if state_fn is None:
        return False
    recorder.register_slot(step, state=state_fn(step))
    return True


def _register_policy_augmented_slot(sim, recorder, step: int, *, episode_id: int, fixed_dt: float, bandwidth_ratio: float) -> int:
    from car_dreamer.toolkit.wam import (
        BevSpec,
        GraphBuildSpec,
        build_stage1_policy_graph,
        enumerate_stage1_policies,
        make_stage1_policy_metadata,
        policy_key,
        visible_object_ids_by_vehicle,
    )

    state_fn = getattr(sim, "_wam_stage1_slot_state", None)
    if state_fn is None:
        return 0
    state = state_fn(step)
    ego = state["ego"]
    collaborators = tuple(state.get("collaborators", ()))
    objects = tuple(state.get("live_states", ()))
    candidate_ids = [int(v.actor_id) for v in collaborators]
    policies = enumerate_stage1_policies(
        candidate_ids,
        bandwidth_ratio=float(bandwidth_ratio),
        frequency_steps=int(getattr(sim._comm_config, "sensor_period_steps", 1)),
    )
    visible_ids = visible_object_ids_by_vehicle(int(ego.actor_id), candidate_ids, objects)
    spec = GraphBuildSpec(
        route_waypoints=int(getattr(sim, "_wam_graph_route_waypoints", 16)),
        max_object_nodes=int(getattr(sim, "_wam_graph_max_object_nodes", 32)),
    )
    bev_spec = getattr(sim, "_wam_bev_spec", BevSpec())
    coverage_fn = getattr(sim, "_build_wam_coverage_for_stage1_policy", None)
    count = 0
    for policy_type, policy in policies:
        graph = build_stage1_policy_graph(
            ego=ego,
            collaborators=collaborators,
            objects=objects,
            policy=policy,
            spec=spec,
            notable_ids=state.get("notable_ids", ()),
            latency_by_vehicle={},
            bev_spec=bev_spec,
            bev_payload_mode=str(getattr(sim, "_wam_bev_payload_mode", "feature")),
            bev_feature_dim=int(getattr(sim, "_wam_bev_feature_dim", 256)),
            bev_feature_dtype_bytes=int(getattr(sim, "_wam_bev_feature_dtype_bytes", 4)),
            gamma_freshness=float(getattr(sim, "_wam_graph_gamma_freshness", 5.0)),
            overhead_bytes=int(getattr(sim, "_comm_overhead_bytes", 64)),
        )
        metadata = make_stage1_policy_metadata(
            step=int(step),
            episode_id=int(episode_id),
            policy_type=policy_type,
            policy=policy,
            candidate_vehicle_ids=candidate_ids,
            notable_object_ids=state.get("notable_ids", ()),
            visible_ids_by_vehicle=visible_ids,
            ego_pose=tuple(state["ego_pose"]),
            fixed_dt=float(fixed_dt),
        )
        coverage = None if coverage_fn is None else coverage_fn(state, policy)
        recorder.register(
            int(step),
            key=policy_key(policy_type, policy),
            graph=graph,
            ego_pose=tuple(state["ego_pose"]),
            metadata=metadata,
            coverage=coverage,
        )
        count += 1
    return count


def _episode_seed(base_seed, episode_id: int):
    return None if base_seed is None else int(base_seed) + int(episode_id)


def main() -> int:
    known, passthrough = parse_args()
    if known.steps <= 0:
        raise ValueError("--steps must be positive")
    if known.future_horizon_s is not None and known.future_horizon_s <= 0:
        raise ValueError("--future-horizon-s must be positive")

    _setup_carla_pythonapi()

    import car_dreamer
    from car_dreamer.toolkit.wam import (
        WAMStage1DataRecorder,
        WAMStage1CommunicationPolicyDataRecorder,
        WAMStage1PolicyDataRecorder,
        STAGE1_POLICY_TYPES,
        wam_stage1_configs_from_env,
    )

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        "--env.wam.build_graph=True",
        "--env.wam.predictor_mode=rule",
        f"--env.wam.policy_sampler_mode={known.policy_sampler}",
        *passthrough,
    ]
    env, config = build_env(known.task, env_args)
    sim = env.unwrapped
    perc_cfg, stage1_cfg = wam_stage1_configs_from_env(config)
    future_horizon_s = float(known.future_horizon_s if known.future_horizon_s is not None else perc_cfg.traj_horizon_s)
    known.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env.reset(seed=_episode_seed(known.seed, 0))
        fixed_dt = _fixed_dt(sim, config)
        if known.policy_augmented:
            sample_period_steps = max(1, int(round(float(stage1_cfg.sample_period_s) / fixed_dt)))
            manifest = {
                "task": known.task,
                "policy_augmented": True,
                "policy_replay_mode": str(known.policy_replay_mode),
                "policy_types": list(STAGE1_POLICY_TYPES),
                "sample_period_s": float(stage1_cfg.sample_period_s),
                "sample_period_steps": int(sample_period_steps),
            }
            if str(known.policy_replay_mode) == "communication":
                recorder = WAMStage1CommunicationPolicyDataRecorder(
                    known.out_dir,
                    fixed_dt=fixed_dt,
                    comm_config=sim._comm_config,
                    link_rate_bps=sim._link_rate_bps,
                    graph_builder=sim._build_wam_graph_for_stage1_slot,
                    coverage_builder=getattr(sim, "_build_wam_coverage_for_stage1_slot", None),
                    horizon_s=future_horizon_s,
                    samples=int(perc_cfg.traj_samples),
                    history_window=int(stage1_cfg.history_window),
                    manifest=manifest,
                    bandwidth_ratio=float(known.policy_bandwidth_ratio),
                    bev_spec=getattr(sim, "_wam_bev_spec"),
                    bev_payload_mode=str(getattr(sim, "_wam_bev_payload_mode", "feature")),
                    bev_feature_dim=int(getattr(sim, "_wam_bev_feature_dim", 256)),
                    bev_feature_dtype_bytes=int(getattr(sim, "_wam_bev_feature_dtype_bytes", 4)),
                    overhead_bytes=int(getattr(sim, "_comm_overhead_bytes", 64)),
                )
            else:
                recorder = WAMStage1PolicyDataRecorder(
                    known.out_dir,
                    fixed_dt=fixed_dt,
                    horizon_s=future_horizon_s,
                    samples=int(perc_cfg.traj_samples),
                    history_window=int(stage1_cfg.history_window),
                    manifest=manifest,
                )
        else:
            recorder = WAMStage1DataRecorder(
                known.out_dir,
                fixed_dt=fixed_dt,
                horizon_s=future_horizon_s,
                samples=int(perc_cfg.traj_samples),
                history_window=int(stage1_cfg.history_window),
                sample_period_s=float(stage1_cfg.sample_period_s),
                graph_builder=sim._build_wam_graph_for_stage1_slot,
                coverage_builder=getattr(sim, "_build_wam_coverage_for_stage1_slot", None),
                receive_window_steps=int(sim._comm_config.prediction_window_steps),
                allow_cross_policy_messages=bool(sim._comm_config.allow_cross_policy_messages),
            )
            sample_period_steps = int(recorder.sample_period_steps)
        print(
            f"Recording {known.steps} steps to {known.out_dir} "
            f"(horizon={future_horizon_s:.1f}s/{perc_cfg.traj_samples} samples, "
            f"window={stage1_cfg.history_window + 1}@{stage1_cfg.sample_period_s:.3f}s, "
            f"sample_period_steps={sample_period_steps}, extra_steps={recorder.horizon_steps}, "
            f"policy_sampler={known.policy_sampler}, policy_augmented={known.policy_augmented}, "
            f"policy_replay_mode={known.policy_replay_mode})",
            flush=True,
        )

        last_record_global_step = int(known.steps) - 1
        last_observe_global_step = last_record_global_step + int(recorder.horizon_steps)
        episode_id = 0
        global_step = 0

        while global_step <= last_observe_global_step:
            episode_step = int(getattr(sim, "_time_step", 0))
            recorder.observe(episode_step, _actor_snapshots(sim))
            messages, active_policy_id = _comm_snapshot(sim)
            if hasattr(recorder, "observe_messages"):
                recorder.observe_messages(episode_step, messages, active_policy_id=active_policy_id)
            is_sample_step = int(episode_step) % int(sample_period_steps) == 0
            if known.policy_augmented and str(known.policy_replay_mode) == "communication":
                state_fn = getattr(sim, "_wam_stage1_slot_state", None)
                if state_fn is not None:
                    recorder.register_step(
                        episode_step,
                        state=state_fn(episode_step),
                        episode_id=episode_id,
                        fixed_dt=fixed_dt,
                        is_sample_step=global_step <= last_record_global_step and is_sample_step,
                    )
            elif global_step <= last_record_global_step and is_sample_step:
                if known.policy_augmented:
                    _register_policy_augmented_slot(
                        sim,
                        recorder,
                        episode_step,
                        episode_id=episode_id,
                        fixed_dt=fixed_dt,
                        bandwidth_ratio=float(known.policy_bandwidth_ratio),
                    )
                else:
                    _register_current_slot(sim, recorder, episode_step)
            recorder.flush_ready(episode_step)

            if global_step % known.print_every == 0:
                print(
                    f"global_step={global_step} episode={episode_id} "
                    f"episode_step={episode_step} written={recorder.written}",
                    flush=True,
                )

            if global_step >= last_observe_global_step:
                break

            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            global_step += 1
            if terminated or truncated:
                print(
                    f"episode={episode_id} ended at global_step={global_step} "
                    f"episode_step={int(getattr(sim, '_time_step', 0))} "
                    f"terminated={terminated} truncated={truncated}"
                    f"{'; stopping' if known.single_episode else '; resetting'}",
                    flush=True,
                )
                if known.single_episode:
                    break
                episode_id += 1
                if known.policy_augmented:
                    recorder.reset_episode()
                else:
                    recorder.reset_episode(episode_id=episode_id)
                env.reset(seed=_episode_seed(known.seed, episode_id))

        recorder.flush_all()
        print(f"Done. wrote {recorder.written} samples to {known.out_dir}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
