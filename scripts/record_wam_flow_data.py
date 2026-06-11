#!/usr/bin/env python3
"""Record WAM Stage-2 training data from the live CARLA env (WAM Design §16.2 data side).

Steps a task, and at each step pairs the policy-conditioned hetero graph (``sim._wam_graph`` +
``sim._wam_policy``) with the GT future object states once the horizon elapses, writing self-contained
``.pt`` samples consumed by ``scripts/train_wam_stage2.py`` / ``WAMFlowDataset``.

Needs CARLA running on ``--carla-port``. Example:
    python scripts/record_wam_flow_data.py --task carla_group_right_turn_auto --carla-port 2000 \
        --steps 400 --out-dir data/wam_flow
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
    parser = argparse.ArgumentParser(description="Record WAM Stage-2 flow-matching training samples.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("data/wam_flow"))
    parser.add_argument("--future-horizon-s", type=float, default=3.0)
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


def _candidate_ids(sim) -> List[int]:
    participants = getattr(sim, "coop_participant_ids", None)
    if participants:
        return sorted(int(i) for i in participants)
    return sorted(int(getattr(v, "id", -1)) for v in getattr(sim, "group_vehs", []) if v is not None)


def _notable_object_ids(sim) -> List[int]:
    records = getattr(sim, "_wam_notable_records", None) or []
    return [int(r.object_state.actor_id) for r in records]


def _register_current_graph(sim, recorder, step: int) -> bool:
    """Register the request-vehicle graph + candidate ordering + notable ids with the recorder."""
    graph = getattr(sim, "_wam_graph", None)
    if graph is None or getattr(sim, "ego", None) is None:
        return False
    recorder.register(
        step,
        graph=graph,
        candidate_ids=_candidate_ids(sim),
        notable_object_ids=_notable_object_ids(sim),
    )
    return True


def _observe_request_bev(sim, recorder, step: int) -> None:
    """Rasterize the request (ego) vehicle's visibility-aware B^sem from its visible objects + route."""
    ego = getattr(sim, "ego", None)
    if ego is None:
        return
    tf = ego.get_transform()
    ego_pose = (float(tf.location.x), float(tf.location.y), float(tf.rotation.yaw))
    objects = list(getattr(sim, "_wam_object_states", []))
    visible = [s for s in objects if bool(getattr(s, "visible_to_ego", False))]
    route_xy = sim._wam_route_xy() if hasattr(sim, "_wam_route_xy") else ()
    recorder.observe_bev(step, ego_pose, visible, route_xy=route_xy)


def main() -> int:
    known, passthrough = parse_args()
    if known.steps <= 0:
        raise ValueError("--steps must be positive")
    if known.future_horizon_s <= 0:
        raise ValueError("--future-horizon-s must be positive")

    _setup_carla_pythonapi()

    import car_dreamer
    from car_dreamer.toolkit.wam import BevSpec, WAMFlowDataRecorder, wam_configs_from_env

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        "--env.wam.build_graph=True",
        *passthrough,
    ]
    env, config = build_env(known.task, env_args)
    sim = env.unwrapped
    graph_cfg, flow_cfg, _ = wam_configs_from_env(config)
    bev_range_m = float(getattr(getattr(getattr(config.env, "wam", None), "graph", None), "bev_range_m", 50.0))
    bev_spec = BevSpec(size=int(graph_cfg.bev_size), range_m=bev_range_m)
    known.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env.reset(seed=known.seed)
        fixed_dt = _fixed_dt(sim, config)
        recorder = WAMFlowDataRecorder(
            known.out_dir,
            fixed_dt=fixed_dt,
            horizon_s=known.future_horizon_s,
            samples=int(flow_cfg.horizon),
            max_members=int(flow_cfg.max_members),
            num_formats=int(flow_cfg.num_formats),
            history_window=int(flow_cfg.history_window),
            bev_spec=bev_spec,
            enable_bev=bool(flow_cfg.enable_bev),
        )
        print(
            f"Recording {known.steps} steps to {known.out_dir} "
            f"(horizon={known.future_horizon_s:.1f}s/{flow_cfg.horizon} samples, "
            f"extra_steps={recorder.horizon_steps})",
            flush=True,
        )

        last_record_step = int(known.steps) - 1
        last_observe_step = last_record_step + int(recorder.horizon_steps)

        while int(getattr(sim, "_time_step", 0)) <= last_observe_step:
            current_step = int(getattr(sim, "_time_step", 0))
            recorder.observe_policy(current_step, getattr(sim, "_wam_policy", None))
            _observe_request_bev(sim, recorder, current_step)
            if current_step <= last_record_step:
                _register_current_graph(sim, recorder, current_step)
            recorder.flush_ready(current_step)

            if current_step % known.print_every == 0:
                print(f"step={current_step} written={recorder.written}", flush=True)

            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset(seed=known.seed)

        recorder.flush_all()
        print(f"Done. wrote {recorder.written} samples to {known.out_dir}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
