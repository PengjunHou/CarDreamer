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


def _register_current_graph(sim, recorder, step: int) -> bool:
    """Slide the current request graph into the window with the ego pose."""
    graph = getattr(sim, "_wam_graph", None)
    ego = getattr(sim, "ego", None)
    if graph is None or ego is None:
        return False
    transform = ego.get_transform()
    ego_pose = (float(transform.location.x), float(transform.location.y), float(transform.rotation.yaw))
    recorder.register(step, graph=graph, ego_pose=ego_pose)
    return True


def main() -> int:
    known, passthrough = parse_args()
    if known.steps <= 0:
        raise ValueError("--steps must be positive")
    if known.future_horizon_s <= 0:
        raise ValueError("--future-horizon-s must be positive")

    _setup_carla_pythonapi()

    import car_dreamer
    from car_dreamer.toolkit.wam import WAMStage1DataRecorder, wam_stage1_configs_from_env

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        "--env.wam.build_graph=True",
        *passthrough,
    ]
    env, config = build_env(known.task, env_args)
    sim = env.unwrapped
    perc_cfg, stage1_cfg = wam_stage1_configs_from_env(config)
    known.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env.reset(seed=known.seed)
        fixed_dt = _fixed_dt(sim, config)
        recorder = WAMStage1DataRecorder(
            known.out_dir,
            fixed_dt=fixed_dt,
            horizon_s=known.future_horizon_s,
            samples=int(perc_cfg.traj_samples),
            history_window=int(stage1_cfg.history_window),
        )
        print(
            f"Recording {known.steps} steps to {known.out_dir} "
            f"(horizon={known.future_horizon_s:.1f}s/{perc_cfg.traj_samples} samples, "
            f"window={stage1_cfg.history_window + 1}, extra_steps={recorder.horizon_steps})",
            flush=True,
        )

        last_record_step = int(known.steps) - 1
        last_observe_step = last_record_step + int(recorder.horizon_steps)

        while int(getattr(sim, "_time_step", 0)) <= last_observe_step:
            current_step = int(getattr(sim, "_time_step", 0))
            recorder.observe(current_step, _actor_snapshots(sim))
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
