#!/usr/bin/env python3
"""Run a CarDreamer task environment directly, without any DreamerV3 training.

This mirrors how ``dreamerv3/my_train.py`` builds the environment
(``car_dreamer.create_task``) but skips everything after it: no gym wrappers,
no replay buffer, no agent, no training loop. It just resets the env and steps
it with sampled actions so you can watch the scene in the web visualization
(http://localhost:<carla_port + 7000>/).

The ego vehicle in the right-turn-auto task is driven by CARLA's BasicAgent, so
the sampled action is ignored -- it only exists to satisfy the Gym interface.

Examples
--------
    # Watch the default task for 500 steps, then hold the viz for 5 min
    python scripts/run_env.py --carla-port 2000 --steps 500 --hold-seconds 300

    # Forward arbitrary env overrides straight to create_task
    python scripts/run_env.py -- --env.num_group_vehs=2 --env.num_background_vehs=4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_carla_pythonapi() -> None:
    """Put the CARLA PythonAPI on sys.path (same as my_train.py)."""
    carla_root = os.environ.get("CARLA_ROOT", "/home/peh324/carla_simulator")
    for sub in ("PythonAPI", "PythonAPI/carla"):
        path = os.path.join(carla_root, sub)
        if path not in sys.path:
            sys.path.append(path)


def build_env(task: str, env_args: List[str]):
    """Build the gym env exactly like create_task (common.yaml + tasks.yaml + CLI overrides)."""
    import gymnasium as gym

    import car_dreamer
    from car_dreamer import toolkit

    config = car_dreamer.load_task_configs(task)
    config, _ = toolkit.Flags(config).parse_known(env_args)
    env = gym.make(config.env.name, config=config.env)
    return env, config


def _live_actor_counts(sim) -> Tuple[int, int]:
    world = getattr(getattr(sim, "_world", None), "_world", None)
    if world is None:
        return 0, 0
    actors = world.get_actors()
    return len(list(actors.filter("vehicle.*"))), len(list(actors.filter("walker.pedestrian.*")))


def print_scene(env, tag: str) -> None:
    sim = env.unwrapped
    ego = getattr(sim, "ego", None)
    ego_id = int(ego.id) if ego is not None else None
    group_vehs = list(getattr(sim, "group_vehs", []))
    in_flight = len(getattr(sim, "_in_flight", []))
    received = getattr(sim, "_received", {})
    ego_msgs = len(received.get(ego_id, [])) if ego_id is not None else 0
    vehicles, walkers = _live_actor_counts(sim)

    print(
        f"[{tag}] step={getattr(sim, '_time_step', '?')} "
        f"ego={ego_id} group_vehs={len(group_vehs)} {[int(v.id) for v in group_vehs]} "
        f"| comm in_flight={in_flight} ego_received={ego_msgs} "
        f"| live vehicles={vehicles} walkers={walkers}",
        flush=True,
    )


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run a CarDreamer task env directly (no training).",
    )
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Seconds to sleep between steps (slows things down so you can watch).")
    parser.add_argument("--hold-seconds", type=float, default=0.0,
                        help="Keep stepping after the main run so the web viz stays live.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-display", dest="display", action="store_false", default=True,
                        help="Disable the web visualization monitor.")
    # Anything after `--` (or any unrecognized --env.* flag) is forwarded to create_task.
    known, passthrough = parser.parse_known_args()
    # argparse keeps a literal `--` separator in the remainder; drop it.
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def main() -> int:
    known, passthrough = parse_args()
    if known.steps < 0:
        raise ValueError("--steps must be non-negative")
    if known.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if known.sleep < 0 or known.hold_seconds < 0:
        raise ValueError("--sleep and --hold-seconds must be non-negative")

    _setup_carla_pythonapi()

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        *passthrough,
    ]

    env, _ = build_env(known.task, env_args)
    viz_url = f"http://localhost:{known.carla_port + 7000}/"

    try:
        if known.display:
            print(f"Web visualization: {viz_url}", flush=True)

        env.reset(seed=known.seed)
        print_scene(env, "after reset")

        for step in range(known.steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if step % known.print_every == 0:
                print_scene(env, f"step {step}")
            if terminated or truncated:
                print(f"episode ended: terminated={terminated} truncated={truncated}", flush=True)
                env.reset(seed=known.seed)
                print_scene(env, "after auto-reset")
            if known.sleep:
                time.sleep(known.sleep)

        if known.hold_seconds:
            end = time.monotonic() + known.hold_seconds
            print(f"Holding for {known.hold_seconds:.1f}s so the viz stays live. Ctrl-C to stop.", flush=True)
            hold_step = 0
            while time.monotonic() < end:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if hold_step % known.print_every == 0:
                    print_scene(env, f"hold {hold_step}")
                if terminated or truncated:
                    env.reset(seed=known.seed)
                hold_step += 1
                if known.sleep:
                    time.sleep(known.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
