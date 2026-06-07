#!/usr/bin/env python3
"""Inspect a CARLA task scene without starting DreamerV3 training."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _actor_ids(actors: Iterable[object]) -> list[int]:
    return [int(actor.id) for actor in actors if actor is not None]


def _get_live_actors(env):
    world = getattr(getattr(env, "_world", None), "_world", None)
    if world is None:
        return [], [], []
    actors = world.get_actors()
    vehicles = list(actors.filter("vehicle.*"))
    walkers = list(actors.filter("walker.pedestrian.*"))
    controllers = list(actors.filter("controller.ai.walker"))
    return vehicles, walkers, controllers


def print_scene(env, tag: str) -> None:
    vehicles, walkers, controllers = _get_live_actors(env)
    ego = getattr(env, "ego", None)
    group_vehs = list(getattr(env, "group_vehs", []))
    background_vehs = list(getattr(env, "background_vehs", []))
    pedestrians = list(getattr(env, "pedestrians", []))
    actor_flow = list(getattr(env, "actor_flow", []))

    print(f"\n[{tag}]", flush=True)
    print("ego:", int(ego.id) if ego is not None else None, flush=True)
    print("group vehicles:", len(group_vehs), _actor_ids(group_vehs), flush=True)
    print("background vehicles:", len(background_vehs), _actor_ids(background_vehs), flush=True)
    print("pedestrians:", len(pedestrians), _actor_ids(pedestrians), flush=True)
    print("actor_flow vehicles:", len(actor_flow), _actor_ids(actor_flow), flush=True)
    print("live CARLA vehicles:", len(vehicles), flush=True)
    print("live CARLA walkers:", len(walkers), flush=True)
    print("walker controllers:", len(controllers), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CARLA task for visual scene inspection and actor counts."
    )
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep stepping after the main run so the web visualization remains live.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.sleep < 0:
        raise ValueError("--sleep must be non-negative")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds must be non-negative")

    env_args = [
        f"--env.world.carla_port={args.carla_port}",
        f"--env.display.enable={bool(args.display)}",
    ]
    import car_dreamer

    env, _ = car_dreamer.create_task(args.task, env_args)
    url = f"http://localhost:{args.carla_port + 7000}/"

    try:
        if args.display:
            print(f"Starting web visualization at http://localhost:{args.carla_port + 7000}/", flush=True)
        env.reset(seed=args.seed)
        print_scene(env, "after reset")
        if args.display:
            print(f"\nOpen visualization: {url}", flush=True)

        for step in range(args.steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if step % args.print_every == 0:
                print_scene(env, f"step {step}")
            if terminated or truncated:
                print(f"\nepisode ended: terminated={terminated} truncated={truncated}")
                break
            if args.sleep:
                time.sleep(args.sleep)

        if args.hold_seconds:
            end_time = time.monotonic() + args.hold_seconds
            print(
                f"\nHolding visualization for {args.hold_seconds:.1f}s. Press Ctrl-C to stop.",
                flush=True,
            )
            hold_step = 0
            while time.monotonic() < end_time:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if hold_step % args.print_every == 0:
                    print_scene(env, f"hold step {hold_step}")
                if terminated or truncated:
                    print(f"\nepisode ended during hold: terminated={terminated} truncated={truncated}", flush=True)
                    break
                hold_step += 1
                if args.sleep:
                    time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
