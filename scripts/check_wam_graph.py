#!/usr/bin/env python3
"""Live sanity check for the WAM policy-conditioned hetero graph (Design step 4).

Runs a task directly, steps it, and prints the per-step hetero-graph node/edge counts
(read from the returned ``info``) alongside whether cooperative perception was triggered.
With ``--embed`` it also runs the §5 initial embedding + §9 HGT encoder on the current
graph and prints the resulting ``H_t`` node-embedding shapes.

Example:
    python scripts/check_wam_graph.py --task carla_group_right_turn_auto --carla-port 2000 \
        --steps 200 --print-every 10 --embed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

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
    parser = argparse.ArgumentParser(description="Print per-step WAM hetero-graph stats.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--embed", action="store_true", default=False,
                        help="also run the §5 embedding + §9 HGT encoder and print H_t shapes")
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    known, passthrough = parser.parse_known_args()
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


_GRAPH_KEYS = (
    "wam_graph_num_vehicle_nodes",
    "wam_graph_num_observation_nodes",
    "wam_graph_num_object_nodes",
    "wam_graph_num_veh_obs_edges",
    "wam_graph_num_obs_obj_edges",
    "wam_graph_num_coop_edges",
)


def _print_step(step: int, info: dict, sim, embed: bool) -> None:
    counts = {key.replace("wam_graph_num_", ""): info.get(key) for key in _GRAPH_KEYS}
    line = (
        f"step={step:4d} triggered={bool(info.get('wam_coop_triggered'))} "
        f"selected={info.get('wam_policy_selected_vehicle_ids')} "
        f"invisible={info.get('wam_invisible_notable_object_ids')} graph={counts}"
    )
    if embed:
        graph = getattr(sim, "_wam_graph", None)
        embeddings = getattr(sim, "_wam_graph_embeddings", None)
        if embeddings is not None:
            shapes = {k: tuple(v.shape) for k, v in embeddings.items()}
            line += f" H_t={shapes}"
        elif graph is not None:
            line += " H_t=<embed_on but not produced>"
    print(line, flush=True)


def main() -> int:
    known, passthrough = parse_args()
    _setup_carla_pythonapi()

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        f"--env.wam.build_graph=True",
        f"--env.wam.graph.embed={bool(known.embed)}",
        *passthrough,
    ]
    env, _ = build_env(known.task, env_args)
    sim = env.unwrapped

    try:
        _, reset_info = env.reset()
        _print_step(0, reset_info, sim, known.embed)
        for step in range(1, known.steps + 1):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if step % known.print_every == 0:
                _print_step(step, info, sim, known.embed)
            if terminated or truncated:
                print(f"episode ended at step={step}: terminated={terminated} truncated={truncated}", flush=True)
                _, reset_info = env.reset()
                _print_step(step, reset_info, sim, known.embed)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
