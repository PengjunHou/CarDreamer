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
from typing import List, Optional, Tuple

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


def record_step(sim, out_path: Path) -> bool:
    """Extract + append one step's graph record. Returns True if a graph was recorded."""
    from car_dreamer.toolkit.wam import append_record_jsonl, hetero_graph_to_record

    graph = getattr(sim, "_wam_graph", None)
    if graph is None:
        return False
    label, policy_id = active_policy_label(sim)
    record = hetero_graph_to_record(
        graph,
        step=int(getattr(sim, "_time_step", 0)),
        policy_label=label,
        policy_id=policy_id,
        extra={"episode": int(getattr(sim, "_episode_count", 0) or 0)},
    )
    append_record_jsonl(record, out_path)
    return True


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Record the per-step WAM cooperative graph to JSONL.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", default="outputs/graph_timeline.jsonl", help="output JSONL path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-display", dest="display", action="store_false", default=False)
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
    env, _ = build_env(known.task, env_args)
    sim = env.unwrapped

    env.reset(seed=known.seed)
    recorded = 0
    try:
        # Single-episode recording: run until the episode ends (or the --steps cap), then stop.
        for step in range(known.steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if record_step(sim, out_path):
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
