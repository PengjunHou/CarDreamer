#!/usr/bin/env python3
"""Run full online episodes under a WAM policy sampler and log per-step uncertainty / bandwidth /
Lyapunov queue state, for the P2-vs-ego-only comparison (Phase 3).

Runs ``--rounds`` episodes with ``policy_sampler_mode=<mode>`` (``lyapunov`` = per-frame (P2)
selection over the 200-candidate budget; ``local_only`` = ego-only baseline that never cooperates),
writing one ``trace_r{r}.csv`` per round plus a ``summary.json`` (round-averaged metrics). Per-step
metrics come from the env ``info`` dict (``wam_total_uncertainty``, ``wam_allocated_bandwidth``,
``wam_lyap_z``, ``wam_lyap_total_backlog``, selected vehicles). The online scheduler path is written
but not previously live-verified, so failures degrade gracefully.

Example
-------
    python scripts/run_wam_lyapunov_online_episode.py --mode lyapunov --rounds 5 \
        --checkpoint outputs/wam_stage1_chunk_v2/stage1_step2000.pt \
        --out-dir outputs/wam_lyapunov_online
    python scripts/run_wam_lyapunov_online_episode.py --mode local_only --rounds 5 \
        --checkpoint outputs/wam_stage1_chunk_v2/stage1_step2000.pt \
        --out-dir outputs/wam_lyapunov_online
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

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


# Column names align with the offline lyapunov_trace.csv so the same analyzers/plots apply
# (z, total_backlog, allocated_bandwidth, step) plus the online-only uncertainty breakdown.
TRACE_FIELDS = [
    "step", "realized_u", "total_uncertainty", "motion_uncertainty", "coverage_uncertainty",
    "allocated_bandwidth", "num_selected", "z", "total_backlog",
]


def _row_from_info(step: int, info: Dict) -> Dict[str, float]:
    u = float(info.get("wam_total_uncertainty", 0.0))
    return {
        "step": int(step),
        "realized_u": u,
        "total_uncertainty": u,
        "motion_uncertainty": float(info.get("wam_motion_uncertainty", 0.0)),
        "coverage_uncertainty": float(info.get("wam_coverage_uncertainty", 0.0)),
        "allocated_bandwidth": float(info.get("wam_allocated_bandwidth", 0.0)),
        "num_selected": len(info.get("wam_policy_selected_vehicle_ids", []) or []),
        "z": float(info.get("wam_lyap_z", 0.0)),
        "total_backlog": float(info.get("wam_lyap_total_backlog", 0.0)),
    }


def _episode_summary(rows: List[Dict[str, float]]) -> Dict[str, float]:
    n = max(len(rows), 1)
    def mean(key):
        return sum(r[key] for r in rows) / n
    z_final = rows[-1]["z"] if rows else 0.0
    return {
        "steps": int(len(rows)),
        "time_avg_uncertainty": mean("total_uncertainty"),
        "time_avg_motion_uncertainty": mean("motion_uncertainty"),
        "time_avg_coverage_uncertainty": mean("coverage_uncertainty"),
        "time_avg_bandwidth": mean("allocated_bandwidth"),
        "coop_step_rate": sum(1 for r in rows if r["num_selected"] > 0) / n,
        "mean_backlog": mean("total_backlog"),
        "z_over_t": z_final / n,
    }


def run_round(env, sim, max_steps: int, seed: int) -> List[Dict[str, float]]:
    env.reset(seed=seed)
    rows: List[Dict[str, float]] = []
    step = 0
    while True:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        rows.append(_row_from_info(step, info))
        step += 1
        if terminated or truncated or (max_steps and step >= max_steps):
            break
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Online WAM episode runner (P2 vs ego-only).")
    p.add_argument("--task", default="carla_group_right_turn_auto")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--mode", choices=("lyapunov", "local_only", "dreamer"), required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--checkpoint", type=Path, default=None, help="Stage-1 U_phi checkpoint")
    p.add_argument("--uwm-checkpoint", type=Path, default=None, help="Stage-2 UWM (Phase 5 proposer)")
    p.add_argument("--dreamer-world-model", type=Path, default=None, help="P2 world-model checkpoint (--mode dreamer)")
    p.add_argument("--dreamer-actor-critic", type=Path, default=None, help="P3 actor-critic checkpoint (--mode dreamer)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/wam_lyapunov_online"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--client-timeout-s", type=float, default=40.0)
    known, passthrough = p.parse_known_args()

    _setup_carla_pythonapi()

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.world.client_timeout_s={known.client_timeout_s}",
        "--env.display.enable=False",
        "--env.wam.build_graph=True",
        f"--env.wam.policy_sampler_mode={known.mode}",
        # queue continuity across sub-actions (paper semantics)
        "--env.communication.flush_old_policy_queue=True",
        "--env.communication.allow_cross_policy_messages=True",
        "--env.communication.sensor_period_s=0.1",
        *[a for a in passthrough if a != "--"],
    ]
    # Use the same U_phi checkpoint for BOTH modes when provided, so the ego-only (local_only)
    # baseline differs from lyapunov *only* in whether it cooperates -- not in the predictor. This
    # makes the Phase-3 delta a clean cooperation effect. Fall back to the rule predictor only when
    # no checkpoint is given.
    if known.checkpoint is not None:
        env_args += [
            "--env.wam.predictor_mode=checkpoint",
            f"--env.wam.predictor_checkpoint={known.checkpoint}",
        ]
    else:
        env_args += ["--env.wam.predictor_mode=rule"]
    if known.uwm_checkpoint is not None:
        env_args += [f"--env.wam.uwm_checkpoint={known.uwm_checkpoint}"]
    if known.mode == "dreamer":
        if known.dreamer_world_model is None or known.dreamer_actor_critic is None:
            raise SystemExit("--mode dreamer requires --dreamer-world-model and --dreamer-actor-critic")
        env_args += [
            f"--env.wam.dreamer_world_model={known.dreamer_world_model}",
            f"--env.wam.dreamer_actor_critic={known.dreamer_actor_critic}",
        ]

    out_dir = known.out_dir / known.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    env, _ = build_env(known.task, env_args)
    sim = env.unwrapped

    per_round: List[Dict[str, float]] = []
    try:
        for r in range(known.rounds):
            rows = run_round(env, sim, known.max_episode_steps, seed=known.seed + r)
            trace_path = out_dir / f"trace_r{r}.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
                w.writeheader()
                w.writerows(rows)
            s = _episode_summary(rows)
            per_round.append(s)
            print(f"[{known.mode}] round {r}: steps={s['steps']} "
                  f"U={s['time_avg_uncertainty']:.4f} bw={s['time_avg_bandwidth']:.4f} "
                  f"coop_rate={s['coop_step_rate']:.3f} backlog={s['mean_backlog']:.3g} "
                  f"Z/T={s['z_over_t']:.4g} -> {trace_path}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    # round-averaged summary
    def agg(key):
        vals = [s[key] for s in per_round]
        m = sum(vals) / max(len(vals), 1)
        var = sum((v - m) ** 2 for v in vals) / max(len(vals), 1)
        return {"mean": m, "std": var ** 0.5}

    summary = {
        "mode": known.mode,
        "rounds": len(per_round),
        "metrics": {k: agg(k) for k in (
            "time_avg_uncertainty", "time_avg_motion_uncertainty", "time_avg_coverage_uncertainty",
            "time_avg_bandwidth", "coop_step_rate", "mean_backlog", "z_over_t", "steps")},
        "per_round": per_round,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[{known.mode}] wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    # The env spins up a non-daemon Flask monitor thread that keeps the process alive after main()
    # returns (summary.json + trace CSVs are already flushed to disk by then). A plain SystemExit
    # would hang on that thread, so batch drivers (sweep_wam_bandwidth.sh) would block until their
    # per-run `timeout` fires. os._exit forces immediate termination.
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(_rc))
