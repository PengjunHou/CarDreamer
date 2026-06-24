#!/usr/bin/env python3
"""One-shot policy replay comparison: record once, evaluate many policies offline.

This pipeline implements the fair comparison workflow:

1. run one CARLA rollout and record policy-augmented samples from the same states;
2. evaluate a Stage-1 checkpoint on every recorded counterfactual policy sample;
3. write a paired summary where deltas are matched by ``episode_id, step``.

Extra ``--env.*`` overrides can be passed after ``--`` and are forwarded to the
recording step.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Record one rollout and replay WAM policies offline.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Stage-1 checkpoint to evaluate")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wam_policy_replay"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/wam_policy_replay"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="total_uncertainty",
                        choices=("uncertainty", "motion_uncertainty", "coverage_uncertainty",
                                 "total_uncertainty", "mean_uncertainty", "ade", "fde",
                                 "motion_uncertainty_notable", "total_uncertainty_notable",
                                 "ade_notable", "fde_notable"))
    parser.add_argument("--baseline", default="ego_only")
    parser.add_argument("--future-horizon-s", type=float, default=None)
    parser.add_argument("--policy-sampler", choices=("request_all", "random_duration"), default="request_all")
    parser.add_argument("--policy-bandwidth-ratio", type=float, default=1.0)
    parser.add_argument(
        "--policy-replay-mode",
        choices=("instant", "communication"),
        default="communication",
        help="offline policy replay mode; communication simulates V2V sender/receiver delays",
    )
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="optional evaluation sample limit")
    parser.add_argument("--display", dest="display", action="store_true", default=False)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="remove data-dir before recording")
    parser.add_argument("--skip-record", action="store_true", help="reuse existing data-dir")
    parser.add_argument("--skip-eval", action="store_true", help="reuse existing uncertainty CSV")
    parser.add_argument("--skip-viz", action="store_true", help="skip summary/HTML generation")
    parser.add_argument(
        "--multi-episode",
        action="store_true",
        help="allow recorder to reset and continue after episode termination; default is one episode only",
    )
    known, passthrough = parser.parse_known_args()
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def _run(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _prepare_data_dir(path: Path, *, overwrite: bool, skip_record: bool) -> None:
    if skip_record:
        if not path.exists():
            raise FileNotFoundError(f"--skip-record requested but data-dir does not exist: {path}")
        return
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"data-dir is not empty: {path}\n"
                "Use --overwrite to replace it, or --skip-record to reuse it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args, passthrough = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    metric = "uncertainty" if str(args.metric) == "mean_uncertainty" else str(args.metric)

    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "uncertainty_by_policy.csv"
    summary_path = out_dir / "policy_uncertainty_summary.csv"
    html_path = out_dir / "policy_uncertainty.html"

    _prepare_data_dir(data_dir, overwrite=bool(args.overwrite), skip_record=bool(args.skip_record))

    if not args.skip_record:
        record_cmd = [
            sys.executable,
            "scripts/record_wam_stage1_data.py",
            "--task",
            str(args.task),
            "--carla-port",
            str(args.carla_port),
            "--steps",
            str(args.steps),
            "--out-dir",
            str(data_dir),
            "--policy-augmented",
            "--policy-sampler",
            str(args.policy_sampler),
            "--policy-bandwidth-ratio",
            str(args.policy_bandwidth_ratio),
            "--policy-replay-mode",
            str(args.policy_replay_mode),
            "--print-every",
            str(args.print_every),
            "--seed",
            str(args.seed),
        ]
        if args.future_horizon_s is not None:
            record_cmd.extend(["--future-horizon-s", str(args.future_horizon_s)])
        if not args.multi_episode:
            record_cmd.append("--single-episode")
        record_cmd.append("--display" if args.display else "--no-display")
        record_cmd.extend(passthrough)
        _run(record_cmd)

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_wam_stage1_uncertainty.py",
            "--checkpoint",
            str(args.checkpoint),
            "--data-dir",
            str(data_dir),
            "--task",
            str(args.task),
            "--out-dir",
            str(out_dir),
            "--csv-name",
            csv_path.name,
            "--device",
            str(args.device),
        ]
        if args.limit is not None:
            eval_cmd.extend(["--limit", str(args.limit)])
        _run(eval_cmd)

    if not args.skip_viz:
        viz_cmd = [
            sys.executable,
            "scripts/visualize_wam_stage1_policy_uncertainty.py",
            "--csv",
            str(csv_path),
            "--out-html",
            str(html_path),
            "--out-summary",
            str(summary_path),
            "--metric",
            metric,
            "--baseline",
            str(args.baseline),
        ]
        _run(viz_cmd)

    print(f"Data samples : {data_dir}", flush=True)
    print(f"Rows CSV     : {csv_path}", flush=True)
    print(f"Summary CSV  : {summary_path}", flush=True)
    if not args.skip_viz:
        print(f"HTML         : {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
