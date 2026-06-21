#!/usr/bin/env python3
"""Offline visualization for fixed-policy WAM comparison records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_uncertainty_plot(csv_path: Path, out_png: Path, *, metric: str) -> Optional[Path]:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} is not in {csv_path}; columns={list(df.columns)}")

    fig, ax1 = plt.subplots(figsize=(13, 5))
    for label, group in df.groupby("policy_label", sort=False):
        ax1.plot(group["step"], group[metric], label=str(label), linewidth=1.8)
    ax1.set_xlabel("step")
    ax1.set_ylabel(metric)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = ax1.twinx()
    for _, group in df.groupby("policy_label", sort=False):
        ax2.step(group["step"], group["num_selected"], where="post", alpha=0.16)
    ax2.set_ylabel("# selected collaborators")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def _write_summary(csv_path: Path, out_csv: Path, *, metric: str) -> Optional[Path]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    summary = (
        df.groupby(["policy_label", "policy_type"], dropna=False)
        .agg(
            steps=("step", "count"),
            mean_metric=(metric, "mean"),
            min_metric=(metric, "min"),
            max_metric=(metric, "max"),
            mean_selected=("num_selected", "mean"),
            mean_graph_veh_veh=("graph_veh_veh", "mean"),
            mean_graph_objects=("graph_objects", "mean"),
        )
        .reset_index()
        .sort_values("mean_metric")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize fixed-policy WAM comparison outputs.")
    parser.add_argument("--in-dir", type=Path, default=Path("outputs/wam_fixed_policy_compare"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--metric",
        choices=(
            "total_uncertainty",
            "motion_uncertainty",
            "coverage_uncertainty",
            "uncertainty_max",
            "uncertainty_mean",
        ),
        default="total_uncertainty",
    )
    parser.add_argument("--html", action="store_true", default=True)
    parser.add_argument("--no-html", dest="html", action="store_false")
    parser.add_argument("--png", action="store_true", default=True)
    parser.add_argument("--no-png", dest="png", action="store_false")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=110)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir or in_dir
    graph_jsonl = args.jsonl or in_dir / "fixed_policy_graph_timeline.jsonl"
    uncertainty_csv = args.csv_path or in_dir / "fixed_policy_uncertainty.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote = []
    if args.png:
        path = _write_uncertainty_plot(
            uncertainty_csv,
            out_dir / f"fixed_policy_{args.metric}.png",
            metric=args.metric,
        )
        if path is not None:
            wrote.append(path)
    summary = _write_summary(
        uncertainty_csv,
        out_dir / f"fixed_policy_{args.metric}_summary.csv",
        metric=args.metric,
    )
    if summary is not None:
        wrote.append(summary)

    if args.html:
        from car_dreamer.toolkit.wam import load_records_jsonl, write_graph_timeline_html

        records = load_records_jsonl(graph_jsonl)
        if not records:
            raise SystemExit(f"no graph records found in {graph_jsonl}")
        html = write_graph_timeline_html(
            records,
            out_dir / "fixed_policy_graph_timeline.html",
            title="Fixed-policy WAM graph comparison",
            fps=float(args.fps),
            dpi=int(args.dpi),
        )
        wrote.append(Path(html))

    for path in wrote:
        print(f"Wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
