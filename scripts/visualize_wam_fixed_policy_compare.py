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


SUMMARY_METRICS = (
    "motion_uncertainty",
    "coverage_uncertainty",
    "total_uncertainty",
    "route_coverage_quality_mean",
    "poor_coverage_risk_mean",
    "uncertainty_max",
    "uncertainty_mean",
    "checkpoint_window_slots",
    "checkpoint_window_has_v2v_graph",
    "checkpoint_window_v2v_slots",
    "checkpoint_window_v2v_slot_rate",
    "checkpoint_final_has_v2v_graph",
    "checkpoint_final_graph_objects",
    "checkpoint_final_ego_visible_objects",
    "checkpoint_final_collab_only_objects",
    "checkpoint_final_collab_object_ratio",
    "checkpoint_window_union_objects",
    "checkpoint_window_union_ego_visible_objects",
    "checkpoint_window_union_collab_only_objects",
    "checkpoint_window_union_collab_object_ratio",
    "checkpoint_prediction_query_objects",
)

COMPACT_SUMMARY_COLUMNS = (
    "policy_label",
    "policy_type",
    "steps",
    "mean_selected",
    "total_uncertainty_mean",
    "total_uncertainty_delta_vs_ego_only",
    "motion_uncertainty_mean",
    "motion_uncertainty_delta_vs_ego_only",
    "coverage_uncertainty_mean",
    "coverage_uncertainty_delta_vs_ego_only",
    "checkpoint_window_has_v2v_graph_mean",
    "checkpoint_window_v2v_prediction_steps",
    "checkpoint_window_v2v_slots_mean",
    "checkpoint_window_v2v_slots_total",
    "checkpoint_window_v2v_slot_rate_mean",
    "checkpoint_final_has_v2v_graph_mean",
    "checkpoint_final_v2v_prediction_steps",
    "checkpoint_final_ego_visible_objects_mean",
    "checkpoint_final_collab_only_objects_mean",
    "checkpoint_final_collab_object_ratio_mean",
    "checkpoint_window_union_ego_visible_objects_mean",
    "checkpoint_window_union_collab_only_objects_mean",
    "checkpoint_window_union_collab_object_ratio_mean",
    "checkpoint_prediction_query_objects_mean",
    "mean_graph_objects",
    "mean_graph_veh_veh",
)


def _compact_summary_path(summary_csv: Path) -> Path:
    suffix = "_summary"
    stem = summary_csv.stem
    compact_stem = f"{stem[:-len(suffix)]}_compact_summary" if stem.endswith(suffix) else f"{stem}_compact"
    return summary_csv.with_name(f"{compact_stem}{summary_csv.suffix}")


def _write_compact_summary(summary, out_csv: Path) -> Optional[Path]:
    columns = [name for name in COMPACT_SUMMARY_COLUMNS if name in summary.columns]
    if not columns:
        return None
    compact = summary.loc[:, columns].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(out_csv, index=False)
    return out_csv


def _write_summary(csv_path: Path, out_csv: Path, *, metric: str, baseline: str = "ego_only") -> Optional[Path]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} is not in {csv_path}; columns={list(df.columns)}")

    group_cols = ["policy_label", "policy_type"]
    grouped = df.groupby(group_cols, dropna=False)
    summary = (
        grouped.size()
        .rename("steps")
        .reset_index()
        .merge(grouped["num_selected"].mean().rename("mean_selected").reset_index(), on=group_cols, how="left")
        .merge(grouped["graph_veh_veh"].mean().rename("mean_graph_veh_veh").reset_index(), on=group_cols, how="left")
        .merge(grouped["graph_objects"].mean().rename("mean_graph_objects").reset_index(), on=group_cols, how="left")
    )

    available = [name for name in SUMMARY_METRICS if name in df.columns]
    baseline_mask = (df["policy_label"] == baseline) | (df["policy_type"] == baseline)
    baseline_means = df.loc[baseline_mask, available].mean(numeric_only=True).to_dict()
    for name in available:
        agg = (
            grouped[name]
            .agg(["mean", "std", "min", "max"])
            .rename(
                columns={
                    "mean": f"{name}_mean",
                    "std": f"{name}_std",
                    "min": f"{name}_min",
                    "max": f"{name}_max",
                }
            )
            .reset_index()
        )
        summary = summary.merge(agg, on=group_cols, how="left")
        if name in baseline_means:
            summary[f"{name}_delta_vs_{baseline}"] = summary[f"{name}_mean"] - float(baseline_means[name])

    for name, out_name in (
        ("checkpoint_window_has_v2v_graph", "checkpoint_window_v2v_prediction_steps"),
        ("checkpoint_final_has_v2v_graph", "checkpoint_final_v2v_prediction_steps"),
        ("checkpoint_window_v2v_slots", "checkpoint_window_v2v_slots_total"),
    ):
        if name in df.columns:
            summary = summary.merge(grouped[name].sum().rename(out_name).reset_index(), on=group_cols, how="left")

    summary["mean_metric"] = summary[f"{metric}_mean"] if f"{metric}_mean" in summary.columns else grouped[metric].mean().values
    summary["min_metric"] = summary[f"{metric}_min"] if f"{metric}_min" in summary.columns else grouped[metric].min().values
    summary["max_metric"] = summary[f"{metric}_max"] if f"{metric}_max" in summary.columns else grouped[metric].max().values
    for col in summary.columns:
        if col.endswith("_std"):
            summary[col] = summary[col].fillna(0.0)
    summary = summary.sort_values(["mean_metric", "policy_label"], ascending=[True, True])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    _write_compact_summary(summary, _compact_summary_path(out_csv))
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
    parser.add_argument("--baseline", default="ego_only", help="baseline policy label or policy_type for delta columns")
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
        baseline=args.baseline,
    )
    if summary is not None:
        wrote.append(summary)
        compact = _compact_summary_path(summary)
        if compact.exists():
            wrote.append(compact)

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
