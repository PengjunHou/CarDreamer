"""Offline visualization helpers for Stage-1 policy uncertainty CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

COMM_METRICS = (
    "comm_window_slots",
    "comm_window_v2v_slots",
    "comm_window_v2v_slot_rate",
    "comm_window_has_v2v_graph",
    "comm_generated_messages",
    "comm_received_messages_by_prediction_step",
)
# Per-object window presence (frames per union object): total (any source) vs ego-only sensing.
# Their gap is the extra frames-per-object cooperation contributed inside the window.
PRESENCE_METRICS = (
    "object_window_presence_mean",
    "object_window_presence_ego_mean",
)
# Task-focused (GT-notable-only) variants emitted by evaluate_stage1_uncertainty_rows.
NOTABLE_METRICS = (
    "motion_uncertainty_notable",
    "total_uncertainty_notable",
    "ade_notable",
    "fde_notable",
)
# [0, 1]-normalized (saturated) uncertainty variants: ``_norm`` over the observed union,
# ``_norm_notable`` over the fixed GT-notable set (with blind-spot penalty).
NORM_METRICS = (
    "motion_uncertainty_norm",
    "total_uncertainty_norm",
    "motion_uncertainty_norm_notable",
    "total_uncertainty_norm_notable",
)
METRICS = (
    "uncertainty",
    "motion_uncertainty",
    "coverage_uncertainty",
    "total_uncertainty",
    "route_coverage_quality_mean",
    "poor_coverage_risk_mean",
    "ade",
    "fde",
    *NOTABLE_METRICS,
    *NORM_METRICS,
    *PRESENCE_METRICS,
    *COMM_METRICS,
)
SUMMARY_METRICS = (
    "motion_uncertainty",
    "coverage_uncertainty",
    "total_uncertainty",
    "route_coverage_quality_mean",
    "poor_coverage_risk_mean",
    "ade",
    "fde",
    *NOTABLE_METRICS,
    *NORM_METRICS,
    *PRESENCE_METRICS,
    *COMM_METRICS,
)
PLOTLY_INSTALL_HINT = (
    "Plotly is required for interactive HTML output. Install it with "
    "`conda install -n cardreamer_gnn plotly` or update/recreate the environment from environment.yml."
)


def _parse_json_field(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _selected_ids_text(selected: Sequence[object]) -> str:
    ids = [int(v) for v in selected]
    return ",".join(str(v) for v in ids)


def _modality_text(modality_by_vehicle: dict) -> str:
    if not modality_by_vehicle:
        return ""
    return ",".join(f"{int(k)}:{v}" for k, v in sorted(modality_by_vehicle.items(), key=lambda item: int(item[0])))


def policy_label(policy_type: str, selected_vehicle_ids: Sequence[object]) -> str:
    """Readable label for per-member policy comparisons."""
    policy_type = str(policy_type)
    selected = [int(v) for v in selected_vehicle_ids]
    if not policy_type or policy_type.lower() == "nan":
        return "unknown_policy"
    if policy_type == "ego_only" or not selected:
        return policy_type
    return f"{policy_type}[{_selected_ids_text(selected)}]"


def load_policy_uncertainty_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load uncertainty CSV and add parsed JSON + readable policy columns."""
    df = pd.read_csv(path)
    required = {"step", "episode_id", "policy_type", "selected_vehicle_ids", "modality_by_vehicle", "uncertainty", "ade", "fde"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns in {path}: {missing}")

    df = df.copy()
    df["selected_vehicle_ids_parsed"] = df["selected_vehicle_ids"].map(lambda v: _parse_json_field(v, []))
    df["modality_by_vehicle_parsed"] = df["modality_by_vehicle"].map(lambda v: _parse_json_field(v, {}))
    if "notable_object_ids" in df.columns:
        df["notable_object_ids_parsed"] = df["notable_object_ids"].map(lambda v: _parse_json_field(v, []))
    else:
        df["notable_object_ids_parsed"] = [[] for _ in range(len(df))]

    for key in ("step", "episode_id"):
        df[key] = pd.to_numeric(df[key], errors="coerce").fillna(-1).astype(int)
    for key in METRICS:
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce")

    df["policy_label"] = [
        policy_label(pt, selected)
        for pt, selected in zip(df["policy_type"], df["selected_vehicle_ids_parsed"])
    ]
    df["selected_members"] = df["selected_vehicle_ids_parsed"].map(_selected_ids_text)
    df["modality"] = df["modality_by_vehicle_parsed"].map(_modality_text)
    df["notable_objects"] = df["notable_object_ids_parsed"].map(_selected_ids_text)
    return df


def add_metric_delta(df: pd.DataFrame, *, metric: str, baseline: str = "ego_only") -> pd.DataFrame:
    """Add ``{metric}_baseline`` and ``{metric}_delta`` using matching episode/step rows."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} is not present in the CSV; columns={list(df.columns)}")
    out = df.copy()
    base_mask = (out["policy_label"] == baseline) | (out["policy_type"] == baseline)
    baseline_df = (
        out.loc[base_mask, ["episode_id", "step", metric]]
        .groupby(["episode_id", "step"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns={metric: f"{metric}_baseline"})
    )
    out = out.merge(baseline_df, on=["episode_id", "step"], how="left")
    out[f"{metric}_delta"] = out[metric] - out[f"{metric}_baseline"]
    return out


def summarize_policies(df: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Per-policy ranking summary. Lower metric is better."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} is not present in the CSV; columns={list(df.columns)}")
    delta_col = f"{metric}_delta"
    best = df.loc[df.groupby(["episode_id", "step"])[metric].idxmin(), ["policy_label"]]
    best_counts = best["policy_label"].value_counts().rename("best_step_count")

    agg = (
        df.groupby(["policy_label", "policy_type", "selected_members", "modality"], dropna=False)
        .agg(
            mean_metric=(metric, "mean"),
            std_metric=(metric, "std"),
            min_metric=(metric, "min"),
            max_metric=(metric, "max"),
            mean_delta=(delta_col, "mean") if delta_col in df.columns else (metric, lambda _: np.nan),
            rows=(metric, "count"),
        )
        .reset_index()
    )
    agg["std_metric"] = agg["std_metric"].fillna(0.0)
    agg["best_step_count"] = agg["policy_label"].map(best_counts).fillna(0).astype(int)
    return agg.sort_values(["mean_metric", "policy_label"], ascending=[True, True]).reset_index(drop=True)


def summarize_policy_breakdown(
    df: pd.DataFrame,
    *,
    baseline: str = "ego_only",
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> pd.DataFrame:
    """Summarize motion/coverage/total uncertainty for each policy."""
    available = [metric for metric in metrics if metric in df.columns]
    if not available:
        raise ValueError(f"none of the requested summary metrics are present; columns={list(df.columns)}")

    out = df.copy()
    for metric in available:
        out = add_metric_delta(out, metric=metric, baseline=baseline)

    group_cols = ["policy_label", "policy_type", "selected_members", "modality"]
    grouped = out.groupby(group_cols, dropna=False)
    summary = grouped.size().rename("rows").reset_index()

    for metric in available:
        # Keep the table compact: per-metric mean + delta-vs-baseline only (no std/min/max).
        agg = grouped[metric].mean().rename(f"{metric}_mean").reset_index()
        summary = summary.merge(agg, on=group_cols, how="left")
        delta_col = f"{metric}_delta"
        if delta_col in out.columns:
            delta = grouped[delta_col].mean().rename(f"{metric}_delta_vs_{baseline}").reset_index()
            summary = summary.merge(delta, on=group_cols, how="left")

    sort_col = "total_uncertainty_mean" if "total_uncertainty_mean" in summary.columns else f"{available[0]}_mean"
    return summary.sort_values([sort_col, "policy_label"], ascending=[True, True]).reset_index(drop=True)


def write_policy_uncertainty_summary_csv(
    csv_path: Union[str, Path],
    out_csv: Union[str, Path],
    *,
    baseline: str = "ego_only",
) -> Path:
    """Write a policy-level uncertainty breakdown summary CSV."""
    df = load_policy_uncertainty_csv(csv_path)
    summary = summarize_policy_breakdown(df, baseline=baseline)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return out_path


def select_top_policy_labels(summary: pd.DataFrame, *, baseline: str = "ego_only", top_k: int = 0) -> List[str]:
    """Return labels to display. ``top_k <= 0`` means all labels."""
    labels = list(summary["policy_label"])
    if top_k is None or int(top_k) <= 0 or int(top_k) >= len(labels):
        return labels
    keep = labels[: int(top_k)]
    baseline_matches = summary[
        (summary["policy_label"] == baseline) | (summary["policy_type"] == baseline)
    ]["policy_label"].tolist()
    for label in baseline_matches:
        if label not in keep:
            keep.append(label)
    return keep


def build_policy_uncertainty_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric: str,
    title: Optional[str] = None,
):
    """Build the Plotly figure for interactive policy comparison."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:
        raise RuntimeError(PLOTLY_INSTALL_HINT) from exc

    delta_col = f"{metric}_delta"
    title = title or f"Stage-1 Policy {metric.title()} Comparison"
    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.36, 0.24, 0.20, 0.20],
        vertical_spacing=0.075,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=(
            f"{metric} by policy",
            f"{metric} delta vs baseline",
            "Policy summary",
            "Ranking table",
        ),
    )

    hover_cols = [
        "step",
        "episode_id",
        "policy_label",
        "selected_members",
        "modality",
        "notable_objects",
        "uncertainty",
        "motion_uncertainty",
        "coverage_uncertainty",
        "total_uncertainty",
        "comm_window_v2v_slot_rate",
        "object_window_presence_mean",
        "object_window_presence_ego_mean",
        "comm_received_messages_by_prediction_step",
        "ade",
        "fde",
    ]
    hover_cols = [col for col in hover_cols if col in df.columns]
    for label, group in df.sort_values(["episode_id", "step"]).groupby("policy_label", sort=False):
        custom = group[hover_cols].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=group["step"],
                y=group[metric],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                customdata=custom,
                hovertemplate=(
                    "step=%{customdata[0]} episode=%{customdata[1]}<br>"
                    "policy=%{customdata[2]}<br>"
                    "selected=%{customdata[3]} modality=%{customdata[4]}<br>"
                    "notable=%{customdata[5]}<br>"
                    f"{metric}=%{{y:.4f}}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        if delta_col in group.columns:
            fig.add_trace(
                go.Scatter(
                    x=group["step"],
                    y=group[delta_col],
                    mode="lines+markers",
                    name=f"{label} delta",
                    legendgroup=label,
                    showlegend=False,
                    customdata=custom,
                    hovertemplate=(
                        "step=%{customdata[0]} episode=%{customdata[1]}<br>"
                        "policy=%{customdata[2]}<br>"
                        f"{metric} delta=%{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

    fig.add_hline(y=0.0, line_dash="dot", line_color="#666", row=2, col=1)
    fig.add_trace(
        go.Bar(
            x=summary["policy_label"],
            y=summary["mean_metric"],
            name=f"mean {metric}",
            marker_color="#4C78A8",
            hovertemplate="policy=%{x}<br>mean=%{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=summary["policy_label"],
            y=summary["mean_delta"],
            name=f"mean delta",
            marker_color="#F58518",
            hovertemplate="policy=%{x}<br>mean delta=%{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    table_cols = [
        "policy_label",
        "policy_type",
        "selected_members",
        "modality",
        "mean_metric",
        "std_metric",
        "min_metric",
        "max_metric",
        "mean_delta",
        "best_step_count",
    ]
    table_values = []
    for col in table_cols:
        values = summary[col].tolist()
        if col.endswith("metric") or col == "mean_delta":
            values = ["" if pd.isna(v) else f"{float(v):.4f}" for v in values]
        table_values.append(values)
    fig.add_trace(
        go.Table(
            header=dict(values=table_cols, fill_color="#E6EEF8", align="left"),
            cells=dict(values=table_values, align="left"),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        barmode="group",
        height=1100,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=120, b=40),
    )
    fig.update_yaxes(title_text=metric, row=1, col=1)
    fig.update_yaxes(title_text=f"{metric} delta", row=2, col=1)
    fig.update_yaxes(title_text="summary", row=3, col=1)
    fig.update_xaxes(title_text="step", row=2, col=1)
    fig.update_xaxes(tickangle=35, row=3, col=1)
    return fig


def write_policy_uncertainty_html(
    csv_path: Union[str, Path],
    out_html: Union[str, Path],
    *,
    metric: str = "uncertainty",
    baseline: str = "ego_only",
    top_k: int = 0,
    title: Optional[str] = None,
) -> Path:
    """Read CSV, build policy comparison figure, and write interactive HTML."""
    df = load_policy_uncertainty_csv(csv_path)
    df = add_metric_delta(df, metric=metric, baseline=baseline)
    full_summary = summarize_policies(df, metric=metric)
    labels = select_top_policy_labels(full_summary, baseline=baseline, top_k=top_k)
    df = df[df["policy_label"].isin(labels)].copy()
    summary = summarize_policies(df, metric=metric)
    fig = build_policy_uncertainty_figure(df, summary, metric=metric, title=title)

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return out_path
