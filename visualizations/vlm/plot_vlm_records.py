"""Visualize VLM records (vlm_records_*.json / .jsonl): per-sensor scores,
sensor alignment metrics, ego-only vs ego+collaborator confidence breakdowns.

Outputs (under ``--output_dir``):
    confidence_table.csv, sensor_alignment_table.csv
    avg_confidence_comparison_by_question.png   (bar chart per question)
    timeseries_confidence_<qid>.png             (one PNG per question)
    timeseries_confidence_gain_<qid>.png
    timeseries_{facing,region,distance,fov}_alignment_<qid>.png
    timeseries_confidence_contribution_<qid>.png

Usage (from repo root):

    python visualizations/vlm/plot_vlm_records.py \\
        --input data/.../vlm_records_terminated_step_NNN.json \\
        --output_dir logdir/vlm_viz
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Loading
# --------------------------------------------------

def load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
            return data["records"]
        raise ValueError("JSON structure unsupported. Expected list or dict with 'records'.")

    raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl")


# --------------------------------------------------
# Sender / mode helpers
# --------------------------------------------------

def infer_sender_mapping(records: List[Dict[str, Any]]) -> Tuple[Optional[int], List[int]]:
    """
    Return:
        ego_sender_id
        non_ego_sender_ids_sorted
    """
    ego_ids = set()
    non_ego_ids = set()

    for rec in records:
        for s in rec.get("per_sensor_scores", []) or []:
            sid = s.get("sender_id")
            if sid is None:
                continue
            if s.get("is_ego"):
                ego_ids.add(sid)
            else:
                non_ego_ids.add(sid)

    ego_sender_id = sorted(ego_ids)[0] if ego_ids else None
    non_ego_sender_ids_sorted = sorted(non_ego_ids)
    return ego_sender_id, non_ego_sender_ids_sorted


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _find_sensor_by_sender(per_sensor_scores: List[Dict[str, Any]], sender_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if sender_id is None:
        return None
    for s in per_sensor_scores or []:
        if s.get("sender_id") == sender_id:
            return s
    return None


def compute_ego_plus_member_confidence(
    ego_only: Optional[Dict[str, Any]],
    member_sensor: Optional[Dict[str, Any]],
    member_aggregated: Optional[Dict[str, Any]],
) -> Optional[float]:
    """
    User-specified formula:
        | ego_only["weight"] * ego_only["belief"] + member["weight"] * member["belief"] |

    Behaviour:
      - Returns None when the member sensor itself is absent from per_sensor_scores
        (i.e. there is genuinely no observation to score).
      - When ego_only lacks "weight"/"belief" (e.g. ego forward camera was filtered
        out for a rear-region question), ego's contribution is treated as 0 and the
        formula degrades to |w_m * b_m| — i.e. that member's contribution alone.
      - When member is absent from aggregated_details.per_sensor (filtered out at
        aggregation time, e.g. answerability/visibility filter), member's effective
        weight in this aggregation is 0 — the member contributed nothing — so the
        result is just |w_e * b_e| (or 0 if ego is also absent).
    The new JSON stores the effective weight under "weight" in aggregated_details
    and under "importance_weight" in per_sensor_scores; we fall back to the latter
    if the aggregated entry is unavailable.
    """
    if member_sensor is None:
        return None

    ego_dict = ego_only if isinstance(ego_only, dict) else {}
    ego_weight = _to_float(ego_dict.get("weight"), 0.0)
    ego_belief = _to_float(ego_dict.get("belief"), 0.0)

    if isinstance(member_aggregated, dict):
        member_weight = _to_float(member_aggregated.get("weight"), None)
        member_belief = _to_float(member_aggregated.get("belief"), 0.0)
    else:
        member_weight = None
        member_belief = _to_float(member_sensor.get("belief"), 0.0)

    if member_weight is None:
        member_weight = _to_float(member_sensor.get("importance_weight"), 0.0)

    return abs(ego_weight * ego_belief + member_weight * member_belief)


# --------------------------------------------------
# Flatten record-level confidence table
# --------------------------------------------------

def member_column_name(sender_id: int) -> str:
    return f"confidence_ego_plus_{int(sender_id)}"


def build_confidence_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    ego_sender_id, member_sender_ids = infer_sender_mapping(records)

    rows = []

    for rec in records:
        per_sensor_scores = rec.get("per_sensor_scores", []) or []
        per_sensor_aggregated = rec.get("aggregated_details") or {}
        ego_only = rec.get("ego_only") or {}
        ego_plus_shared = rec.get("ego_plus_shared") or {}

        ego_only_conf = _to_float(ego_only.get("confidence"), 0.0)
        ego_plus_shared_conf = _to_float(ego_plus_shared.get("confidence"), 0.0)
        confidence_gain = _to_float(rec.get("confidence_gain"), 0.0)
        agg_per_sensor = per_sensor_aggregated.get("per_sensor") or []

        row: Dict[str, Any] = {
            "step": rec.get("step"),
            "question_id": rec.get("question_id"),
            "question_type": rec.get("question_type"),
            "ego_sender_id": ego_sender_id,
            "confidence_ego_only": ego_only_conf,
            "confidence_ego_plus_shared": ego_plus_shared_conf,
            "confidence_gain": confidence_gain,
        }
        for mid in member_sender_ids:
            ms = _find_sensor_by_sender(per_sensor_scores, mid)
            ma = _find_sensor_by_sender(agg_per_sensor, mid)
            row[member_column_name(mid)] = compute_ego_plus_member_confidence(ego_only, ms, ma)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = ["step", "ego_sender_id", "confidence_ego_only",
                    "confidence_ego_plus_shared", "confidence_gain"]
    numeric_cols += [member_column_name(mid) for mid in member_sender_ids]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def list_member_confidence_columns(conf_df: pd.DataFrame) -> List[str]:
    """Member columns are 'confidence_ego_plus_<sender_id>' (numeric suffix),
    excluding the special 'confidence_ego_plus_shared'."""
    cols = []
    for c in conf_df.columns:
        if not c.startswith("confidence_ego_plus_"):
            continue
        suffix = c[len("confidence_ego_plus_"):]
        if suffix.isdigit():
            cols.append(c)
    return sorted(cols, key=lambda c: int(c[len("confidence_ego_plus_"):]))


# --------------------------------------------------
# Flatten sensor-level alignment table
# --------------------------------------------------

def build_sensor_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for rec in records:
        agg_list = (rec.get("aggregated_details") or {}).get("per_sensor") or []
        agg_by_key = {a.get("sensor_key"): a for a in agg_list if a.get("sensor_key") is not None}

        for s in rec.get("per_sensor_scores", []) or []:
            region_alignment = _to_float(s.get("region_alignment"), 0.0)
            facing_alignment = _to_float(s.get("facing_alignment"), 0.0)
            distance_alignment = _to_float(s.get("distance_alignment"), 0.0)
            fov_alignment = _to_float(s.get("fov_alignment"), 0.0)

            # Effective weight & confidence contribution live in aggregated_details.per_sensor,
            # not per_sensor_scores. Sensors filtered out of aggregation get 0.
            agg = agg_by_key.get(s.get("sensor_key")) or {}
            weight = _to_float(agg.get("weight"), 0.0)
            belief = _to_float(agg.get("belief", s.get("belief")), 0.0)
            contribution = _to_float(agg.get("confidence"), 0.0)
            importance_weight = _to_float(
                agg.get("importance_weight", s.get("importance_weight")), 0.0
            )

            rows.append(
                {
                    "step": rec.get("step"),
                    "question_id": rec.get("question_id"),
                    "question_type": rec.get("question_type"),
                    "sensor_key": s.get("sensor_key"),
                    "sender_id": s.get("sender_id"),
                    "sensor_name": s.get("sensor_name"),
                    "is_ego": s.get("is_ego"),
                    "used_for_aggregation": s.get("used_for_aggregation"),
                    "facing_alignment": facing_alignment,
                    "region_alignment": region_alignment,
                    "distance_alignment": distance_alignment,
                    "fov_alignment": fov_alignment,
                    "importance_weight": importance_weight,
                    "weight": weight,
                    "belief": belief,
                    "confidence_contribution": contribution,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "step",
        "sender_id",
        "facing_alignment",
        "region_alignment",
        "distance_alignment",
        "fov_alignment",
        "importance_weight",
        "weight",
        "belief",
        "confidence_contribution",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------
# Plot 1: combined bar chart
# --------------------------------------------------

SMOOTHING_WINDOW = 10
MAX_STEP = 80


def _smooth(series: pd.Series, window: int = SMOOTHING_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _clip_steps(df: pd.DataFrame, max_step: Optional[int] = MAX_STEP) -> pd.DataFrame:
    if max_step is None or "step" not in df.columns:
        return df
    return df[df["step"] <= max_step]


def _pretty_legend(col: str, member_cols: List[str]) -> str:
    if col == "confidence_ego_only":
        return "Ego MA only"
    if col == "confidence_ego_plus_shared":
        return "Ego MA + All Collaborators"
    if col in member_cols:
        return f"Ego MA + Collaborator {member_cols.index(col) + 1}"
    return col.replace("confidence_", "")


def save_avg_confidence_bar(conf_df: pd.DataFrame, output_dir: Path) -> None:
    if conf_df.empty:
        return

    member_cols = list_member_confidence_columns(conf_df)
    series_cols = ["confidence_ego_only", *member_cols, "confidence_ego_plus_shared"]

    grouped = (
        conf_df.groupby("question_id", dropna=False)[series_cols]
        .mean()
        .reset_index()
        .sort_values("question_id")
    )

    labels = [f"Q{i + 1}" for i in range(len(grouped))]
    x = list(range(len(labels)))
    n = len(series_cols)
    width = 0.8 / max(n, 1)

    plt.figure(figsize=(max(12, len(labels) * (n * 0.35 + 0.5)), 5.5))
    for i, col in enumerate(series_cols):
        offset = (i - (n - 1) / 2.0) * width
        legend_label = _pretty_legend(col, member_cols)
        plt.bar([xi + offset for xi in x], grouped[col], width=width, label=legend_label)

    plt.xticks(x, labels, rotation=0, ha="center", fontsize=14)
    plt.ylabel("Sementic Confidence", fontsize=14, )
    # plt.title("Average confidence by question")
    plt.legend(fontsize=12, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "avg_confidence_comparison_by_question.png", dpi=220)
    plt.close()


# --------------------------------------------------
# Plot 2: confidence time series for each question
# --------------------------------------------------

def save_confidence_timeseries(conf_df: pd.DataFrame, output_dir: Path) -> None:
    if conf_df.empty:
        return

    member_cols = list_member_confidence_columns(conf_df)
    series_cols = ["confidence_ego_only", *member_cols, "confidence_ego_plus_shared"]

    for qid, sub in conf_df.groupby("question_id", dropna=False):
        sub = sub.sort_values("step").copy()
        if sub["step"].isna().all():
            continue
        for col in series_cols:
            sub[col] = _smooth(sub[col])
        sub = _clip_steps(sub)
        if sub.empty:
            continue

        plt.figure(figsize=(9, 5))
        for col in series_cols:
            plt.plot(sub["step"], sub[col], label=_pretty_legend(col, member_cols))

        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Semantic Confidence", fontsize=14)
        # plt.title(f"Confidence over time: {qid}", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()

        safe_name = str(qid).replace("/", "_")
        plt.savefig(output_dir / f"timeseries_confidence_{safe_name}.png", dpi=220)
        plt.close()


# --------------------------------------------------
# Plot 3: confidence gain time series for each question
# --------------------------------------------------

def save_confidence_gain_timeseries(conf_df: pd.DataFrame, output_dir: Path) -> None:
    if conf_df.empty or "confidence_gain" not in conf_df.columns:
        return

    for qid, sub in conf_df.groupby("question_id", dropna=False):
        sub = sub.sort_values("step").copy()
        if sub["step"].isna().all():
            continue
        sub["confidence_gain"] = _smooth(sub["confidence_gain"])
        sub = _clip_steps(sub)
        if sub.empty:
            continue

        plt.figure(figsize=(9, 5))
        plt.plot(sub["step"], sub["confidence_gain"], label="confidence_gain")
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Semantic Confidence Gain", fontsize=14)
        # plt.title(f"Confidence gain over time: {qid}", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()

        safe_name = str(qid).replace("/", "_")
        plt.savefig(output_dir / f"timeseries_confidence_gain_{safe_name}.png", dpi=220)
        plt.close()


# --------------------------------------------------
# Generic per-question sensor line plot
# --------------------------------------------------

def save_sensor_metric_timeseries(
    sensor_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    title_prefix: str,
    filename_prefix: str,
) -> None:
    if sensor_df.empty or metric not in sensor_df.columns:
        return

    for qid, sub in sensor_df.groupby("question_id", dropna=False):
        sub = sub.sort_values(["sensor_key", "step"])
        if sub["step"].isna().all():
            continue

        plt.figure(figsize=(9, 5))

        for sensor_key, sensor_sub in sub.groupby("sensor_key", dropna=False):
            sensor_sub = sensor_sub.sort_values("step").copy()
            sensor_sub[metric] = _smooth(sensor_sub[metric])
            sensor_sub = _clip_steps(sensor_sub)
            if sensor_sub.empty:
                continue
            plt.plot(sensor_sub["step"], sensor_sub[metric], label=str(sensor_key))

        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel(metric, fontsize=14)
        # plt.title(f"{title_prefix}: {qid}", fontsize=14)
        plt.legend(fontsize=12, loc="upper left")
        plt.tight_layout()

        safe_name = str(qid).replace("/", "_")
        plt.savefig(output_dir / f"{filename_prefix}_{safe_name}.png", dpi=220)
        plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VLM records with updated confidence formula.")
    parser.add_argument("--input", type=str, required=True, help="Path to records json/jsonl file")
    parser.add_argument("--output_dir", type=str, default="vlm_viz_selected", help="Directory to save outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)

    conf_df = build_confidence_table(records)
    sensor_df = build_sensor_table(records)

    if not conf_df.empty:
        conf_df.to_csv(output_dir / "confidence_table.csv", index=False)
    if not sensor_df.empty:
        sensor_df.to_csv(output_dir / "sensor_alignment_table.csv", index=False)

    save_avg_confidence_bar(conf_df, output_dir)
    save_confidence_timeseries(conf_df, output_dir)
    save_confidence_gain_timeseries(conf_df, output_dir)

    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="facing_alignment",
        title_prefix="Facing alignment over time",
        filename_prefix="timeseries_facing_alignment",
    )

    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="region_alignment",
        title_prefix="Region alignment over time",
        filename_prefix="timeseries_region_alignment",
    )

    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="distance_alignment",
        title_prefix="Distance alignment over time",
        filename_prefix="timeseries_distance_alignment",
    )
    
    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="fov_alignment",
        title_prefix="FOV alignment over time",
        filename_prefix="timeseries_fov_alignment",
    )

    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="confidence_contribution",
        title_prefix="Confidence contribution over time",
        filename_prefix="timeseries_confidence_contribution",
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
