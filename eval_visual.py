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
    ego_only: Dict[str, Any],
    member_sensor: Optional[Dict[str, Any]],
    member_aggregated:Dict[str, Any],
) -> Optional[float]:
    """
    New user-specified formula:
        | ego_only["weight"] * ego_only["belief"] + member["weight"] * member["belief"] |

    For the member sensor, if "weight" is missing, fall back to "importance_weight"
    because the new JSON stores the effective weight there.
    """
    if member_sensor is None:
        return None

    ego_weight = _to_float(ego_only.get("weight"), 0.0)
    ego_belief = _to_float(ego_only.get("belief"), 0.0)

    member_weight = _to_float(member_aggregated.get("weight"), None)
    if member_weight is None:
        print("member weight is None")
        member_weight = _to_float(member_sensor.get("importance_weight"), 0.0)
    member_belief = _to_float(member_aggregated.get("belief"), 0.0)

    return abs(ego_weight * ego_belief + member_weight * member_belief)


# --------------------------------------------------
# Flatten record-level confidence table
# --------------------------------------------------

def build_confidence_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    ego_sender_id, member_sender_ids = infer_sender_mapping(records)

    member1_id = member_sender_ids[0] if len(member_sender_ids) >= 1 else None
    member2_id = member_sender_ids[1] if len(member_sender_ids) >= 2 else None

    rows = []

    for rec in records:
        per_sensor_scores = rec.get("per_sensor_scores", []) or []
        per_sensor_aggregated = rec.get("aggregated_details") or {}
        ego_only = rec.get("ego_only") or {}
        ego_plus_shared = rec.get("ego_plus_shared") or {}

        ego_only_conf = _to_float(ego_only.get("confidence"), 0.0)
        ego_plus_shared_conf = _to_float(ego_plus_shared.get("confidence"), 0.0)
        confidence_gain = _to_float(rec.get("confidence_gain"), 0.0)

        member1_sensor = _find_sensor_by_sender(per_sensor_scores, member1_id)
        member2_sensor = _find_sensor_by_sender(per_sensor_scores, member2_id)
        member1_aggregated = _find_sensor_by_sender(per_sensor_aggregated.get("per_sensor"), member1_id)
        member2_aggregated = _find_sensor_by_sender(per_sensor_aggregated.get("per_sensor"), member2_id)

        ego_plus_member1_conf = compute_ego_plus_member_confidence(ego_only, member1_sensor, member1_aggregated)
        ego_plus_member2_conf = compute_ego_plus_member_confidence(ego_only, member2_sensor, member2_aggregated)

        rows.append(
            {
                "step": rec.get("step"),
                "question_id": rec.get("question_id"),
                "question_type": rec.get("question_type"),
                "ego_sender_id": ego_sender_id,
                "member1_sender_id": member1_id,
                "member2_sender_id": member2_id,
                "confidence_ego_only": ego_only_conf,
                "confidence_ego_plus_member1": ego_plus_member1_conf,
                "confidence_ego_plus_member2": ego_plus_member2_conf,
                "confidence_ego_plus_shared": ego_plus_shared_conf,
                "confidence_gain": confidence_gain,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "step",
        "ego_sender_id",
        "member1_sender_id",
        "member2_sender_id",
        "confidence_ego_only",
        "confidence_ego_plus_member1",
        "confidence_ego_plus_member2",
        "confidence_ego_plus_shared",
        "confidence_gain",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------
# Flatten sensor-level alignment table
# --------------------------------------------------

def build_sensor_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for rec in records:
        for s in rec.get("per_sensor_scores", []) or []:
            region_alignment = _to_float(s.get("region_alignment"), 0.0)
            facing_alignment = _to_float(s.get("facing_alignment"), 0.0)
            distance_alignment = _to_float(s.get("distance_alignment"), 0.0)
            fov_alignment = _to_float(s.get("fov_alignment"), 0.0)
            
            contribution = _to_float(s.get("weight"), 0.0)

            rows.append(
                {
                    "step": rec.get("step"),
                    "question_id": rec.get("question_id"),
                    "question_type": rec.get("question_type"),
                    "sensor_key": s.get("sensor_key"),
                    "sender_id": s.get("sender_id"),
                    "sensor_name": s.get("sensor_name"),
                    "is_ego": s.get("is_ego"),
                    "facing_alignment": facing_alignment,
                    "region_alignment": region_alignment,
                    "distance_alignment": distance_alignment,
                    "fov_alignment": fov_alignment,
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
        "confidence_contribution",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------
# Plot 1: combined bar chart
# --------------------------------------------------

def save_avg_confidence_bar(conf_df: pd.DataFrame, output_dir: Path) -> None:
    if conf_df.empty:
        return

    grouped = (
        conf_df.groupby("question_id", dropna=False)
        .agg(
            ego_only=("confidence_ego_only", "mean"),
            ego_plus_member1=("confidence_ego_plus_member1", "mean"),
            ego_plus_member2=("confidence_ego_plus_member2", "mean"),
            ego_plus_shared=("confidence_ego_plus_shared", "mean"),
        )
        .reset_index()
        .sort_values("question_id")
    )

    labels = grouped["question_id"].tolist()
    x = list(range(len(labels)))
    width = 0.2

    plt.figure(figsize=(max(10, len(labels) * 1.0), 5.5))
    plt.bar([i - 1.5 * width for i in x], grouped["ego_only"], width=width, label="ego_only")
    plt.bar([i - 0.5 * width for i in x], grouped["ego_plus_member1"], width=width, label="ego_plus_member1")
    plt.bar([i + 0.5 * width for i in x], grouped["ego_plus_member2"], width=width, label="ego_plus_member2")
    plt.bar([i + 1.5 * width for i in x], grouped["ego_plus_shared"], width=width, label="ego_plus_shared")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Average confidence")
    plt.title("Average confidence by question")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "avg_confidence_comparison_by_question.png", dpi=220)
    plt.close()


# --------------------------------------------------
# Plot 2: confidence time series for each question
# --------------------------------------------------

def save_confidence_timeseries(conf_df: pd.DataFrame, output_dir: Path) -> None:
    if conf_df.empty:
        return

    for qid, sub in conf_df.groupby("question_id", dropna=False):
        sub = sub.sort_values("step")
        if sub["step"].isna().all():
            continue

        plt.figure(figsize=(9, 5))
        plt.plot(sub["step"], sub["confidence_ego_only"], label="ego_only")
        plt.plot(sub["step"], sub["confidence_ego_plus_member1"], label="ego_plus_member1")
        plt.plot(sub["step"], sub["confidence_ego_plus_member2"], label="ego_plus_member2")
        plt.plot(sub["step"], sub["confidence_ego_plus_shared"], label="ego_plus_shared")

        plt.xlabel("Step")
        plt.ylabel("Confidence")
        plt.title(f"Confidence over time: {qid}")
        plt.legend()
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
        sub = sub.sort_values("step")
        if sub["step"].isna().all():
            continue

        plt.figure(figsize=(9, 5))
        plt.plot(sub["step"], sub["confidence_gain"], label="confidence_gain")
        plt.xlabel("Step")
        plt.ylabel("Confidence gain")
        plt.title(f"Confidence gain over time: {qid}")
        plt.legend()
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
            sensor_sub = sensor_sub.sort_values("step")
            plt.plot(sensor_sub["step"], sensor_sub[metric], label=str(sensor_key))

        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.title(f"{title_prefix}: {qid}")
        plt.legend()
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
        filename_prefix="timeseries_distance_alignment",
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
