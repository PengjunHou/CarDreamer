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


def compute_subset_confidence(per_sensor_scores, include_sender_ids):
    pos = 0.0
    neg = 0.0
    conf = 0.0

    for s in per_sensor_scores or []:
        sid = s.get("sender_id")
        if sid not in include_sender_ids:
            continue

        w = float(s.get("importance_positive") or 0.0)
        pos_s = float(s.get("positive_score") or 0.0)
        neg_s = float(s.get("negative_score") or 0.0)

        pos += w * pos_s
        neg += w * neg_s
        conf += abs(pos_s - neg_s) * w

    return pos, neg, conf


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

        ego_only_conf = float((rec.get("ego_only") or {}).get("confidence") or 0.0)
        ego_plus_shared_conf = float((rec.get("ego_plus_shared") or {}).get("confidence") or 0.0)

        ego_plus_member1_conf = None
        ego_plus_member2_conf = None

        if ego_sender_id is not None and member1_id is not None:
            _, _, ego_plus_member1_conf = compute_subset_confidence(
                per_sensor_scores,
                {ego_sender_id, member1_id},
            )

        if ego_sender_id is not None and member2_id is not None:
            _, _, ego_plus_member2_conf = compute_subset_confidence(
                per_sensor_scores,
                {ego_sender_id, member2_id},
            )

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
            region_alignment = float(s.get("region_alignment") or 0.0)
            facing_alignment = float(s.get("facing_alignment") or 0.0)
            distance_alignment = float(s.get("distance_alignment") or 0.0)

            contribution = (region_alignment + facing_alignment) * distance_alignment

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
    parser = argparse.ArgumentParser(description="Visualize VLM records with only requested plots.")
    parser.add_argument("--input", type=str, required=True, help="Path to records json/jsonl file")
    parser.add_argument("--output_dir", type=str, default="vlm_viz_selected", help="Directory to save outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)

    conf_df = build_confidence_table(records)
    sensor_df = build_sensor_table(records)

    # save tables for debugging / checking
    if not conf_df.empty:
        conf_df.to_csv(output_dir / "confidence_table.csv", index=False)
    if not sensor_df.empty:
        sensor_df.to_csv(output_dir / "sensor_alignment_table.csv", index=False)

    # (1)
    save_avg_confidence_bar(conf_df, output_dir)

    # (2)
    save_confidence_timeseries(conf_df, output_dir)

    # (3) facing_alignment
    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="facing_alignment",
        title_prefix="Facing alignment over time",
        filename_prefix="timeseries_facing_alignment",
    )

    # (4) region_alignment
    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="region_alignment",
        title_prefix="Region alignment over time",
        filename_prefix="timeseries_region_alignment",
    )

    # (5) distance_alignment
    save_sensor_metric_timeseries(
        sensor_df,
        output_dir,
        metric="distance_alignment",
        title_prefix="Distance alignment over time",
        filename_prefix="timeseries_distance_alignment",
    )

    # (6) contribution = (region_alignment + facing_alignment) * distance_alignment
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