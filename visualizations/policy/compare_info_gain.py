"""Compare ego information gain (confidence_gain) across fixed policies P1-P8.

For each policy and each question_id, we average confidence_gain across all
records (ignoring the continuously-updated vlm_records_live.json file). The
script saves:

  - <out_dir>/policy_question_mean_gain.csv
        Long-form table: policy_id, question_id, mean_gain, std_gain, n_records
  - <out_dir>/policy_question_mean_gain_pivot.csv
        Wide table indexed by question_id, one column per policy.
  - <out_dir>/policy_parameters.csv
        Parameters of each policy (selection / frequency / bandwidth rules).
  - <out_dir>/policy_information_gain_by_question.png
        Grouped bar chart: x = question_id, bar groups = P1..P8.
  - <out_dir>/policy_information_gain_overall.png
        Mean confidence_gain per policy (averaged across all questions).

Usage:
    python compare_policy_information_gain.py \
        --root data/emulation_fixed_20260430 \
        --out-dir logdir/policy_info_gain
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_PARAMETERS: List[Dict[str, str]] = [
    {
        "policy_id": "P1",
        "name": "Ego-Only",
        "selection": "no collaborator (alpha=0 for every vehicle)",
        "num_selected": "0",
        "sharing_frequency_nu": "0",
        "bandwidth_allocation": "0 (no shared bandwidth)",
        "notes": "Baseline; ego relies solely on its own sensors",
    },
    {
        "policy_id": "P2",
        "name": "Full-Low-Equal",
        "selection": "all candidate vehicles",
        "num_selected": "all",
        "sharing_frequency_nu": "low (0.2)",
        "bandwidth_allocation": "equal split (B / N)",
        "notes": "Maximum coverage but each link is low-rate",
    },
    {
        "policy_id": "P3",
        "name": "Full-High-Equal",
        "selection": "all candidate vehicles",
        "num_selected": "all",
        "sharing_frequency_nu": "high (1.0)",
        "bandwidth_allocation": "equal split (B / N)",
        "notes": "Maximum coverage at high frequency",
    },
    {
        "policy_id": "P4",
        "name": "Top2-Mid-Equal",
        "selection": "top 2 by sender_collab score",
        "num_selected": "2",
        "sharing_frequency_nu": "mid (0.5) for both",
        "bandwidth_allocation": "equal split (B / 2)",
        "notes": "Pick the two highest-value senders",
    },
    {
        "policy_id": "P5",
        "name": "Top2-Adaptive-Value",
        "selection": "top 2 by sender_collab score",
        "num_selected": "2",
        "sharing_frequency_nu": "rank-1 high (1.0), rank-2 mid (0.5)",
        "bandwidth_allocation": "value-weighted (proportional to s_collab)",
        "notes": "Adaptive frequency + value-aware bandwidth",
    },
    {
        "policy_id": "P6",
        "name": "Nearest2-Mid-Dist",
        "selection": "2 nearest vehicles (smallest distance_m)",
        "num_selected": "2",
        "sharing_frequency_nu": "mid (0.5)",
        "bandwidth_allocation": "distance-weighted (proportional to 1/distance)",
        "notes": "Geometry-driven baseline",
    },
    {
        "policy_id": "P7",
        "name": "Random2-Mid-Equal",
        "selection": "random 2 vehicles",
        "num_selected": "2",
        "sharing_frequency_nu": "mid (0.5)",
        "bandwidth_allocation": "equal split (B / 2)",
        "notes": "Random-selection baseline",
    },
    {
        "policy_id": "P8",
        "name": "Random3-Adaptive-Value",
        "selection": "random 3 vehicles",
        "num_selected": "3",
        "sharing_frequency_nu": "rank-1 high (1.0), others mid (0.5)",
        "bandwidth_allocation": "value-weighted (proportional to s_collab)",
        "notes": "Stochastic selection with adaptive freq+bandwidth",
    },
]


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    raise ValueError(f"Unexpected JSON layout in {path}")


def collect_records(root: Path, policy_ids: List[str]) -> pd.DataFrame:
    """Walk root/<policy_id>/vlm_records_*.json and collect per-record gain rows."""
    rows: List[Dict[str, Any]] = []
    for pid in policy_ids:
        pdir = root / pid
        if not pdir.exists():
            print(f"[warn] missing policy directory: {pdir}")
            continue
        files = sorted(pdir.glob("vlm_records_*.json"))
        # Skip the continuously-updated live log because it overlaps with the
        # terminated episode dumps.
        files = [f for f in files if f.name != "vlm_records_live.json"]
        if not files:
            print(f"[warn] no vlm_records_*.json in {pdir}")
            continue
        for fpath in files:
            records = load_records(fpath)
            for rec in records:
                gain = rec.get("confidence_gain")
                if gain is None:
                    continue
                rows.append(
                    {
                        "policy_id": pid,
                        "episode_file": fpath.name,
                        "step": rec.get("step"),
                        "question_id": rec.get("question_id"),
                        "question_type": rec.get("question_type"),
                        "confidence_gain": float(gain),
                    }
                )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["policy_id", "question_id"], dropna=False)["confidence_gain"]
        .agg(mean_gain="mean", std_gain="std", n_records="size")
        .reset_index()
    )
    grouped["std_gain"] = grouped["std_gain"].fillna(0.0)
    return grouped


def plot_grouped_bars(agg: pd.DataFrame, policy_ids: List[str], out_path: Path) -> None:
    if agg.empty:
        return
    pivot = agg.pivot(index="question_id", columns="policy_id", values="mean_gain")
    # Order columns and fill missing
    pivot = pivot.reindex(columns=policy_ids)
    questions = list(pivot.index)
    n_policies = len(policy_ids)
    width = 0.8 / max(n_policies, 1)
    x = np.arange(len(questions))

    fig_w = max(12.0, len(questions) * (n_policies * 0.35 + 0.6))
    plt.figure(figsize=(fig_w, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, pid in enumerate(policy_ids):
        vals = pivot[pid].fillna(0.0).to_numpy() if pid in pivot.columns else np.zeros(len(questions))
        offset = (i - (n_policies - 1) / 2.0) * width
        plt.bar(x + offset, vals, width=width, label=pid, color=cmap(i % 10))

    plt.xticks(x, questions, rotation=35, ha="right")
    plt.ylabel("Mean confidence_gain (information gain)")
    plt.title("Ego information gain by question across policies P1-P8")
    plt.legend(ncol=min(n_policies, 4), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_overall(agg: pd.DataFrame, policy_ids: List[str], out_path: Path) -> None:
    if agg.empty:
        return
    overall = (
        agg.groupby("policy_id", dropna=False)["mean_gain"]
        .mean()
        .reindex(policy_ids)
        .fillna(0.0)
    )
    plt.figure(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    bars = plt.bar(
        overall.index,
        overall.values,
        color=[cmap(i % 10) for i in range(len(overall))],
    )
    for bar, v in zip(bars, overall.values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
    plt.ylabel("Mean confidence_gain (averaged across questions)")
    plt.title("Average ego information gain per policy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ego information gain across P1-P8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", default="data/emulation_fixed_20260430",
        help="Directory containing one subdirectory per policy (P1..P8).",
    )
    parser.add_argument(
        "--out-dir", default="logdir/policy_info_gain",
        help="Where to write the resulting tables and figures.",
    )
    parser.add_argument(
        "--policy-ids", nargs="+",
        default=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_df = collect_records(root, args.policy_ids)
    if records_df.empty:
        raise SystemExit(f"No records found under {root}.")

    print(f"Loaded {len(records_df)} records across {records_df['policy_id'].nunique()} policies "
          f"and {records_df['question_id'].nunique()} questions.")

    agg = aggregate(records_df)
    agg.to_csv(out_dir / "policy_question_mean_gain.csv", index=False)

    pivot = agg.pivot(index="question_id", columns="policy_id", values="mean_gain")
    pivot = pivot.reindex(columns=args.policy_ids).sort_index()
    pivot.to_csv(out_dir / "policy_question_mean_gain_pivot.csv")

    params_df = pd.DataFrame(POLICY_PARAMETERS)
    params_df = params_df[params_df["policy_id"].isin(args.policy_ids)].reset_index(drop=True)
    params_df.to_csv(out_dir / "policy_parameters.csv", index=False)

    plot_grouped_bars(agg, args.policy_ids, out_dir / "policy_information_gain_by_question.png")
    plot_overall(agg, args.policy_ids, out_dir / "policy_information_gain_overall.png")

    print(f"\nSaved outputs to: {out_dir}")
    print("\n--- Mean confidence_gain by (policy, question) ---")
    print(pivot.round(4).to_string())
    print("\n--- Policy parameters ---")
    print(params_df.to_string(index=False))


if __name__ == "__main__":
    main()
