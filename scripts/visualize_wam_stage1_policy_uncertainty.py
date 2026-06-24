#!/usr/bin/env python3
"""Interactive visualization for Stage-1 policy uncertainty CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage-1 policy uncertainty as interactive HTML.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=Path("outputs/wam_stage1_uncertainty/uncertainty_by_policy.csv"),
        help="input uncertainty_by_policy.csv from evaluate_wam_stage1_uncertainty.py",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("outputs/wam_stage1_policy_viz/policy_uncertainty.html"),
        help="output interactive HTML path",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="optional output policy-level summary CSV path",
    )
    parser.add_argument("--html", action="store_true", default=True, help="write interactive HTML")
    parser.add_argument("--no-html", dest="html", action="store_false", help="skip interactive HTML")
    parser.add_argument(
        "--metric",
        choices=("uncertainty", "motion_uncertainty", "coverage_uncertainty", "total_uncertainty",
                 "mean_uncertainty", "ade", "fde",
                 "motion_uncertainty_notable", "total_uncertainty_notable", "ade_notable", "fde_notable",
                 "motion_uncertainty_norm", "total_uncertainty_norm"),
        default="total_uncertainty",
    )
    parser.add_argument("--baseline", default="ego_only", help="baseline policy label or policy_type")
    parser.add_argument("--top-k", type=int, default=0, help="show only top-k policies by mean metric; 0 shows all")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metric = "uncertainty" if str(args.metric) == "mean_uncertainty" else str(args.metric)

    from car_dreamer.toolkit.wam.stage1_policy_viz import (
        write_policy_uncertainty_html,
        write_policy_uncertainty_summary_csv,
    )

    if args.html:
        out_path = write_policy_uncertainty_html(
            args.csv_path,
            args.out_html,
            metric=metric,
            baseline=args.baseline,
            top_k=args.top_k,
            title=args.title,
        )
        print(f"Wrote interactive policy uncertainty visualization to {out_path}", flush=True)
    if args.out_summary is not None:
        summary_path = write_policy_uncertainty_summary_csv(
            args.csv_path,
            args.out_summary,
            baseline=args.baseline,
        )
        print(f"Wrote policy uncertainty summary to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
