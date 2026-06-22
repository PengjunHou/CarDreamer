import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from car_dreamer.toolkit.wam import (
    add_metric_delta,
    load_policy_uncertainty_csv,
    policy_label,
    summarize_policy_breakdown,
    summarize_policies,
    write_policy_uncertainty_html,
    write_policy_uncertainty_summary_csv,
)


FIELDS = [
    "step",
    "episode_id",
    "policy_type",
    "selected_vehicle_ids",
    "modality_by_vehicle",
    "notable_object_ids",
    "uncertainty",
    "motion_uncertainty",
    "coverage_uncertainty",
    "total_uncertainty",
    "route_coverage_quality_mean",
    "poor_coverage_risk_mean",
    "ade",
    "fde",
]


def _write_synthetic_csv(path: Path) -> None:
    rows = [
        (0, 0, "ego_only", [], {}, [100], 1.0, 0.8, 0.2, 1.0, 0.8, 0.2, 5.0, 6.0),
        (0, 0, "single_candidate_objlist", [444], {444: "objlist"}, [100], 0.8, 0.7, 0.1, 0.8, 0.9, 0.1, 4.0, 5.0),
        (0, 0, "single_candidate_bev", [447], {447: "bev"}, [100], 1.2, 0.7, 0.5, 1.2, 0.5, 0.5, 6.0, 7.0),
        (0, 0, "all_candidates_bev", [444, 447], {444: "bev", 447: "bev"}, [100], 0.7, 0.6, 0.1, 0.7, 0.9, 0.1, 3.0, 4.0),
        (1, 0, "ego_only", [], {}, [100], 1.1, 0.8, 0.3, 1.1, 0.7, 0.3, 5.5, 6.5),
        (1, 0, "single_candidate_objlist", [444], {444: "objlist"}, [100], 0.9, 0.7, 0.2, 0.9, 0.8, 0.2, 4.5, 5.5),
        (1, 0, "single_candidate_bev", [447], {447: "bev"}, [100], 1.3, 0.7, 0.6, 1.3, 0.4, 0.6, 6.5, 7.5),
        (1, 0, "all_candidates_bev", [444, 447], {444: "bev", 447: "bev"}, [100], 0.6, 0.5, 0.1, 0.6, 0.9, 0.1, 3.5, 4.5),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for (
            step,
            episode,
            policy_type,
            selected,
            modality,
            notable,
            unc,
            motion_unc,
            coverage_unc,
            total_unc,
            coverage_quality,
            poor_coverage,
            ade,
            fde,
        ) in rows:
            writer.writerow(
                {
                    "step": step,
                    "episode_id": episode,
                    "policy_type": policy_type,
                    "selected_vehicle_ids": json.dumps(selected),
                    "modality_by_vehicle": json.dumps({str(k): v for k, v in modality.items()}),
                    "notable_object_ids": json.dumps(notable),
                    "uncertainty": unc,
                    "motion_uncertainty": motion_unc,
                    "coverage_uncertainty": coverage_unc,
                    "total_uncertainty": total_unc,
                    "route_coverage_quality_mean": coverage_quality,
                    "poor_coverage_risk_mean": poor_coverage,
                    "ade": ade,
                    "fde": fde,
                }
            )


class Stage1PolicyVizTest(unittest.TestCase):
    def test_policy_label_includes_selected_members(self):
        self.assertEqual(policy_label("ego_only", []), "ego_only")
        self.assertEqual(policy_label("single_candidate_bev", [447]), "single_candidate_bev[447]")
        self.assertEqual(policy_label("all_candidates_objlist", [444, 447]), "all_candidates_objlist[444,447]")

    def test_load_delta_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uncertainty_by_policy.csv"
            _write_synthetic_csv(path)
            df = load_policy_uncertainty_csv(path)

        self.assertIn("single_candidate_objlist[444]", set(df["policy_label"]))
        self.assertIn("all_candidates_bev[444,447]", set(df["policy_label"]))
        obj_row = df[df["policy_label"] == "single_candidate_objlist[444]"].iloc[0]
        self.assertEqual(obj_row["selected_members"], "444")
        self.assertEqual(obj_row["modality"], "444:objlist")

        df = add_metric_delta(df, metric="uncertainty", baseline="ego_only")
        delta = df[(df["step"] == 0) & (df["policy_label"] == "single_candidate_objlist[444]")][
            "uncertainty_delta"
        ].iloc[0]
        self.assertAlmostEqual(float(delta), -0.2, places=5)

        summary = summarize_policies(df, metric="uncertainty")
        required = {
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
        }
        self.assertTrue(required.issubset(set(summary.columns)))
        best = summary.iloc[0]
        self.assertEqual(best["policy_label"], "all_candidates_bev[444,447]")
        self.assertTrue(np.isfinite(float(best["mean_metric"])))
        self.assertEqual(int(best["best_step_count"]), 2)

    def test_writes_policy_breakdown_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "uncertainty_by_policy.csv"
            out_csv = Path(tmp) / "policy_summary.csv"
            _write_synthetic_csv(csv_path)
            df = load_policy_uncertainty_csv(csv_path)
            summary = summarize_policy_breakdown(df, baseline="ego_only")
            write_policy_uncertainty_summary_csv(csv_path, out_csv, baseline="ego_only")

            self.assertTrue(out_csv.exists())
            required = {
                "policy_label",
                "motion_uncertainty_mean",
                "coverage_uncertainty_mean",
                "total_uncertainty_mean",
                "total_uncertainty_delta_vs_ego_only",
                "rows",
            }
            self.assertTrue(required.issubset(set(summary.columns)))
            best = summary.iloc[0]
            self.assertEqual(best["policy_label"], "all_candidates_bev[444,447]")
            self.assertAlmostEqual(float(best["total_uncertainty_mean"]), 0.65, places=5)

    def test_writes_interactive_html(self):
        try:
            import plotly  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("plotly is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "uncertainty_by_policy.csv"
            out_html = Path(tmp) / "policy_uncertainty.html"
            _write_synthetic_csv(csv_path)
            write_policy_uncertainty_html(csv_path, out_html, metric="uncertainty", baseline="ego_only")
            text = out_html.read_text(encoding="utf-8")

        self.assertGreater(len(text), 1000)
        self.assertIn("Plotly", text)
        self.assertIn("ego_only", text)
        self.assertIn("single_candidate_bev[447]", text)
        self.assertIn("all_candidates_bev[444,447]", text)


if __name__ == "__main__":
    unittest.main()
