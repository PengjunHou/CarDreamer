"""Smoke test for the offline Lyapunov driver (scripts/run_wam_lyapunov_offline.py) — CARLA-free, no ckpt."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_driver():
    path = REPO_ROOT / "scripts" / "run_wam_lyapunov_offline.py"
    spec = importlib.util.spec_from_file_location("run_wam_lyapunov_offline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OfflineDriverSmokeTest(unittest.TestCase):
    def test_runs_and_writes_trace(self):
        import pandas as pd

        from car_dreamer.toolkit.wam import SchedulerConfig

        driver = _load_driver()
        contexts = driver.build_synthetic_contexts(steps=18, n_members=1, route_len=50.0, seed=1, bev_size=10)
        cfg = SchedulerConfig(lam=1.0, c0=0.5, budget_bandwidth=0.4, F_max_slots=6, n_min_slots=3,
                              duration_grid=(3, 6), bandwidth_grid=(0.5, 1.0), j_max=1, ts_seconds=0.1)
        cfg._alpha = 0.5
        with tempfile.TemporaryDirectory() as d:
            rows, summary, paths = driver.run_and_write(contexts, model=None, cfg=cfg, out_dir=Path(d))
            self.assertEqual(len(rows), 18)
            self.assertTrue(paths["csv"].exists())
            df = pd.read_csv(paths["csv"])
            for col in ("step", "epoch", "z", "cost_rate_total", "allocated_bandwidth", "realized_u"):
                self.assertIn(col, df.columns)
            # epoch (install) rows have strictly increasing steps
            epoch_steps = df.loc[df["epoch"] > 0.5, "step"].tolist()
            self.assertGreaterEqual(len(epoch_steps), 2)
            self.assertTrue(all(a < b for a, b in zip(epoch_steps, epoch_steps[1:])))
            # summary reports the Prop-1 witnesses
            for key in ("time_avg_bandwidth", "z_over_t", "mean_backlog", "u_c", "epochs"):
                self.assertIn(key, summary)
            self.assertTrue(paths["summary"].exists())


if __name__ == "__main__":
    unittest.main()
