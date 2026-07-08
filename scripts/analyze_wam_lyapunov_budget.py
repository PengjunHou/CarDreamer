#!/usr/bin/env python3
"""Budget compliance + queue stability (V2X paper Sec IV, Prop 1): ``Z(T)/T → 0`` and ``B̄ ≤ B̄_bgt``.

A single episode's trace shows (a) the cumulative time-average allocated bandwidth converging at/below the
budget line, (b) the virtual-queue witness ``Z(t)/t`` decaying toward 0, and (c) the total backlog
``Σ_m Q_m`` staying bounded (mean-rate stable).

    python scripts/analyze_wam_lyapunov_budget.py --steps 300 --budget 0.4 --out-dir outputs/wam_lyapunov_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _wam_lyapunov_common import driver, load_model, make_axes, save, scheduler_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--stage1-ckpt", default=None)
    ap.add_argument("--out-dir", default="outputs/wam_lyapunov_analysis")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    d = driver()
    model = load_model(args.stage1_ckpt)
    contexts = d.build_synthetic_contexts(steps=args.steps, seed=0)
    cfg = scheduler_config(lam=args.lam, budget_bandwidth=args.budget, ts_seconds=0.1)
    _, rows, summary = d.run_scheduler(contexts, model=model, cfg=cfg, alpha=0.5)
    df = pd.DataFrame(rows)

    t = df["step"].to_numpy() + 1.0
    cum_bandwidth = np.cumsum(df["allocated_bandwidth"].to_numpy()) / t
    z_over_t = df["z"].to_numpy() / t

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.assign(cum_mean_bandwidth=cum_bandwidth, z_over_t=z_over_t).to_csv(out / "budget.csv", index=False)

    plt, fig, (a1, a2, a3) = make_axes(1, 3, figsize=(13.0, 3.8))
    a1.plot(df["step"], cum_bandwidth, label="cumulative B̄(t)")
    a1.axhline(args.budget, color="k", ls="--", label="B̄_bgt")
    a1.set_xlabel("slot t"); a1.set_ylabel("bandwidth ratio"); a1.set_title("B̄ ≤ B̄_bgt"); a1.legend()
    a2.plot(df["step"], z_over_t, color="tab:green")
    a2.set_xlabel("slot t"); a2.set_ylabel("Z(t)/t"); a2.set_title("budget witness Z(T)/T → 0")
    a3.plot(df["step"], df["total_backlog"], color="tab:red")
    a3.set_xlabel("slot t"); a3.set_ylabel("Σ Q_m (payload units)"); a3.set_title("backlog bounded")
    path = save(plt, fig, out / "budget.png")
    print(f"[budget] B̄={summary['time_avg_bandwidth']:.3f} (budget {args.budget})  Z(T)/T={summary['z_over_t']:.4g}  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
