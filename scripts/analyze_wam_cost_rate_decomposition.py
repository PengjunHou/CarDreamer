#!/usr/bin/env python3
"""Cost-rate decomposition (V2X paper Sec IV.C, eq 42): the (P2) objective as interpretable per-slot rates.

Plots the four adaptive-price cost rates of the installed chunk over the episode — amortized planning
(``Λc0/F``), uncertainty (``Λ/F·ΣŨ``, price 1), bandwidth (``Z/F·ΣB``, price ``Z``), and net-load
(``Σ_m Q_m/F·Σ(L̂−R̂Ts)``, price ``Q_m``) — plus their total. Shows how the trade-offs are governed by the
learned prices rather than hand-tuned weights.

    python scripts/analyze_wam_cost_rate_decomposition.py --steps 200 --out-dir outputs/wam_lyapunov_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _wam_lyapunov_common import driver, load_model, make_axes, save, scheduler_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--stage1-ckpt", default=None)
    ap.add_argument("--out-dir", default="outputs/wam_lyapunov_analysis")
    args = ap.parse_args()

    import pandas as pd

    d = driver()
    model = load_model(args.stage1_ckpt)
    contexts = d.build_synthetic_contexts(steps=args.steps, seed=0)
    cfg = scheduler_config(lam=args.lam, budget_bandwidth=args.budget, ts_seconds=0.1)
    _, rows, _ = d.run_scheduler(contexts, model=model, cfg=cfg, alpha=0.5)
    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["step", "planning_rate", "uncertainty_rate", "bandwidth_rate", "net_load_rate", "cost_rate_total"]].to_csv(
        out / "cost_rate.csv", index=False
    )

    # net_load_rate can be negative (a backlogged link that drains) -> plot as lines, not a stack.
    plt, fig, ax = make_axes(1, 1, figsize=(9.0, 4.5))
    for col, color in [("planning_rate", "tab:gray"), ("uncertainty_rate", "tab:blue"),
                       ("bandwidth_rate", "tab:green"), ("net_load_rate", "tab:red")]:
        ax.plot(df["step"], df[col], label=col, color=color)
    ax.plot(df["step"], df["cost_rate_total"], label="total", color="black", lw=2.0)
    ax.set_xlabel("slot t"); ax.set_ylabel("cost rate  g(a)")
    ax.set_title("(P2) cost-rate decomposition (eq 42)")
    ax.legend(loc="best", fontsize=8)
    path = save(plt, fig, out / "cost_rate.png")
    print(f"[cost-rate] wrote {out/'cost_rate.csv'} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
