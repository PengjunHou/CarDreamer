#!/usr/bin/env python3
"""Prop-1 utility-vs-backlog trade-off (V2X paper Sec IV, Prop 1): sweep Λ, plot ``[O(1/Λ), O(Λ)]``.

For each Λ we run the offline scheduler and record the time-average semantic uncertainty (utility, expected
to *decrease* ~``O(1/Λ)``) and the time-average total backlog ``Σ_m Q_m`` (expected to *grow* ~``O(Λ)``).

    python scripts/analyze_wam_lyapunov_tradeoff.py --steps 150 --out-dir outputs/wam_lyapunov_analysis
"""

from __future__ import annotations

import argparse

from _wam_lyapunov_common import contexts_for, driver, load_model, make_axes, save, scheduler_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--stage1-ckpt", default=None)
    ap.add_argument("--out-dir", default="outputs/wam_lyapunov_analysis")
    args = ap.parse_args()

    d = driver()
    model = load_model(args.stage1_ckpt)
    if model is None:
        print("[tradeoff] NOTE: no --stage1-ckpt -> rule fallback (U^mot is collaborator-blind), so the "
              "Λ utility/backlog curve is muted. Pass a Stage-1 checkpoint for the full [O(1/Λ),O(Λ)] curve.")
    contexts = contexts_for(model, steps=args.steps, seed=0)

    rows = []
    for lam in args.lams:
        cfg = scheduler_config(lam=float(lam), budget_bandwidth=args.budget, ts_seconds=0.1)
        _, _, summary = d.run_scheduler(contexts, model=model, cfg=cfg, alpha=0.5)
        rows.append({"lam": float(lam), "utility_uncertainty": summary["time_avg_uncertainty"],
                     "mean_backlog": summary["mean_backlog"], "time_avg_bandwidth": summary["time_avg_bandwidth"],
                     "epochs": summary["epochs"]})
        print(f"  Λ={lam:>6}: U={summary['time_avg_uncertainty']:.4f}  backlog={summary['mean_backlog']:.3g}  B̄={summary['time_avg_bandwidth']:.3f}")

    import pandas as pd

    from pathlib import Path
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tradeoff.csv", index=False)

    plt, fig, (ax1, ax2) = make_axes(1, 2, figsize=(11.0, 4.0))
    ax1.plot(df["lam"], df["utility_uncertainty"], "o-")
    ax1.set_xscale("log"); ax1.set_xlabel("Λ"); ax1.set_ylabel("time-avg uncertainty  U̅")
    ax1.set_title("Utility ~ O(1/Λ)  (lower is better)")
    ax2.plot(df["lam"], df["mean_backlog"], "s-", color="tab:red")
    ax2.set_xscale("log"); ax2.set_xlabel("Λ"); ax2.set_ylabel("mean total backlog  Σ Q_m")
    ax2.set_title("Backlog ~ O(Λ)")
    path = save(plt, fig, out / "tradeoff.png")
    print(f"[tradeoff] wrote {out/'tradeoff.csv'} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
