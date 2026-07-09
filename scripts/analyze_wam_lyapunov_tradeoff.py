#!/usr/bin/env python3
"""Perception–communication trade-off (V2X paper Sec IV.C): sweep the budget ``B̄_bgt``.

Per the paper, the perception↔communication trade-off is set by the physically-meaningful bandwidth budget
``B̄_bgt`` (priced online by the virtual queue ``Z``), NOT by the Lyapunov weight ``Λ`` (which controls only
the convergence/backlog gap, ``U_c ≤ U_c^opt(B̄_bgt) + O(1/Λ)`` — nearly flat in ``U``). So the visible
trade-off is ``U`` vs ``B̄_bgt``: a larger budget admits more cooperation and lowers the task-oriented
uncertainty. Use ``--sweep lam`` for the (subtle) convergence/backlog view instead.

    python scripts/analyze_wam_lyapunov_tradeoff.py --stage1-ckpt outputs/wam_stage1/stage1_step1000.pt --steps 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _wam_lyapunov_common import contexts_for, driver, load_model, make_axes, save, scheduler_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", choices=("budget", "lam"), default="budget")
    ap.add_argument("--values", type=float, nargs="+", default=None)
    ap.add_argument("--lam", type=float, default=1.0, help="fixed Λ when --sweep budget")
    ap.add_argument("--budget", type=float, default=0.3, help="fixed B̄_bgt when --sweep lam")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--stage1-ckpt", default=None)
    ap.add_argument("--out-dir", default="outputs/wam_lyapunov_analysis")
    args = ap.parse_args()

    import pandas as pd

    d = driver()
    model = load_model(args.stage1_ckpt)
    if model is None:
        print("[tradeoff] NOTE: no --stage1-ckpt -> rule fallback; the trade-off is still coverage-driven "
              "(U^cov), but a Stage-1 checkpoint gives the model's cooperative motion estimate too.")
    contexts = contexts_for(model, steps=args.steps, seed=0, sensor_period=5)

    def run(lam, budget):
        cfg = scheduler_config(lam=lam, budget_bandwidth=budget, F_max_slots=20, n_min_slots=10,
                               bandwidth_grid=(0.3, 0.6, 1.0), duration_grid=(20,), j_max=1, ts_seconds=0.1)
        _, _, s = d.run_scheduler(contexts, model=model, cfg=cfg, alpha=0.5)
        return s

    rows = []
    if args.sweep == "budget":
        vals = args.values or [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        for b in vals:
            s = run(args.lam, float(b))
            rows.append({"budget": float(b), "utility_uncertainty": s["time_avg_uncertainty"],
                         "bandwidth": s["time_avg_bandwidth"], "mean_backlog": s["mean_backlog"]})
            print(f"  B̄_bgt={b:<5}: U={s['time_avg_uncertainty']:.4f}  B̄={s['time_avg_bandwidth']:.3f}")
        xcol, xlabel = "budget", "communication budget  B̄_bgt"
    else:
        vals = args.values or [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        for lam in vals:
            s = run(float(lam), args.budget)
            rows.append({"lam": float(lam), "utility_uncertainty": s["time_avg_uncertainty"],
                         "bandwidth": s["time_avg_bandwidth"], "mean_backlog": s["mean_backlog"]})
            print(f"  Λ={lam:<6}: U={s['time_avg_uncertainty']:.4f}  backlog={s['mean_backlog']:.3g}")
        xcol, xlabel = "lam", "Lyapunov weight Λ (convergence knob)"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"tradeoff_{args.sweep}.csv", index=False)

    plt, fig, (ax1, ax2) = make_axes(1, 2, figsize=(11.0, 4.0))
    ax1.plot(df[xcol], df["utility_uncertainty"], "o-", color="tab:blue", label="U (uncertainty)")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("time-avg uncertainty  U̅"); ax1.legend()
    ax1.set_title("Utility vs " + ("budget (perception–communication)" if args.sweep == "budget" else "Λ (~O(1/Λ))"))
    if args.sweep == "lam":
        ax1.set_xscale("log")
    ax2.plot(df["bandwidth"], df["utility_uncertainty"], "s-", color="tab:red")
    ax2.set_xlabel("bandwidth used  B̄"); ax2.set_ylabel("time-avg uncertainty  U̅")
    ax2.set_title("Trade-off frontier (U vs B̄)")
    path = save(plt, fig, out / f"tradeoff_{args.sweep}.png")
    print(f"[tradeoff] wrote {out/('tradeoff_'+args.sweep+'.csv')} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
