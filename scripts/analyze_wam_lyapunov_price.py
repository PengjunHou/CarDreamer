#!/usr/bin/env python3
"""Adaptive shadow-price convergence (V2X paper Sec IV.C): the virtual queue ``Z(t)`` is the learned
budget price. This plots the ``Z(t)`` trajectory (and the total backlog) over an episode — ``Z`` rises
while consumption exceeds the budget and drains when it is frugal, converging to the level at which the
long-term budget binds, i.e. it is an online estimate of the shadow price a scalarized objective would need
hand-tuned.

    python scripts/analyze_wam_lyapunov_price.py --steps 300 --budget 0.4 --out-dir outputs/wam_lyapunov_analysis
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

    import pandas as pd

    d = driver()
    model = load_model(args.stage1_ckpt)
    contexts = d.build_synthetic_contexts(steps=args.steps, seed=0)
    cfg = scheduler_config(lam=args.lam, budget_bandwidth=args.budget, ts_seconds=0.1)
    _, rows, _ = d.run_scheduler(contexts, model=model, cfg=cfg, alpha=0.5)
    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt, fig, ax = make_axes(1, 1, figsize=(8.0, 4.2))
    ax.plot(df["step"], df["z"], color="tab:green", label="Z(t)  (spectrum shadow price)")
    ax.set_xlabel("slot t"); ax.set_ylabel("Z(t)")
    ax2 = ax.twinx()
    ax2.plot(df["step"], df["allocated_bandwidth"], color="tab:blue", alpha=0.4, label="allocated bandwidth")
    ax2.axhline(args.budget, color="k", ls="--", alpha=0.6)
    ax2.set_ylabel("allocated bandwidth ratio")
    ax.set_title("Adaptive budget price Z(t) vs allocation")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    path = save(plt, fig, out / "price.png")
    print(f"[price] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
