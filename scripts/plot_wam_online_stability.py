#!/usr/bin/env python3
"""Queue-stability figures from an online Lyapunov trace (Phase 4).

Reads a per-step trace CSV written by ``run_wam_lyapunov_online_episode.py`` (columns ``step``,
``allocated_bandwidth``, ``z``, ``total_backlog``) and draws the three Proposition-1 witnesses:
(a) cumulative time-average allocated bandwidth converging at/below the budget, (b) ``Z(t)/t -> 0``
(mean-rate stability of the budget virtual queue), (c) total backlog ``Σ Q_m`` bounded.

Example
-------
    python scripts/plot_wam_online_stability.py \
        --trace outputs/wam_lyapunov_online/lyapunov/trace_r0.csv --budget 0.4 \
        --out outputs/wam_lyapunov_online/stability_lyapunov.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot online Lyapunov queue-stability witnesses.")
    ap.add_argument("--trace", type=Path, required=True, help="per-step trace CSV (run_wam_lyapunov_online_episode)")
    ap.add_argument("--budget", type=float, default=0.4, help="B̄_bgt budget line")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.trace)
    t = df["step"].to_numpy() + 1.0
    cum_bandwidth = np.cumsum(df["allocated_bandwidth"].to_numpy()) / t
    z_over_t = df["z"].to_numpy() / t if "z" in df.columns else np.zeros_like(t)
    backlog = df["total_backlog"].to_numpy() if "total_backlog" in df.columns else np.zeros_like(t)

    out = args.out or args.trace.with_suffix(".stability.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    a1.plot(df["step"], cum_bandwidth, label="cumulative B̄(t)")
    a1.axhline(args.budget, color="tab:red", ls="--", label=f"budget {args.budget}")
    a1.set_xlabel("slot t"); a1.set_ylabel("bandwidth ratio"); a1.set_title("B̄ ≤ B̄_bgt"); a1.legend()
    a2.plot(df["step"], z_over_t, color="tab:green")
    a2.set_xlabel("slot t"); a2.set_ylabel("Z(t)/t"); a2.set_title("Z(t)/t → 0 (mean-rate stable)")
    a3.plot(df["step"], backlog, color="tab:red")
    a3.set_xlabel("slot t"); a3.set_ylabel("Σ Q_m (bits)"); a3.set_title("backlog bounded")
    fig.tight_layout()
    fig.savefig(out, dpi=120)

    b_bar = float(cum_bandwidth[-1]) if len(cum_bandwidth) else 0.0
    z_t = float(z_over_t[-1]) if len(z_over_t) else 0.0
    print(f"[stability] B̄={b_bar:.3f} (budget {args.budget})  Z(T)/T={z_t:.4g}  "
          f"mean_backlog={float(np.mean(backlog)):.3g}  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
