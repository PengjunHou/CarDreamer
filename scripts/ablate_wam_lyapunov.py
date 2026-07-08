#!/usr/bin/env python3
"""Ablation sweeps for the Lyapunov policy search (V2X paper Sec IV).

Sweeps a grid over the scheduler knobs — Λ (utility/backlog), B̄_bgt (perception/communication),
c0 (decision cadence), α (motion/coverage mix), F_max (horizon cap), and J (chunk length) — and tabulates
the resulting time-average uncertainty ``U_c``, bandwidth ``B̄``, backlog, epochs/replans, and budget
witness ``Z(T)/T``. By default it sweeps Λ × B̄_bgt; pass ``--axis`` to sweep a single knob.

    python scripts/ablate_wam_lyapunov.py --steps 120 --out-dir outputs/wam_lyapunov_analysis
    python scripts/ablate_wam_lyapunov.py --axis alpha --values 0.0 0.25 0.5 0.75 1.0 --steps 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _wam_lyapunov_common import driver, load_model, make_axes, save, scheduler_config

_KNOBS = {
    "lam": ("lam", float), "budget": ("budget_bandwidth", float), "c0": ("c0", float),
    "alpha": (None, float), "f_max": ("F_max_slots", int), "j": ("j_max", int),
}


def _run(d, model, contexts, base, axis, value):
    kw = dict(base)
    alpha = base.get("_alpha", 0.5)
    attr, cast = _KNOBS[axis]
    if attr is None:  # alpha lives on the scorer, not SchedulerConfig
        alpha = cast(value)
    else:
        kw[attr] = cast(value)
    cfg = scheduler_config(**{k: v for k, v in kw.items() if not k.startswith("_")})
    _, _, s = d.run_scheduler(contexts, model=model, cfg=cfg, alpha=alpha)
    return {axis: value, "u_c": s["u_c"], "time_avg_uncertainty": s["time_avg_uncertainty"],
            "time_avg_bandwidth": s["time_avg_bandwidth"], "mean_backlog": s["mean_backlog"],
            "epochs": s["epochs"], "replans": s["replans"], "z_over_t": s["z_over_t"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=sorted(_KNOBS), default=None, help="single knob to sweep (default: Λ × B̄_bgt grid)")
    ap.add_argument("--values", type=float, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--stage1-ckpt", default=None)
    ap.add_argument("--out-dir", default="outputs/wam_lyapunov_analysis")
    args = ap.parse_args()

    import pandas as pd

    d = driver()
    model = load_model(args.stage1_ckpt)
    contexts = d.build_synthetic_contexts(steps=args.steps, seed=0)
    base = dict(lam=1.0, budget_bandwidth=0.4, c0=0.5, F_max_slots=20, n_min_slots=5, j_max=2, ts_seconds=0.1, _alpha=0.5)

    rows = []
    if args.axis is not None:
        vals = args.values or ([0.1, 0.3, 1.0, 3.0, 10.0] if args.axis == "lam" else [0.0, 0.25, 0.5, 0.75, 1.0])
        for v in vals:
            rows.append(_run(d, model, contexts, base, args.axis, v))
            print(f"  {args.axis}={v}: U_c={rows[-1]['u_c']:.4f} B̄={rows[-1]['time_avg_bandwidth']:.3f}")
        sweep_name = args.axis
    else:  # default 2-D grid Λ × B̄_bgt
        for lam in [0.3, 1.0, 3.0]:
            for bud in [0.3, 0.5]:
                b2 = dict(base); b2["lam"] = lam
                r = _run(d, model, contexts, b2, "budget", bud)
                r["lam"] = lam
                rows.append(r)
                print(f"  Λ={lam} B̄_bgt={bud}: U_c={r['u_c']:.4f} B̄={r['time_avg_bandwidth']:.3f} Z/T={r['z_over_t']:.3g}")
        sweep_name = "lam_x_budget"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"ablation_{sweep_name}.csv", index=False)

    plt, fig, ax = make_axes(1, 1, figsize=(8.0, 4.2))
    if args.axis is not None:
        ax.plot(df[args.axis], df["time_avg_uncertainty"], "o-", label="U̅")
        ax.plot(df[args.axis], df["time_avg_bandwidth"], "s-", label="B̄")
        ax.set_xlabel(args.axis)
    else:
        for lam, g in df.groupby("lam"):
            ax.plot(g["budget"], g["time_avg_uncertainty"], "o-", label=f"U̅ (Λ={lam})")
        ax.set_xlabel("B̄_bgt")
    ax.set_ylabel("metric"); ax.set_title(f"Ablation: {sweep_name}"); ax.legend(fontsize=8)
    path = save(plt, fig, out / f"ablation_{sweep_name}.png")
    print(f"[ablate] wrote {out/('ablation_'+sweep_name+'.csv')} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
