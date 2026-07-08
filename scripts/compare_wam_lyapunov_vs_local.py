#!/usr/bin/env python3
"""No-degradation guarantee (V2X paper Sec IV.C): the installed chunk's cost rate never exceeds the
always-feasible maximal-duration local-only baseline. At every decision epoch we score both under the same
queue state and assert ``selected ≤ local``.

    python scripts/compare_wam_lyapunov_vs_local.py --steps 200 --out-dir outputs/wam_lyapunov_analysis
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

    from car_dreamer.toolkit.wam import (
        LyapunovScheduler, WorldActionScorer, action_cost_rate, local_only_chunk,
    )

    d = driver()
    model = load_model(args.stage1_ckpt)
    contexts = d.build_synthetic_contexts(steps=args.steps, seed=0)
    cfg = scheduler_config(lam=args.lam, budget_bandwidth=args.budget, ts_seconds=0.1)
    scorer = WorldActionScorer(perception_model=model, alpha=0.5)
    sched = LyapunovScheduler(cfg, scorer, request_vehicle_id=int(contexts[0].ego.actor_id))

    epochs = []
    for step in range(len(contexts)):
        ctx = contexts[step]
        if sched.is_decision_epoch(step, None):
            # local-only baseline scored under the *current* (pre-plan) queue state
            local = local_only_chunk(n_slots=cfg.F_max_slots, n_min=cfg.n_min_slots)
            lroll = scorer.score_chunk(ctx, local)
            lbr = action_cost_rate(
                horizon_slots=local.horizon_slots, lam=cfg.lam, c0=cfg.c0,
                per_slot_uncertainty=lroll.per_slot_uncertainty, per_slot_bandwidth=lroll.per_slot_bandwidth,
                z=sched.lyap.z.value, link_backlogs=sched.lyap.backlogs(),
                per_member_arrival_bits=lroll.per_member_predicted_load_bits,
                per_member_service_bits=lroll.per_member_predicted_service_bits,
            )
            _, br, chunk, _ = sched.plan(ctx, step)
            epochs.append({"step": step, "selected_cost": br.total, "local_cost": lbr.total,
                           "selected_is_local": float(all(sa.is_local_only for sa in chunk.sub_actions))})
        sched.observe_slot(ctx, step)

    df = pd.DataFrame(epochs)
    violations = int((df["selected_cost"] > df["local_cost"] + 1e-9).sum())
    ok = violations == 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "no_degradation.csv", index=False)

    plt, fig, ax = make_axes(1, 1, figsize=(8.0, 4.2))
    ax.plot(df["step"], df["local_cost"], "s--", color="tab:gray", label="local-only baseline cost rate")
    ax.plot(df["step"], df["selected_cost"], "o-", color="tab:blue", label="installed chunk cost rate")
    ax.set_xlabel("decision epoch (slot)"); ax.set_ylabel("cost rate  g(a)")
    ax.set_title(f"No-degradation: installed ≤ local-only  ({'PASS' if ok else f'{violations} VIOLATIONS'})")
    ax.legend()
    path = save(plt, fig, out / "no_degradation.png")
    print(f"[no-degradation] epochs={len(df)}  violations={violations}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  wrote {out/'no_degradation.csv'} and {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
