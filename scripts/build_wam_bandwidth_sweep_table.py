#!/usr/bin/env python3
"""Build the WAM Phase-3 bandwidth-sweep table + figure from sweep_wam_bandwidth.sh outputs.

Reads the shared ego-only baseline (``<sweep>/ego_only/summary.json``) and one lyapunov summary per
bandwidth (``<sweep>/bw_<hz>/lyapunov/summary.json``), then writes:
  - ``sweep_table.csv``  : one row per bandwidth (lyapunov metrics + delta vs the shared ego-only)
  - ``sweep_bandwidth.png`` : cooperation benefit / rationing / stability vs log-bandwidth

Usage
-----
    python scripts/build_wam_bandwidth_sweep_table.py --sweep-dir outputs/wam_bandwidth_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _m(summary: Dict, key: str) -> float:
    return float(summary["metrics"][key]["mean"])


def _s(summary: Dict, key: str) -> float:
    return float(summary["metrics"][key]["std"])


def _hz_label(hz: int) -> str:
    if hz >= 1_000_000:
        return f"{hz // 1_000_000}MHz"
    return f"{hz // 1000}kHz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="outputs/wam_bandwidth_sweep")
    ap.add_argument("--budget", type=float, default=0.4, help="B̄_bgt bandwidth-ratio budget line")
    ap.add_argument("--bandwidths", default="6000000,3000000,1000000,500000,100000,50000,20000,10000")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    ego = _load(sweep / "ego_only" / "local_only" / "summary.json")
    if ego is None:
        print(f"[error] missing shared ego-only baseline at {sweep/'ego_only'/'local_only'/'summary.json'}")
        return 1
    u_ego, u_ego_std = _m(ego, "time_avg_uncertainty"), _s(ego, "time_avg_uncertainty")
    cov_ego = _m(ego, "time_avg_coverage_uncertainty")
    mot_ego = _m(ego, "time_avg_motion_uncertainty")

    bws = [int(x) for x in args.bandwidths.split(",") if x]
    rows: List[Dict] = []
    for hz in bws:
        ly = _load(sweep / f"bw_{hz}" / "lyapunov" / "summary.json")
        if ly is None:
            print(f"[skip] bw={hz}: no lyapunov summary yet")
            continue
        rows.append({
            "bandwidth_hz": hz,
            "bandwidth": _hz_label(hz),
            "coop_step_rate": round(_m(ly, "coop_step_rate"), 4),
            "time_avg_bandwidth": round(_m(ly, "time_avg_bandwidth"), 4),
            "z_over_t": round(_m(ly, "z_over_t"), 4),
            "mean_backlog": round(_m(ly, "mean_backlog"), 4),
            "U_lyap": round(_m(ly, "time_avg_uncertainty"), 4),
            "U_lyap_std": round(_s(ly, "time_avg_uncertainty"), 4),
            "U_ego": round(u_ego, 4),
            "U_ego_std": round(u_ego_std, 4),
            "delta_U": round(_m(ly, "time_avg_uncertainty") - u_ego, 4),
            "delta_U_coverage": round(_m(ly, "time_avg_coverage_uncertainty") - cov_ego, 4),
            "delta_U_motion": round(_m(ly, "time_avg_motion_uncertainty") - mot_ego, 4),
            "rounds": ly.get("rounds", 0),
        })

    if not rows:
        print("[error] no bandwidth rows found -- run the sweep first")
        return 1

    out_csv = sweep / "sweep_table.csv"
    fields = ["bandwidth_hz", "bandwidth", "coop_step_rate", "time_avg_bandwidth", "z_over_t",
              "mean_backlog", "U_lyap", "U_lyap_std", "U_ego", "U_ego_std", "delta_U",
              "delta_U_coverage", "delta_U_motion", "rounds"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out_csv}\n")
    # echo the table
    print(f"{'bandwidth':>10} {'coop':>6} {'B̄':>7} {'Z/T':>7} {'U_lyap':>8} {'U_ego':>7} {'ΔU':>8}")
    for r in rows:
        print(f"{r['bandwidth']:>10} {r['coop_step_rate']:>6.3f} {r['time_avg_bandwidth']:>7.3f} "
              f"{r['z_over_t']:>7.3f} {r['U_lyap']:>8.3f} {r['U_ego']:>7.3f} {r['delta_U']:>8.3f}")
    print(f"\n(shared ego-only baseline: U={u_ego:.3f}±{u_ego_std:.3f})")

    # ---- figure: metrics vs log-bandwidth ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable ({exc}); CSV written, skipping figure")
        return 0

    rows_lo2hi = sorted(rows, key=lambda r: r["bandwidth_hz"])
    x = [r["bandwidth_hz"] for r in rows_lo2hi]
    labels = [r["bandwidth"] for r in rows_lo2hi]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.plot(x, [r["delta_U"] for r in rows_lo2hi], "o-", color="C0", label="Δ total U")
    ax.plot(x, [r["delta_U_coverage"] for r in rows_lo2hi], "s--", color="C2", label="Δ coverage U")
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔU  (lyapunov − ego-only)")
    ax.set_title("Cooperation benefit vs bandwidth\n(more negative = bigger benefit)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(args.budget, color="C3", lw=1.0, ls="--", label=f"budget B̄={args.budget}")
    ax.plot(x, [r["coop_step_rate"] for r in rows_lo2hi], "o-", color="C1", label="coop rate")
    ax.plot(x, [r["time_avg_bandwidth"] for r in rows_lo2hi], "^-", color="C4", label="B̄ (time-avg bw)")
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("rate / ratio")
    ax.set_title("Rationing vs bandwidth\n(P2 throttles coop as bw shrinks)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, [r["z_over_t"] for r in rows_lo2hi], "o-", color="C5", label="Z(T)/T")
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Z(T)/T")
    ax.set_title("Queue stability vs bandwidth\n(→0 = budget-bound & stable)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("WAM Phase-3 bandwidth sweep: P2 (Lyapunov) vs ego-only  ·  one v3 U_φ, one shared ego baseline",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = sweep / "sweep_bandwidth.png"
    fig.savefig(out_png, dpi=130)
    print(f"[ok] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
