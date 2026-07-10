#!/usr/bin/env python3
"""Render the WAM Phase-3 bandwidth sweep as THREE independent, publication-grade figures.

Unlike ``build_wam_bandwidth_sweep_table.py`` (which packs all three views into one 1x3 strip),
this script emits one standalone single-column figure per scientific claim, each exported as
editable SVG + PDF + PNG:

  fig1_coop_benefit   : ΔU (P2 - ego-only) vs bandwidth   -- cooperation benefit / coverage effect
  fig2_rationing      : coop-rate & time-avg B̄ vs budget  -- P2 rations coop to the physical link
  fig3_queue_stability: Z(T)/T vs bandwidth                -- Lyapunov virtual-queue stability

Reads ``<sweep>/sweep_table.csv`` (already produced by the sweep builder).

Usage
-----
    python scripts/plot_wam_bandwidth_panels.py --sweep-dir outputs/wam_bandwidth_sweep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MANDATORY: editable SVG text (Nature-figure API rule) ─────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Restrained palette (semantic, shared across the three figures).
C = {
    "total":    "#0F4D92",  # blue    -- total uncertainty delta (hero, fig 1)
    "coverage": "#8BCF8B",  # green   -- coverage delta (subordinate, fig 1)
    "coop":     "#E28E2C",  # orange  -- cooperation rate (hero, fig 2)
    "bw":       "#7C6CCF",  # violet  -- time-averaged bandwidth ratio (fig 2)
    "budget":   "#B64342",  # red     -- soft budget line (fig 2)
    "queue":    "#42949E",  # teal    -- Lyapunov virtual-queue ratio (hero, fig 3)
    "zero":     "#767676",  # neutral -- reference lines
}

FIGSIZE = (3.6, 3.0)   # single-column standalone figure


def load_rows(sweep: Path) -> List[Dict[str, float]]:
    with (sweep / "sweep_table.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "bandwidth":
                r[k] = float(v)
    return sorted(rows, key=lambda r: r["bandwidth_hz"])  # low -> high


def _log_bw_axis(ax, x, labels) -> None:
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Policy link bandwidth (Hz, log scale)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(width=0.8, length=3)


def save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_base}.{{svg,pdf,png}}")


def fig1_coop_benefit(rows, x, labels, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=C["zero"], lw=0.8, ls="--", zorder=1)
    ax.plot(x, [r["delta_U"] for r in rows], "o-", color=C["total"], lw=1.8,
            ms=5, label="Δ total U", zorder=3)
    ax.plot(x, [r["delta_U_coverage"] for r in rows], "s--", color=C["coverage"],
            lw=1.5, ms=4, label="Δ coverage U", zorder=2)
    _log_bw_axis(ax, x, labels)
    ax.set_ylabel("ΔU  (P2 − ego-only)")
    # direction cue placed in the empty upper-left region (no title/data collision)
    ax.text(0.03, 0.97, "lower = cooperation reduces U\n(rise = more objects admitted)",
            transform=ax.transAxes, fontsize=6.4, color=C["zero"], va="top", style="italic")
    ax.set_title("Cooperation benefit vs bandwidth", fontsize=8.5, pad=6)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.03, 0.5))
    save(fig, out)


def fig2_rationing(rows, x, labels, out: Path, budget: float) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(budget, color=C["budget"], lw=1.0, ls="--", zorder=1,
               label=f"budget B̄ = {budget:g}")
    ax.plot(x, [r["coop_step_rate"] for r in rows], "o-", color=C["coop"], lw=1.8,
            ms=5, label="coop-step rate", zorder=3)
    ax.plot(x, [r["time_avg_bandwidth"] for r in rows], "^-", color=C["bw"], lw=1.6,
            ms=5, label="B̄  (time-avg bw ratio)", zorder=2)
    _log_bw_axis(ax, x, labels)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("rate / bandwidth ratio")
    ax.set_title("P2 rations cooperation to the link", fontsize=8.5, pad=6)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, out)


def fig3_queue_stability(rows, x, labels, out: Path, budget: float) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=C["zero"], lw=0.8, ls=":", zorder=1)
    # overspend rate B̄ - B̄_bgt: in the binding regime Z(T)/T == this line exactly (Prop 1);
    # where it goes negative the budget is slack and Z(T)/T detaches toward 0.
    ax.plot(x, [r["time_avg_bandwidth"] - budget for r in rows], "s--", color=C["zero"],
            lw=1.4, ms=4, zorder=2, label="B̄ − B̄_bgt (overspend)")
    ax.plot(x, [r["z_over_t"] for r in rows], "o-", color=C["queue"], lw=1.8,
            ms=5, zorder=3, label="Z(T)/T (virtual queue)")
    _log_bw_axis(ax, x, labels)
    ax.set_ylabel("bandwidth-ratio units / slot")
    ax.text(0.03, 0.50, "Z(T)/T → 0 :\nbudget-bound\n& stable",
            transform=ax.transAxes, fontsize=6.4, color=C["queue"], va="center", style="italic")
    ax.text(0.97, 0.30, "budget binds: Z grows\nat rate B̄ − B̄_bgt\n(soft cap overspent)",
            transform=ax.transAxes, fontsize=6.4, color=C["zero"], va="center", ha="right")
    ax.set_title("Virtual-queue (budget) stability vs bandwidth", fontsize=8.5, pad=6)
    ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(0.02, 0.88))
    save(fig, out)


def fig4_physical_backlog(rows, x, labels, out: Path) -> None:
    """Physical link backlog Q_m (payload units) -- the *hard* queue. Distinct from the virtual
    budget queue Z: this is the actual transmit backlog, and only blows up when the link cannot
    drain arrivals (leftmost, starved bandwidth)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(1.0, color=C["zero"], lw=0.8, ls=":", zorder=1)
    yb = [r["mean_backlog"] for r in rows]
    ax.plot(x, yb, "o-", color=C["queue"], lw=1.8, ms=5, zorder=3, label="mean Σ Q_m  (backlog)")
    # flag the one point where the physical queue actually backs up
    i_spike = max(range(len(yb)), key=lambda i: yb[i])
    ax.scatter([x[i_spike]], [yb[i_spike]], s=70, facecolor="none",
               edgecolor=C["budget"], lw=1.6, zorder=4)
    ax.annotate("link can't drain\narrivals → backs up",
                xy=(x[i_spike], yb[i_spike]), xytext=(10, -4), textcoords="offset points",
                fontsize=6.4, color=C["budget"], va="top",
                arrowprops=dict(arrowstyle="->", color=C["budget"], lw=0.8))
    _log_bw_axis(ax, x, labels)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("mean total link backlog  Σ Q_m  (payloads)")
    ax.text(0.30, 0.30, "≈ 1 payload in flight\n(steady, drained each slot)",
            transform=ax.transAxes, fontsize=6.4, color=C["zero"], va="center", style="italic")
    ax.set_title("Physical link-queue stability vs bandwidth", fontsize=8.5, pad=6)
    ax.legend(fontsize=6.8, loc="upper center")
    save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="outputs/wam_bandwidth_sweep")
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--out-subdir", default="panels")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    rows = load_rows(sweep)
    x = [r["bandwidth_hz"] for r in rows]
    labels = [r["bandwidth"] for r in rows]
    outdir = sweep / args.out_subdir

    fig1_coop_benefit(rows, x, labels, outdir / "fig1_coop_benefit")
    fig2_rationing(rows, x, labels, outdir / "fig2_rationing", args.budget)
    fig3_queue_stability(rows, x, labels, outdir / "fig3_queue_stability", args.budget)
    fig4_physical_backlog(rows, x, labels, outdir / "fig4_physical_backlog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
