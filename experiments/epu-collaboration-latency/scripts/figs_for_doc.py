"""Generate the report figures (matplotlib, no CARLA) from the archived CSVs."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

RULES = ["all", "nearest2", "nearest1", "random1"]
LAB = {"all": "all", "nearest2": "nearest-2", "nearest1": "nearest-1", "random1": "random-1"}
COL = {"all": "#2a78d6", "nearest2": "#1baf7a", "nearest1": "#eb6834", "random1": "#e34948"}
DASH = {"all": "-", "nearest2": "--", "nearest1": ":", "random1": "-."}
KS = list(range(11))


def _smooth(y, lo=0.0, hi=None):
    """Savitzky-Golay denoise + cubic-spline interpolation, clipped to a valid range."""
    y = np.asarray(y, float)
    yy = savgol_filter(y, window_length=5, polyorder=2, mode="interp")
    xf = np.linspace(KS[0], KS[-1], 200)
    yf = make_interp_spline(KS, yy, k=3)(xf)
    if lo is not None:
        yf = np.clip(yf, lo, hi if hi is not None else yf.max())
    return xf, yf


def read_wide(path):
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row["rule"]] = [float(row[f"k{k}"]) * 1000.0 for k in KS]
    return rows


def read_driving(path):
    d = {m: {r: [None] * 11 for r in RULES} for m in ["collision_rate", "success_rate", "mean_speed"]}
    with open(path) as f:
        for row in csv.DictReader(f):
            rr, k = row["rule"], int(row["k"])
            for m in d:
                d[m][rr][k] = float(row[m])
    return d


# Fig 1: IU vs latency (freshness, simple env)
fresh = read_wide(os.path.join(RES, "simple_freshness", "iu_freshness_table.csv"))
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for r in RULES:
    xf, yf = _smooth(fresh[r], lo=0.0)          # smoothed trend
    ax.plot(xf, yf, DASH[r], color=COL[r], lw=2.2, label=LAB[r])
    ax.plot(KS, fresh[r], "o", color=COL[r], ms=4, alpha=0.35)  # faint raw points
ax.set_xlabel("communication latency k (steps; 1 step = 100 ms)")
ax.set_ylabel(r"mean IU ($\times 10^{-3}$, lower = better)")
ax.set_title("IU vs latency — simple env, freshness formula (τ=1 s)")
ax.set_xticks(KS); ax.grid(alpha=0.3); ax.legend(title="collaboration rule")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_iu_vs_latency_freshness.png"), dpi=150); plt.close(fig)

# Fig 2: driving metrics 3-panel (simple env)
dr = read_driving(os.path.join(RES, "driving_simple.csv"))
titles = {"collision_rate": "Collision rate (lower better)",
          "success_rate": "Success rate (higher better)",
          "mean_speed": "Mean speed (m/s)"}
fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)
for ax, m in zip(axes, ["collision_rate", "success_rate", "mean_speed"]):
    for r in RULES:
        ax.plot(KS, dr[m][r], DASH[r], color=COL[r], lw=2, marker="o", ms=3.5, label=LAB[r])
    ax.set_ylabel(titles[m]); ax.grid(alpha=0.3)
axes[0].legend(title="rule", fontsize=8, ncol=2)
axes[-1].set_xlabel("communication latency k (steps; 1 step = 100 ms)")
axes[-1].set_xticks(KS)
fig.suptitle("Driving metrics vs latency — simple env (right_turn_hard.ckpt)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_driving_metrics.png"), dpi=150); plt.close(fig)

# Fig 3: formula comparison for the `all` rule (gamma flat vs freshness monotone)
gam = read_wide(os.path.join(RES, "simple_gamma_iu_table.csv"))
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(KS, gam["all"], "--", color="#888780", lw=2, marker="s", ms=4, label="position-residual γ (old) — flat")
ax.plot(KS, fresh["all"], "-", color="#2a78d6", lw=2, marker="o", ms=4, label="freshness trust(k) (new) — monotone")
ax.set_xlabel("communication latency k (steps; 1 step = 100 ms)")
ax.set_ylabel(r"mean IU, rule=all ($\times 10^{-3}$)")
ax.set_title("Latency term: position-residual γ vs freshness trust(k) — rule=all, simple env")
ax.set_xticks(KS); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_formula_comparison.png"), dpi=150); plt.close(fig)

print("wrote:", os.listdir(FIG))
