"""Driving metrics vs latency as THREE separate, smoothed figures.

Reads driving_simple.csv and emits one figure per metric (collision / success / speed).
Each curve is smoothed with a Savitzky-Golay filter (denoise) + cubic-spline interpolation
(visual smoothness), clipped to the metric's valid range so the spline cannot overshoot.
Raw per-condition points are drawn faintly so the smoothing does not hide the noise.
"""
import argparse
import csv
import os

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

RULES = ["all", "nearest2", "nearest1", "random1"]
LAB = {"all": "all", "nearest2": "nearest-2", "nearest1": "nearest-1", "random1": "random-1"}
COL = {"all": "#2a78d6", "nearest2": "#1baf7a", "nearest1": "#eb6834", "random1": "#e34948"}
DASH = {"all": "-", "nearest2": "--", "nearest1": ":", "random1": "-."}
KS = np.arange(11)

METRICS = [
    ("collision_rate", "Collision rate (lower better)", (0.0, 1.0), "fig_collision_vs_latency.png"),
    ("success_rate", "Success rate (higher better)", (0.0, 1.0), "fig_success_vs_latency.png"),
    ("mean_speed", "Mean speed (m/s)", (0.0, None), "fig_speed_vs_latency.png"),
]


def read_driving(path):
    d = {m: {r: [np.nan] * 11 for r in RULES} for m in ["collision_rate", "success_rate", "mean_speed"]}
    with open(path) as f:
        for row in csv.DictReader(f):
            rr, k = row["rule"], int(row["k"])
            for m in d:
                d[m][rr][k] = float(row[m])
    return d


def smooth(y, lo, hi):
    y = np.asarray(y, float)
    yy = savgol_filter(y, window_length=5, polyorder=2, mode="interp")
    xf = np.linspace(0, 10, 200)
    yf = make_interp_spline(KS, yy, k=3)(xf)
    if lo is not None:
        yf = np.clip(yf, lo, hi if hi is not None else yf.max())
    return xf, yf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/driving_simple.csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--title-suffix", default="simple env (right_turn_hard.ckpt)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dr = read_driving(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    for metric, ylabel, (lo, hi), fname in METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for r in RULES:
            y = dr[metric][r]
            xf, yf = smooth(y, lo, hi)
            ax.plot(xf, yf, DASH[r], color=COL[r], lw=2.2, label=LAB[r])
            ax.plot(KS, y, "o", color=COL[r], ms=4, alpha=0.35)  # faint raw points
        ax.set_xlabel("communication latency k (steps; 1 step = 100 ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split(' (')[0]} vs latency — {args.title_suffix}")
        ax.set_xticks(KS)
        ax.grid(alpha=0.3)
        ax.legend(title="rule", fontsize=9, ncol=2)
        fig.tight_layout()
        out = os.path.join(args.outdir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
