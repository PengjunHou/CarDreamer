"""Three-panel comparison: Intention (IU), Coverage (U^cov), and Combined uncertainty vs latency.

Reads the three tables from iu_coverage_table.py (results/combined/{iu,coverage,combined}_table.csv),
one line per collaboration rule. Values are shown x1e-3. Faint raw points + optional smoothing.
"""
import argparse
import csv
import os

import numpy as np

RULES = ["all", "nearest2", "nearest1", "random1"]
LAB = {"all": "all", "nearest2": "nearest-2", "nearest1": "nearest-1", "random1": "random-1"}
COL = {"all": "#2a78d6", "nearest2": "#1baf7a", "nearest1": "#eb6834", "random1": "#e34948"}
DASH = {"all": "-", "nearest2": "--", "nearest1": ":", "random1": "-."}
KS = list(range(11))


def read_wide(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["rule"]] = [float(row[f"k{k}"]) * 1000.0 for k in KS]
    return rows


def smooth(y):
    from scipy.interpolate import make_interp_spline
    from scipy.signal import savgol_filter
    yy = savgol_filter(np.asarray(y, float), 5, 2, mode="interp")
    xf = np.linspace(0, 10, 200)
    return xf, np.clip(make_interp_spline(KS, yy, k=3)(xf), 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/epu-collaboration-latency/results/combined")
    ap.add_argument("--out", default="experiments/epu-collaboration-latency/figures/fig6_iu_coverage_combined.png")
    ap.add_argument("--smooth", action="store_true")
    args = ap.parse_args()

    panels = [
        ("iu_table.csv", r"Intention IU  ($\times10^{-3}$)"),
        ("coverage_table.csv", r"Coverage $U^{cov}$  ($\times10^{-3}$)"),
        ("combined_table.csv", r"Combined  ($\times10^{-3}$)"),
    ]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.0), sharex=True)
    for ax, (fname, ylabel) in zip(axes, panels):
        data = read_wide(os.path.join(args.dir, fname))
        for r in RULES:
            if r not in data:
                continue
            y = data[r]
            if args.smooth:
                xf, yf = smooth(y)
                ax.plot(xf, yf, DASH[r], color=COL[r], lw=2.2, label=LAB[r])
                ax.plot(KS, y, "o", color=COL[r], ms=4, alpha=0.35)
            else:
                ax.plot(KS, y, DASH[r], color=COL[r], lw=2, marker="o", ms=4, label=LAB[r])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_title("Intention vs Coverage vs Combined uncertainty (simple env, lower = better)")
    axes[0].legend(title="rule", fontsize=8, ncol=2)
    axes[-1].set_xlabel("communication latency k (steps; 1 step = 100 ms)")
    axes[-1].set_xticks(KS)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
