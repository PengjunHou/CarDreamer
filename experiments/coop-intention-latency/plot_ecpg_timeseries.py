"""Plot ECPG(t) per latency: one chart per k, 4 strategies, mean +/- std band.

Each episode's per-frame ECPG is normalized to episode-progress [0,1] (episodes
vary in length; naive start-align + min-truncate loses the signal), then averaged
across episodes per (rule, k). Only k layers with all 4 rules present are plotted.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ecpg_from_geometry import frame_ecpg, frame_epu, load  # noqa: E402

RULES = ["all", "nearest2", "nearest1", "random1"]
COLORS = {"all": "tab:blue", "nearest2": "tab:green", "nearest1": "tab:orange", "random1": "tab:red"}
N = 50  # progress-normalized sample points
METRICS = {"ecpg": (frame_ecpg, "ECPG(t)", "Per-frame ECPG (cooperative gain)"),
           "epu": (frame_epu, "EPU(t)", "Ego Perception Uncertainty (lower = better)")}


def curves_for(geom_dir, rule, k, fn):
    path = os.path.join(geom_dir, f"{rule}_k{k}.jsonl")
    if not os.path.isfile(path):
        return None
    eps = load(path)
    out = []
    for frames in eps.values():
        c = [fn(f) for f in frames]
        if len(c) < 2:
            continue
        out.append(np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(c)), c))
    return np.array(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="logdir/ecpg_geom")
    ap.add_argument("--out", default="experiments/coop-intention-latency/figures")
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--metric", choices=list(METRICS), default="ecpg")
    args = ap.parse_args()
    fn, ylabel, title_prefix = METRICS[args.metric]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, N)
    plotted = []
    for k in range(args.max_k + 1):
        data = {r: curves_for(args.geom, r, k, fn) for r in RULES}
        if any(v is None for v in data.values()):
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for r in RULES:
            R = data[r]
            m, s = R.mean(0), R.std(0)
            ax.plot(x, m, color=COLORS[r], lw=2, label=f"{r} (n={len(R)})")
            ax.fill_between(x, m - s, m + s, color=COLORS[r], alpha=0.15)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("episode progress (0 = start, 1 = end)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} -- latency k={k} ({k*100} ms)")
        ax.legend(title="rule", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(args.out, f"{args.metric}_timeseries_k{k}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        plotted.append(k)
        print(f"wrote {out}")
    print(f"plotted {args.metric} k = {plotted}")


if __name__ == "__main__":
    main()
