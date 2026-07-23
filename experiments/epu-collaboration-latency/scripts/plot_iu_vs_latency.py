"""Plot mean entropy-IU vs communication latency, one line per collaboration rule.

Reads the iu_entropy_table.csv produced by iu_entropy_table.py (columns: rule, k0..k10)
and draws IU (x1e-3, lower = better) against latency k. This is the strategy x latency
comparison view (complements the per-timestep IU(t) curves in plot_ecpg_timeseries.py).
"""

import argparse
import csv
import os


RULES = ["all", "nearest2", "nearest1", "random1"]
COLORS = {"all": "tab:blue", "nearest2": "tab:green", "nearest1": "tab:orange", "random1": "tab:red"}
LABELS = {
    "all": "all (every flow vehicle)",
    "nearest2": "nearest-2",
    "nearest1": "nearest-1",
    "random1": "random-1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/coop-intention-latency/group_full/iu_entropy_table.csv")
    ap.add_argument("--out", default="experiments/coop-intention-latency/group_full/iu_vs_latency.png")
    ap.add_argument("--title", default="Intention Uncertainty vs communication latency\n(richer scene: +background traffic; lower = better)")
    args = ap.parse_args()

    rows = {}
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        ks = [c for c in reader.fieldnames if c.startswith("k")]
        for row in reader:
            rows[row["rule"]] = row
    kvals = [int(k[1:]) for k in ks]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for rule in RULES:
        if rule not in rows:
            continue
        y = []
        for k in ks:
            v = rows[rule].get(k, "")
            y.append(float(v) * 1000.0 if v not in ("", "nan") else float("nan"))
        ax.plot(kvals, y, marker="o", lw=2, color=COLORS[rule], label=LABELS[rule])

    ax.set_xlabel("communication latency k (steps; 1 step = 100 ms)")
    ax.set_ylabel(r"mean IU  ($\times 10^{-3}$, lower = better)")
    ax.set_title(args.title)
    ax.set_xticks(kvals)
    ax.grid(alpha=0.3)
    ax.legend(title="collaboration rule")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
