"""Aggregate the coop-perception latency/selection matrix into CSV + plots + summary.

Usage: python analyze_coop_matrix.py [--outdir logdir/eval/coop]
"""

import argparse
import csv
import json
import os
import re


def cell_stats(path):
    n = dest = coll = timeout = 0
    scores, speeds = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "episode/score" in d:
                n += 1
                scores.append(d["episode/score"])
            dest += d.get("stats/sum_destination_reached", 0)
            coll += d.get("stats/sum_is_collision", 0)
            timeout += d.get("stats/sum_time_exceeded", 0)
            if "stats/mean_speed_norm" in d:
                speeds.append(d["stats/mean_speed_norm"])
    if n == 0:
        return None
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "episodes": n,
        "success": dest / n,
        "collision": coll / n,
        "timeout": timeout / n,
        "score": mean(scores),
        "speed": mean(speeds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="logdir/eval/coop")
    args = parser.parse_args()

    rows = []
    for tag in sorted(os.listdir(args.outdir)):
        metrics = os.path.join(args.outdir, tag, "metrics.jsonl")
        if not os.path.isfile(metrics):
            continue
        stats = cell_stats(metrics)
        if stats is None:
            continue
        m = re.fullmatch(r"(all|nearest1|nearest2|random1)_k(\d+)", tag)
        rule = m.group(1) if m else tag
        latency = int(m.group(2)) if m else None
        rows.append({"tag": tag, "rule": rule, "latency": latency, **stats})

    csv_path = os.path.join(args.outdir, "coop_matrix_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} cells)")

    # Markdown summary table
    md_path = os.path.join(args.outdir, "coop_matrix_summary.md")
    with open(md_path, "w") as f:
        f.write("| cell | eps | success | collision | timeout | score | speed |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['tag']} | {r['episodes']} | {r['success']:.1%} | {r['collision']:.1%} "
                f"| {r['timeout']:.1%} | {r['score']:.1f} | {r['speed']:.2f} |\n"
            )
    print(f"wrote {md_path}")

    # Plots: success & collision vs latency, one line per rule + reference hlines
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable, skipping plots")
        return

    matrix = [r for r in rows if r["latency"] is not None]
    refs = {r["tag"]: r for r in rows if r["latency"] is None}
    rules = sorted({r["rule"] for r in matrix})

    plot_specs = [
        ("success", "success_vs_latency.png", "success rate", (0, 1.05)),
        ("collision", "collision_vs_latency.png", "collision rate", (0, None)),
        ("speed", "speed_vs_latency.png", "avg speed (m/s)", (0, None)),
    ]
    for metric, fname, ylabel, ylim in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for rule in rules:
            pts = sorted((r["latency"], r[metric]) for r in matrix if r["rule"] == rule)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=rule)
        # "none" (no sharing) has no latency axis -> horizontal floor line.
        # Annotate it directly so it's readable even when it sits near y=0.
        if "none" in refs:
            y = refs["none"][metric]
            ax.axhline(y, linestyle="-.", color="gray", linewidth=1.2)
            ax.annotate(f"none (no sharing) = {y:.2f}", xy=(0, y), xytext=(0.3, y),
                        va="bottom", fontsize=8, color="gray")
        # desired speed reference on the speed plot
        if metric == "speed":
            ax.axhline(4.0, linestyle=":", color="green", linewidth=1)
            ax.annotate("desired_speed = 4", xy=(0, 4.0), xytext=(0.3, 4.0),
                        va="bottom", fontsize=8, color="green")
        for tag, style in [("ref_sfov_native", "--"), ("ref_fov_native", ":")]:
            if tag in refs:
                ax.axhline(refs[tag][metric], linestyle=style, color="gray", linewidth=1, label=tag)
        ax.set_xlabel("communication latency (steps, 100ms each)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"{ylabel} vs communication latency, by collaborator-selection rule")
        ax.legend(fontsize=8, title="rule")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(args.outdir, fname)
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
