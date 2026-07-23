"""Aggregate per-episode driving metrics (collision / success / speed) per (rule, latency).

Reads the eval metrics.jsonl written by each geometry-sweep config
(<geom>/run_<rule>_k<k>/metrics.jsonl) and computes, per config:
    n_ep, collision_rate, success_rate, mean_speed
This is the SAME data that produced the EPU tables (the very eval runs), just read from
the eval side instead of the recorded geometry. No CARLA needed.
"""

import argparse
import csv
import glob
import json
import os
import re


def agg_config(run_dir):
    mfile = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(mfile):
        return None
    coll = []; succ = []; spd = []
    for line in open(mfile):
        r = json.loads(line)
        if "stats/sum_is_collision" not in r:
            continue
        coll.append(1.0 if r["stats/sum_is_collision"] > 0 else 0.0)
        succ.append(1.0 if r.get("stats/sum_destination_reached", 0) > 0 else 0.0)
        spd.append(float(r.get("stats/mean_speed_norm", 0.0)))
    n = len(coll)
    if n == 0:
        return None
    return {"n_ep": n,
            "collision_rate": sum(coll) / n,
            "success_rate": sum(succ) / n,
            "mean_speed": sum(spd) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="logdir/ecpg_geom")
    ap.add_argument("--out", default="experiments/coop-intention-latency/driving_simple.csv")
    args = ap.parse_args()

    pat = re.compile(r"run_(all|nearest1|nearest2|random1)_k(\d+)$")
    rows = {}
    for d in sorted(glob.glob(os.path.join(args.geom, "run_*_k*"))):
        m = pat.search(os.path.basename(d))
        if not m or "smoke" in d:
            continue
        rule, k = m.group(1), int(m.group(2))
        a = agg_config(d)
        if a:
            rows[(rule, k)] = a

    rules = ["all", "nearest2", "nearest1", "random1"]
    with open(args.out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["rule", "k", "n_ep", "collision_rate", "success_rate", "mean_speed"])
        for rule in rules:
            for k in range(11):
                a = rows.get((rule, k))
                if a:
                    w.writerow([rule, k, a["n_ep"], round(a["collision_rate"], 3),
                                round(a["success_rate"], 3), round(a["mean_speed"], 3)])
    print(f"wrote {args.out}")

    for metric in ["collision_rate", "success_rate", "mean_speed"]:
        print(f"\n=== {metric} ===")
        print("rule     " + "  ".join(f"k{k:<2d}" for k in range(11)))
        for rule in rules:
            cells = []
            for k in range(11):
                a = rows.get((rule, k))
                cells.append(f"{a[metric]:5.2f}" if a else "  -- ")
            print(f"{rule:8s} " + "  ".join(cells))
    print("\nn_ep per config:")
    for rule in rules:
        ns = [str(rows[(rule, k)]["n_ep"]) if (rule, k) in rows else "-" for k in range(11)]
        print(f"  {rule:8s} " + " ".join(ns))


if __name__ == "__main__":
    main()
