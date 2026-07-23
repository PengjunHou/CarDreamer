"""IU with entropy U_self but a TIME-FRESHNESS latency term (replaces position-residual gamma).

Shared-vehicle residual uncertainty:  u_i = U_self * (1 - trust(k)),  trust(k) = exp(-k*DT/tau)
  - U_self is the SAME map-affordance route entropy as iu_entropy_table.u_self_entropy (unchanged).
  - k = per-config communication latency (rec["latency"]); trust is constant within a config.
  - k=0 -> trust=1 -> u=0 (fresh intention resolves the maneuver).
  - k large -> trust->0 -> u->U_self (stale intention as useless as the no-comm prior).
Unshared vehicles: u_i = U_self (full prior). Only the latency coupling differs from the gamma version;
this makes latency the direct, monotone driver, gated by U_self (only junction vehicles contribute).
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ecpg_from_geometry import relevance  # noqa: E402
from iu_entropy_table import u_self_entropy  # noqa: E402

import carla  # noqa: E402

DT = 0.1  # s per step (10 Hz)


def frame_epu_fresh(rec, cmap, cache, tau):
    ego_wp = rec["ego_wp"]; veh = rec["veh"]; shared = rec["shared"]
    k = int(rec["latency"])
    tk = math.exp(-k * DT / tau)   # freshness of a k-step-old broadcast
    epu = 0.0; any_rel = False
    for vid, v in veh.items():
        w = relevance(ego_wp, v)
        if w <= 0:
            continue
        any_rel = True
        key = (round(v["pos"][0], 1), round(v["pos"][1], 1), round(v["v"][0], 1), round(v["v"][1], 1))
        if key in cache:
            u_self = cache[key]
        else:
            u_self = u_self_entropy(cmap, v["pos"], v["v"]); cache[key] = u_self
        if vid in shared and shared[vid]:
            u = u_self * (1.0 - tk)     # fresh (k=0) -> 0 ; stale (k large) -> U_self
        else:
            u = u_self
        epu += w * u
    return epu, any_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="logdir/ecpg_geom")
    ap.add_argument("--port", type=int, default=2010)
    ap.add_argument("--tau", type=float, default=1.0, help="intention coherence timescale (s)")
    ap.add_argument("--out", default="experiments/coop-intention-latency/simple_freshness")
    args = ap.parse_args()

    client = carla.Client("127.0.0.1", args.port); client.set_timeout(60.0)
    world = client.get_world()
    if "Town03" not in world.get_map().name:
        world = client.load_world("Town03")
    cmap = world.get_map(); cache = {}
    os.makedirs(args.out, exist_ok=True)
    print(f"map: {cmap.name} | tau={args.tau}s | geom={args.geom}")

    stats = {}
    for f in sorted(glob.glob(os.path.join(args.geom, "*_k*.jsonl"))):
        tag = os.path.basename(f)[:-6]
        vals = []
        for line in open(f):
            rec = json.loads(line)
            epu, rel = frame_epu_fresh(rec, cmap, cache, args.tau)
            if rel:
                vals.append(epu)
        stats[tag] = (sum(vals) / len(vals)) if vals else float("nan")
        print(f"  {tag}: mean_EPU={stats[tag]:.4f}  (relevant frames={len(vals)})", flush=True)

    rules = ["all", "nearest2", "nearest1", "random1"]
    lines = ["", f"=== mean IU (entropy U_self, freshness latency, tau={args.tau}s), lower = better ===",
             "rule     " + "  ".join(f"k{k:<2d}" for k in range(11))]
    for r in rules:
        row = f"{r:8s} " + "  ".join(f"{stats.get(f'{r}_k{k}', float('nan'))*1000:5.1f}" for k in range(11))
        lines.append(row)
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(args.out, "iu_freshness_table.txt"), "w") as fp:
        fp.write(table + "\n")
    with open(os.path.join(args.out, "iu_freshness_table.csv"), "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["rule"] + [f"k{k}" for k in range(11)])
        for r in rules:
            w.writerow([r] + [round(stats.get(f"{r}_k{k}", float("nan")), 4) for k in range(11)])
    print(f"\nwrote {args.out}/iu_freshness_table.txt and .csv")


if __name__ == "__main__":
    main()
