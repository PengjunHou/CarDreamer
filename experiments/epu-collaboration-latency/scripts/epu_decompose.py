"""Decompose entropy-EPU into shared-vehicle vs unshared-vehicle contributions.

Answers "why doesn't EPU rise with latency?": latency (gamma) only touches SHARED
vehicles, but in the richer scene most relevant vehicles are the never-shared scenario
background cars, whose u_i = U_self is latency-independent. This prints, per config:
    total EPU, shared-only EPU, unshared-only EPU, mean #relevant, mean #shared
so we can see the shared component rise with k while the total stays flat.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ecpg_from_geometry import relevance, survival  # noqa: E402
from epu_entropy_table import u_self_entropy  # noqa: E402

import carla  # noqa: E402


def frame_decompose(rec, cmap, cache):
    ego_wp = rec["ego_wp"]; veh = rec["veh"]; shared = rec["shared"]
    tot = sh = un = 0.0
    n_rel = n_sh = 0
    any_rel = False
    for vid, v in veh.items():
        w = relevance(ego_wp, v)
        if w <= 0:
            continue
        any_rel = True; n_rel += 1
        key = (round(v["pos"][0], 1), round(v["pos"][1], 1), round(v["v"][0], 1), round(v["v"][1], 1))
        if key in cache:
            u_self = cache[key]
        else:
            u_self = u_self_entropy(cmap, v["pos"], v["v"]); cache[key] = u_self
        if vid in shared and shared[vid]:
            g = survival(v, shared[vid])
            u = u_self * (1.0 - g); sh += w * u; n_sh += 1
        else:
            u = u_self; un += w * u
        tot += w * u
    return tot, sh, un, n_rel, n_sh, any_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="logdir/group_epu_geom")
    ap.add_argument("--port", type=int, default=2010)
    ap.add_argument("--rules", default="all,nearest1")
    args = ap.parse_args()

    client = carla.Client("127.0.0.1", args.port); client.set_timeout(60.0)
    world = client.get_world()
    if "Town03" not in world.get_map().name:
        world = client.load_world("Town03")
    cmap = world.get_map(); cache = {}

    rules = args.rules.split(",")
    print(f"{'config':14s} {'totalEPU':>9s} {'sharedEPU':>10s} {'unsharedEPU':>12s} {'meanRel':>8s} {'meanShared':>11s}")
    for rule in rules:
        for k in range(11):
            f = os.path.join(args.geom, f"{rule}_k{k}.jsonl")
            if not os.path.isfile(f):
                continue
            tots = []; shs = []; uns = []; nrels = []; nshs = []
            for line in open(f):
                rec = json.loads(line)
                tot, sh, un, n_rel, n_sh, any_rel = frame_decompose(rec, cmap, cache)
                if any_rel:
                    tots.append(tot); shs.append(sh); uns.append(un); nrels.append(n_rel); nshs.append(n_sh)
            n = max(len(tots), 1)
            print(f"{rule+'_k'+str(k):14s} {sum(tots)/n*1000:9.1f} {sum(shs)/n*1000:10.2f} "
                  f"{sum(uns)/n*1000:12.1f} {sum(nrels)/n:8.2f} {sum(nshs)/n:11.3f}", flush=True)
        print()


if __name__ == "__main__":
    main()
