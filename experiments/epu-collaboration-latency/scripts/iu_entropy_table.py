"""IU with entropy-based U_self (map-affordance route uncertainty).

U_self(i) = normalized entropy of vehicle i's plausible routes over horizon H,
enumerated from the CARLA (Town03) map topology -- junction branching gives high
entropy (turn/straight ambiguous), a single lane gives ~0 (predictable). This
replaces the earlier speed-only U_self so that two same-speed vehicles differ:
one at a junction (high sharing value) vs one on a straight lane (redundant).

Requires a running CARLA with Town03 (client-side waypoint queries only; no sim).
Reads the recorded per-frame geometry, recomputes IU, and prints the
mean-IU comparison table over rules x latencies.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import carla

# ---- shared factors (relevance, survival) reused from the geometry module ----
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ecpg_from_geometry import relevance, survival  # noqa: E402

H = 3.0
STEP = 2.0
L_ENUM = 30.0   # route-enumeration distance: long enough to resolve junction branches
STEP_CAP = 26   # bound the outward levels
PSI0 = 45.0     # deg; turn-angle scale in the branch prior
N_REF = 3.0     # normalizer (a 3-way junction -> ~1.0)
VISIT_CAP = 600
DEDUP = 5.0     # m; terminals within this are the same maneuver


def _dist_to_junction(wp0, max_d):
    """Distance ahead to the first junction waypoint (map lookahead)."""
    wp, d = wp0, 0.0
    while d < max_d:
        if wp.is_junction:
            return d
        nxt = wp.next(STEP)
        if not nxt:
            return None
        wp = nxt[0]
        d += STEP
    return None


def u_self_entropy(cmap, pos, vel):
    """
    Normalized route-entropy of vehicle i, gated by junction proximity.

    U_self = reach * H_route/log(N_ref)
      reach = min(1, speed*H / d_junction)  -- how much of the way to the next
              junction i covers within the decision horizon (a far junction is
              not an imminent uncertainty); 0 if no junction reachable.
      H_route = entropy over distinct maneuvers (turn/straight) at that junction,
              with a straight>turn behavioral prior.
    Straight lane, no junction ahead -> U_self=0 (i is predictable).
    """
    speed = math.hypot(vel[0], vel[1])
    wp0 = cmap.get_waypoint(carla.Location(x=pos[0], y=pos[1], z=0.0),
                            project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp0 is None:
        return 0.0
    reach_dist = speed * H
    d_junc = _dist_to_junction(wp0, L_ENUM)
    if d_junc is None:
        return 0.0                       # no junction within lookahead -> predictable
    reach = min(1.0, reach_dist / max(d_junc, 1.0))
    if reach <= 0.0:
        return 0.0
    # enumerate routes far enough to resolve the branch topology
    stack = [(wp0, 0.0)]
    terminals, visits = [], 0
    while stack and visits < VISIT_CAP:
        wp, d = stack.pop()
        visits += 1
        if d >= L_ENUM:
            terminals.append(wp)
            continue
        nxts = wp.next(STEP)
        if not nxts:
            terminals.append(wp)
            continue
        for c in nxts:
            stack.append((c, d + STEP))
    uniq = []
    for t in terminals:
        p = t.transform.location
        if not any(math.hypot(p.x - u[0], p.y - u[1]) < DEDUP for u in uniq):
            uniq.append((p.x, p.y, t.transform.rotation.yaw))
    if len(uniq) <= 1:
        return 0.0
    yaw0 = wp0.transform.rotation.yaw
    weights = [math.exp(-abs((y - yaw0 + 180) % 360 - 180) / PSI0) for (_, _, y) in uniq]
    s = sum(weights)
    probs = [w / s for w in weights]
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return reach * ent / math.log(N_REF)


def frame_epu_entropy(rec, cmap, cache):
    ego_wp = rec["ego_wp"]
    veh = rec["veh"]
    shared = rec["shared"]
    epu = 0.0
    any_relevant = False
    for vid, v in veh.items():
        w = relevance(ego_wp, v)
        if w <= 0:
            continue
        any_relevant = True
        # cache U_self by rounded position (map is static -> reuse)
        key = (round(v["pos"][0], 1), round(v["pos"][1], 1), round(v["v"][0], 1), round(v["v"][1], 1))
        if key in cache:
            u_self = cache[key]
        else:
            u_self = u_self_entropy(cmap, v["pos"], v["v"])
            cache[key] = u_self
        if vid in shared and shared[vid]:
            g = survival(v, shared[vid])
            u = u_self * (1.0 - g)
        else:
            u = u_self
        epu += w * u
    return epu, any_relevant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="logdir/ecpg_geom")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="experiments/coop-intention-latency")
    args = ap.parse_args()

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if "Town03" not in world.get_map().name:
        world = client.load_world("Town03")
    cmap = world.get_map()
    print(f"map: {cmap.name}")

    cache = {}
    # mean IU over relevant frames, per config
    stats = {}
    for f in sorted(glob.glob(os.path.join(args.geom, "*_k*.jsonl"))):
        tag = os.path.basename(f)[:-6]
        vals = []
        for line in open(f):
            rec = json.loads(line)
            epu, rel = frame_epu_entropy(rec, cmap, cache)
            if rel:
                vals.append(epu)
        stats[tag] = (sum(vals) / len(vals)) if vals else 0.0
        print(f"  {tag}: mean_EPU={stats[tag]:.3f}  (relevant frames={len(vals)})", flush=True)

    # table: rules x latencies
    rules = ["all", "nearest2", "nearest1", "random1"]
    lines = ["", "=== mean IU (entropy U_self), lower = better ===",
             "rule     " + "  ".join(f"k{k:<2d}" for k in range(11))]
    for r in rules:
        row = f"{r:8s} " + "  ".join(f"{stats.get(f'{r}_k{k}', float('nan')):.2f}" for k in range(11))
        lines.append(row)
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(args.out, "iu_entropy_table.txt"), "w") as fp:
        fp.write(table + "\n")

    import csv
    with open(os.path.join(args.out, "iu_entropy_table.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["rule"] + [f"k{k}" for k in range(11)])
        for r in rules:
            w.writerow([r] + [round(stats.get(f"{r}_k{k}", float("nan")), 3) for k in range(11)])
    print(f"\nwrote {args.out}/iu_entropy_table.txt and .csv")


if __name__ == "__main__":
    main()
