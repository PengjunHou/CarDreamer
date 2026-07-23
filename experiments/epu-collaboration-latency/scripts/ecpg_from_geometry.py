"""Per-frame ECPG from recorded geometry (Path B).

Reads the per-frame geometry jsonl dumped by CarlaWptFixedEnv
(CARDREAMER_RECORD_GEOMETRY) and computes, for each frame t, the Effective
Cooperative Perception Gain contributed by the shared (delayed) intentions:

    ECPG(t) = coverage-aggregate over shared vehicles i of
                  w_i(t)  x  U_i(t)  x  gamma_i(t)

  w_i     relevance: spatio-temporal conflict between vehicle i and the ego's
          planned path. Predict i by constant velocity over horizon H; take the
          minimum ego-path-to-i separation over H; w = max(0, 1 - d_min/d_safe).
  U_i     novelty: how much sharing reduces ego's own predictive uncertainty
          about i. Proxy = kinematic reachable-set spread over H
          (~ speed * H), normalized; small for a slow/steady car ego can
          already predict, large for a fast car with many futures.
  gamma_i latency survival, SELF-CALIBRATED and model-free: the ego holds the
          delayed plan W_i (broadcast at t-k) and observes i's CURRENT position
          x_i(t). Residual r_i = || x_i(t) - W_i_head ||, where W_i_head is
          where the stale plan placed i "now". gamma = 1 - r_i/r_tol, clipped
          to [gamma_min, 1]; gamma<0 => the stale plan actively misleads.

Redundancy: shared vehicles are aggregated over the conflict region so
overlapping collaborators do not double-count and a fresh one can cover a stale
one (submodular union of the positive parts; negative parts diluted by count).

All quantities come from geometry + kinematics + the shared waypoints. No model.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

# ---- constants (scenario-reasoned; documented, not fit to outcomes) ----
DT = 0.1          # s per step (10 Hz)
HORIZON = 6.0     # s conflict-prediction horizon: long enough to connect a
                  # vehicle still approaching a junction to the conflict it will
                  # create seconds later (its intention-uncertainty NOW is
                  # relevant to a conflict SOON, not at the same instant)
D_SAFE = 8.0      # m: conflict separation below which a vehicle is fully relevant
R_TOL = 4.0       # m: stale-plan residual tolerance (~ merge-gap scale)
GAMMA_MIN = -1.0  # max misleading harm per collaborator
V_REF = 6.0       # m/s: speed normalizing the reachable-set spread


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def relevance(ego_wp, veh):
    """w_i: min space-time separation between ego's path and i's CV prediction."""
    if not ego_wp:
        return 0.0
    p = veh["pos"]
    v = veh["v"]
    steps = int(HORIZON / DT)
    d_min = float("inf")
    # ego marches along its waypoint list at ~one wp per few steps; approximate
    # by scanning all ego waypoints (they already densely sample the path) vs i's
    # constant-velocity future positions, time-aligned by index along the path.
    for s in range(0, steps, 3):
        ix, iy = p[0] + v[0] * s * DT, p[1] + v[1] * s * DT
        # nearest ego waypoint reachable by ego within a comparable time window
        lo = max(0, s - 6)
        hi = min(len(ego_wp), s + 6)
        for w in ego_wp[lo:hi] or ego_wp[:6]:
            d = _dist((ix, iy), w)
            if d < d_min:
                d_min = d
    if d_min == float("inf"):
        return 0.0
    return max(0.0, 1.0 - d_min / D_SAFE)


def novelty(veh):
    """U_i: normalized reachable-set spread proxy (~ speed over horizon)."""
    speed = math.hypot(veh["v"][0], veh["v"][1])
    return min(1.0, speed / V_REF)


def survival(veh, shared_path):
    """gamma_i: self-calibrated from residual between stale plan and observed pos."""
    if not shared_path:
        return 0.0
    # the stale plan's head is where the (t-k) broadcast anchored i; compare to
    # i's current observed position -- how far the plan has already been violated.
    head = shared_path[0]
    r = _dist(veh["pos"], head)
    g = 1.0 - r / R_TOL
    return max(GAMMA_MIN, min(1.0, g))


def frame_ecpg(rec):
    """ECPG contributed by all shared vehicles this frame."""
    ego_wp = rec["ego_wp"]
    shared = rec["shared"]
    veh = rec["veh"]
    pos, neg, n_pos = [], 0.0, 0
    for sid, path in shared.items():
        if sid not in veh:
            continue
        w = relevance(ego_wp, veh[sid])
        u = novelty(veh[sid])
        g = survival(veh[sid], path)
        contrib = w * u * g
        if contrib >= 0:
            pos.append(contrib)
            n_pos += 1
        else:
            neg += contrib
    # submodular union of the positive parts (redundancy: fresh covers stale),
    # plus diluted negative part (more collaborators dampen a misleading one)
    pos.sort(reverse=True)
    union = 0.0
    for i, c in enumerate(pos):
        union += c * (0.6 ** i)   # diminishing returns for overlapping collaborators
    n_shared = max(1, len(shared))
    return union + neg / n_shared


def frame_epu(rec):
    """
    Ego Perception Uncertainty this frame: the relevance-weighted RESIDUAL
    predictive uncertainty about surrounding vehicles' futures, given what ego
    currently knows (own kinematic prediction + any shared, possibly stale plan).

        EPU(t) = sum over relevant vehicles i of  w_i * u_i
        u_i = U_i_self                        if i's intention is not shared
              U_i_self * (1 - gamma_i)        if shared (stale by gamma_i)

    gamma=1 (fresh)   -> u_i=0        (uncertainty collapses)
    gamma=0 (useless) -> u_i=U_i_self (as if not shared)
    gamma<0 (mislead) -> u_i>U_i_self (stale plan corrupts belief: worse than none)

    Lower EPU = better understanding. Summed over ALL relevant vehicles (not
    just shared ones), so unshared/irrelevant collaborators leave uncertainty high.
    """
    ego_wp = rec["ego_wp"]
    veh = rec["veh"]
    shared = rec["shared"]
    epu = 0.0
    for vid, v in veh.items():
        w = relevance(ego_wp, v)
        if w <= 0:
            continue
        u_self = novelty(v)  # ego's own prior uncertainty about this vehicle
        if vid in shared and shared[vid]:
            g = survival(v, shared[vid])
            u = u_self * (1.0 - g)   # 1-g in [0,2]: >u_self when g<0 (misleading)
        else:
            u = u_self
        epu += w * u
    return epu


def load(path):
    """Group frames by (rule, latency, episode)."""
    eps = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        eps[(r["rule"], r["latency"], r["ep"])].append(r)
    for k in eps:
        eps[k].sort(key=lambda r: r["t"])
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="logdir/ecpg_geom/*.jsonl")
    ap.add_argument("--out", default="experiments/coop-intention-latency")
    args = ap.parse_args()

    series = defaultdict(list)  # (rule,latency) -> list of per-episode ECPG(t) lists
    for path in glob.glob(args.glob):
        for (rule, lat, ep), frames in load(path).items():
            curve = [frame_ecpg(f) for f in frames]
            series[(rule, lat)].append(curve)
    print("loaded (rule,latency): episodes")
    for key in sorted(series):
        print(f"  {key}: {len(series[key])} episodes, "
              f"mean len {sum(len(c) for c in series[key])/len(series[key]):.0f}")

    # save raw per-episode curves for downstream plotting
    dump = {f"{r}_k{k}": v for (r, k), v in series.items()}
    with open(os.path.join(args.out, "ecpg_timeseries.json"), "w") as f:
        json.dump(dump, f)
    print(f"wrote {os.path.join(args.out, 'ecpg_timeseries.json')}")


if __name__ == "__main__":
    main()
