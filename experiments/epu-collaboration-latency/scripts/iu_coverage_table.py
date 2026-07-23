"""Combine Intention Uncertainty (IU) with the ported coverage uncertainty (U^cov).

For each recorded frame we compute, from geometry only:
  - IU        : intention uncertainty (iu_freshness_table.frame_epu_fresh; needs CARLA map for U_self)
  - U^cov     : coverage uncertainty (ported coverage.py; pure geometry, NO CARLA)
  - combined  : alpha * IU_lin + (1-alpha) * U^cov,  IU_lin = clip(IU / IU_REF, 0, 1)

Coverage inputs are derived from the recorded geometry (ego pose + route = ego_wp; collaborators =
the SAME shared.keys() as intention sharing, pose from veh pos+velocity; freshness = exp(-gamma*k*dt);
past route = the episode's ego_pos history; occlusion off, actor_polygons=None). Because the recorded
runs are full-observability, coverage assumes a nominal ego FOV (default 150 deg / 32 m) -> it is the
counterfactual "route coverage the ego WOULD have if FOV-limited". Three tables (rules x latency) are
emitted: IU (x1e-3), U^cov ([0,1]), combined.
"""
import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage import BevSpec, CoverageConfig, build_coverage_raster  # noqa: E402
from ecpg_from_geometry import load  # noqa: E402

DT = 0.1
RULES = ["all", "nearest2", "nearest1", "random1"]


def _downsample(points, step_m):
    """Thin a polyline to ~step_m spacing (keeps first + last). Cuts the O(N*S) corridor cost:
    the dense planner waypoints have hundreds of points; 2 m spacing preserves the 8 m-wide corridor."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= step_m:
            out.append(p)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def _route_yaw_deg(ego, ego_wp, ego_hist):
    """Ego heading (deg): direction to the first route waypoint >2 m ahead; fall back to motion delta."""
    for w in ego_wp:
        dx, dy = w[0] - ego[0], w[1] - ego[1]
        if math.hypot(dx, dy) > 2.0:
            return math.degrees(math.atan2(dy, dx))
    if len(ego_hist) >= 1:
        dx, dy = ego[0] - ego_hist[-1][0], ego[1] - ego_hist[-1][1]
        if math.hypot(dx, dy) > 1e-3:
            return math.degrees(math.atan2(dy, dx))
    return 0.0


def frame_coverage(rec, ego_hist, cov_cfg, spec, fov, srange, gamma):
    """Coverage uncertainty U^cov for one recorded frame (pure geometry)."""
    ego = rec["ego_pos"]
    ego_wp = rec["ego_wp"]
    veh = rec["veh"]
    shared = rec["shared"]
    k = int(rec["latency"])
    ego_yaw = _route_yaw_deg(ego, ego_wp, ego_hist)
    ego_pose = (float(ego[0]), float(ego[1]), ego_yaw)
    route_xy = _downsample(ego_wp, 2.0)                    # dense planner wpts -> ~2 m spacing
    past_route_xy = _downsample(ego_hist[-60:], 2.0)       # recent past only (>10 m at any speed)
    ego_observer = (0, float(ego[0]), float(ego[1]), ego_yaw)

    fresh = math.exp(-gamma * k * DT)  # same latency for every collaborator in a config
    collaborator_observers = []
    freshness = []
    for cid in shared:
        if cid in veh:
            p = veh[cid]["pos"]
            v = veh[cid]["v"]
            yaw = math.degrees(math.atan2(v[1], v[0])) if math.hypot(v[0], v[1]) > 0.1 else ego_yaw
            collaborator_observers.append((int(cid), float(p[0]), float(p[1]), yaw))
            freshness.append(fresh)

    _, metrics = build_coverage_raster(
        ego_pose=ego_pose, route_xy=route_xy, past_route_xy=past_route_xy,
        ego_observer=ego_observer, collaborator_observers=tuple(collaborator_observers),
        actor_polygons=None,
        ego_fov=fov, ego_sight_range=srange,
        collaborator_fov=fov, collaborator_sight_range=srange,
        config=cov_cfg, spec=spec, collaborator_freshness=tuple(freshness),
    )
    return float(metrics["coverage_uncertainty"])


def _write_table(stats, rules, title, out_path, scale):
    lines = ["", title, "rule     " + "  ".join(f"k{k:<2d}" for k in range(11))]
    for r in rules:
        row = f"{r:8s} " + "  ".join(f"{stats.get(f'{r}_k{k}', float('nan')) * scale:5.1f}" for k in range(11))
        lines.append(row)
    table = "\n".join(lines)
    print(table)
    with open(out_path + ".txt", "w") as fp:
        fp.write(table + "\n")
    with open(out_path + ".csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["rule"] + [f"k{k}" for k in range(11)])
        for r in rules:
            w.writerow([r] + [round(stats.get(f"{r}_k{k}", float("nan")), 5) for k in range(11)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="logdir/ecpg_geom")
    ap.add_argument("--port", type=int, default=2010)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="combined = alpha*IU_lin + (1-alpha)*Ucov")
    ap.add_argument("--iu-ref", type=float, default=0.04, help="IU_lin = clip(IU/iu_ref, 0, 1)")
    ap.add_argument("--fov", type=float, default=150.0)
    ap.add_argument("--sight-range", type=float, default=32.0)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--bev-size", type=int, default=32)
    ap.add_argument("--bev-range", type=float, default=40.0)
    ap.add_argument("--coverage-only", action="store_true", help="skip per-frame IU (no CARLA needed)")
    ap.add_argument("--iu-csv", default=None,
                    help="existing IU table csv (rule,k0..k10 raw); combine at config level, CARLA-free")
    ap.add_argument("--out", default="experiments/epu-collaboration-latency/results/combined")
    args = ap.parse_args()

    cmap = cache = None
    if not args.coverage_only:
        import carla
        from iu_freshness_table import frame_epu_fresh
        client = carla.Client("127.0.0.1", args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        if "Town03" not in world.get_map().name:
            world = client.load_world("Town03")
        cmap = world.get_map()
        cache = {}
        print(f"map: {cmap.name}")

    spec = BevSpec(size=args.bev_size, range_m=args.bev_range)
    cov_cfg = CoverageConfig()
    os.makedirs(args.out, exist_ok=True)

    iu_stats, cov_stats, comb_stats = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(args.geom, "*_k*.jsonl"))):
        tag = os.path.basename(f)[:-6]
        iu_vals, cov_vals, comb_vals = [], [], []
        for _key, frames in load(f).items():
            ego_hist = []
            for rec in frames:
                ucov = frame_coverage(rec, ego_hist, cov_cfg, spec, args.fov, args.sight_range, args.gamma)
                if args.coverage_only:
                    iu, rel = 0.0, True
                else:
                    iu, rel = frame_epu_fresh(rec, cmap, cache, args.tau)
                if rel:
                    iu_lin = min(max(iu / args.iu_ref, 0.0), 1.0)
                    iu_vals.append(iu)
                    cov_vals.append(ucov)
                    comb_vals.append(args.alpha * iu_lin + (1.0 - args.alpha) * ucov)
                ego_hist.append((float(rec["ego_pos"][0]), float(rec["ego_pos"][1])))
        n = max(len(cov_vals), 1)
        iu_stats[tag] = sum(iu_vals) / n
        cov_stats[tag] = sum(cov_vals) / n
        comb_stats[tag] = sum(comb_vals) / n
        print(f"  {tag}: IU={iu_stats[tag]*1000:.1f}e-3  Ucov={cov_stats[tag]:.3f}  "
              f"combined={comb_stats[tag]:.3f}  (frames={len(cov_vals)})", flush=True)

    # Config-level combine from an existing IU csv (CARLA-free path). Overrides the per-frame IU.
    if args.iu_csv:
        iu_stats = {}
        with open(args.iu_csv) as fp:
            for row in csv.DictReader(fp):
                for k in range(11):
                    iu_stats[f"{row['rule']}_k{k}"] = float(row[f"k{k}"])
        comb_stats = {}
        for tag, cov in cov_stats.items():
            iu = iu_stats.get(tag)
            if iu is None:
                continue
            iu_lin = min(max(iu / args.iu_ref, 0.0), 1.0)
            comb_stats[tag] = args.alpha * iu_lin + (1.0 - args.alpha) * cov

    _write_table(cov_stats, RULES, "=== coverage uncertainty U^cov ([0,1], lower=better) ===",
                 os.path.join(args.out, "coverage_table"), scale=1000.0)
    have_iu = (not args.coverage_only) or bool(args.iu_csv)
    if have_iu:
        _write_table(iu_stats, RULES, "=== IU (x1e-3, lower=better) ===",
                     os.path.join(args.out, "iu_table"), scale=1000.0)
        _write_table(comb_stats, RULES,
                     f"=== combined = {args.alpha}*IU_lin + {1-args.alpha}*Ucov  (IU_ref={args.iu_ref}, x1e-3) ===",
                     os.path.join(args.out, "combined_table"), scale=1000.0)
    print(f"\nwrote tables to {args.out}/")


if __name__ == "__main__":
    main()
