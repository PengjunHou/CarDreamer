"""Validate a task's ``env.scenario_actors`` spawn points against the map -- no CARLA server.

``carla.Map`` can be built straight from an OpenDRIVE ``.xodr`` file, so every geometry check the
simulator would apply (lane projection, junction membership, lane heading) is reproducible offline.
Run this after editing scenario coordinates instead of booting CARLA to eyeball them.

Reports, per scripted vehicle: the point actually spawned (after ``vehicle_start_mode`` resolution
and ``vehicle_spawn_z_offset``), how far the projection moved it, and any problem from
:func:`car_dreamer.toolkit.scenario_actors.spawn_problems` -- crucially including *inside a
junction*, which the snap-distance log cannot catch because the projection leaves such a point
essentially where it was.

Pedestrians are reported too (distance to the nearest driving lane / sidewalk), but never flagged:
a walker crossing a junction is usually deliberate.

Usage::

    python scripts/check_scenario_spawns.py --task carla_group_right_turn_auto
    python scripts/check_scenario_spawns.py --task carla_group_right_turn_auto --xodr /path/Town03.xodr

Exit code is 1 if any vehicle spawn has a problem, so it can gate CI.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import carla  # noqa: E402

import car_dreamer  # noqa: E402
from car_dreamer.toolkit.scenario_actors import (  # noqa: E402
    _to_location,
    _to_transform,
    parse_scenario_specs,
    resolve_vehicle_start,
    spawn_problems,
)

# Where a CARLA install keeps the OpenDRIVE files, newest-cache-first.
XODR_SEARCH = (
    "~/carlaCache/*/Carla/Maps/OpenDrive/{town}.xodr",
    "~/carla_simulator/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr",
)


def find_xodr(town: str) -> Path:
    for pattern in XODR_SEARCH:
        expanded = Path(pattern.format(town=town)).expanduser()
        matches = sorted(Path(expanded.anchor).glob(str(expanded.relative_to(expanded.anchor))))
        if matches:
            return matches[-1]
    raise SystemExit(
        f"Could not find {town}.xodr. Pass --xodr /path/to/{town}.xodr explicitly."
    )


def distance_xy(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--xodr", default=None, help="OpenDRIVE file; auto-detected from the task's town if omitted")
    args = parser.parse_args()

    config = car_dreamer.load_task_configs(args.task)
    town = str(config.env.world.town)
    xodr = Path(args.xodr).expanduser() if args.xodr else find_xodr(town)
    carla_map = carla.Map(town, xodr.read_text())
    print(f"task={args.task} town={town} xodr={xodr}\n")

    specs = parse_scenario_specs(getattr(config.env, "scenario_actors", None))
    scripted = [s for s in specs["vehicles"] if s["start"] is not None]
    background = [s for s in specs["vehicles"] if s["start"] is None]

    failures = 0
    print(f"{'#':<3}{'requested':<26}{'spawned':<26}{'snap':>6}{'mode':>7}  status")
    print("-" * 96)
    for i, spec in enumerate(scripted):
        resolved, mode, snap, _status = resolve_vehicle_start(
            carla_map, _to_transform(spec["start"]), spec, specs
        )
        problems = spawn_problems(carla_map, resolved)
        raw = spec["start"]
        req = f"({raw[0]:.2f},{raw[1]:.2f},{raw[2] if len(raw) > 2 else 0.0:.2f})"
        got = f"({resolved.location.x:.2f},{resolved.location.y:.2f},{resolved.location.z:.2f})"
        status = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
        failures += bool(problems)
        print(f"{i:<3}{req:<26}{got:<26}{snap:6.2f}{mode:>7}  {status}")
        if spec["stationary"]:
            print(f"{'':<3}  stationary parked observer (blocks its lane for the whole episode)")
        elif spec["autopilot_roam"]:
            print(f"{'':<3}  autopilot_roam: spawns here, then Traffic Manager drives it away")

    total_background = sum(int(s["count"]) for s in background)
    if total_background:
        print(f"\n{total_background} background vehicle(s): map-wide random spawn points (not position-checked)")

    peds = specs["pedestrians"]
    scripted_peds = [p for p in peds if p["start"] is not None]
    if scripted_peds:
        print(f"\nscripted pedestrians (mode={specs['pedestrian_start_mode']}, informational only):")
        for i, ped in enumerate(scripted_peds):
            loc = _to_location(ped["start"])
            drive = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            walk = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
            d_drive = distance_xy(loc, drive.transform.location) if drive else float("inf")
            d_walk = distance_xy(loc, walk.transform.location) if walk else float("inf")
            where = "on a driving lane" if drive and d_drive <= drive.lane_width / 2 else "off the roadway"
            junction = " (inside a junction)" if drive is not None and drive.is_junction else ""
            print(
                f"  {i}: ({loc.x:.2f},{loc.y:.2f})  d_driving={d_drive:.2f}m  "
                f"d_sidewalk={d_walk:.2f}m  {where}{junction}"
            )
    print(f"\n{len(scripted) - failures}/{len(scripted)} scripted vehicle spawns OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
