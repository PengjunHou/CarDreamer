"""Trace where and when the scenario's cooperative vehicles leave their lane (needs a running CARLA).

Diagnostic for "some vehicles end up in weird positions". It separates the two things that get
confused with each other:

* **spawn** -- where the vehicle is placed. Run with ``--warmup 0`` so ``reset()`` returns before
  Traffic Manager drives anything; the t=0 row is then the pure spawn pose.
* **driving** -- where TM takes it afterwards. The per-step trace shows the first step each vehicle
  drifts off its lane, and where.

Note ``env.reset()`` normally already ticks ``env.reset_warmup_ticks`` (20 for
carla_group_right_turn_auto) *inside* reset, i.e. ~2s of TM driving before you can observe anything.
``--warmup 0`` disables that so the spawn pose is observable.

Usage::

    cd <repo root>            # car_dreamer is not installed; cwd decides which copy runs
    python scripts/trace_scenario_vehicles.py --carla-port 2000 --steps 100 --warmup 0

Reports per vehicle: blueprint, spawn pose, and the first step it goes off-lane (lateral offset
beyond half a lane width) with the location, plus a periodic trace table.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _setup_carla_pythonapi() -> None:
    """Put the CARLA PythonAPI on sys.path (same as run_env.py / my_train.py).

    The task env imports ``agents.navigation.basic_agent``, which ships with the CARLA install
    rather than the ``carla`` pip package -- so this must run before create_task imports the env.
    """
    carla_root = os.environ.get("CARLA_ROOT", "/home/peh324/carla_simulator")
    for sub in ("PythonAPI", "PythonAPI/carla"):
        path = os.path.join(carla_root, sub)
        if path not in sys.path:
            sys.path.append(path)


_setup_carla_pythonapi()

import carla  # noqa: E402

import car_dreamer  # noqa: E402


def lane_report(carla_map, transform):
    """(off_lane, lateral_m, road_id, is_junction, misalign_deg) for a pose."""
    wp = carla_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return True, float("inf"), -1, False, 0.0
    lateral = math.hypot(
        transform.location.x - wp.transform.location.x,
        transform.location.y - wp.transform.location.y,
    )
    misalign = abs((transform.rotation.yaw - wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
    return lateral > wp.lane_width / 2.0, lateral, wp.road_id, wp.is_junction, misalign


def speed_of(actor):
    v = actor.get_velocity()
    return math.hypot(v.x, v.y)


def read_pose(actor):
    """(transform, speed) or (None, None) if CARLA has destroyed the actor.

    ``actor.is_alive`` is a client-side flag: it only flips when *this* client calls destroy().
    A vehicle despawned server-side (fell through the map, removed by the sim) still reports
    is_alive=True and only fails when you actually touch it -- so catch the RuntimeError.
    """
    try:
        return actor.get_transform(), speed_of(actor)
    except RuntimeError:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=0,
                        help="env.reset_warmup_ticks. 0 (default) makes the t=0 row the pure spawn pose.")
    args = parser.parse_args()

    env, _cfg = car_dreamer.create_task(
        args.task,
        [
            f"--env.world.carla_port={args.carla_port}",
            f"--env.reset_warmup_ticks={args.warmup}",
            "--env.display.enable=False",
        ],
    )
    sim = env.unwrapped

    # Ground truth for the spawn contract: what does reset_spawn actually ask CARLA for, and what
    # comes back? Patched before reset() so every scenario spawn is recorded. This is what separates
    # "the start transform was resolved wrong" from "the vehicle moved/was replaced after spawning".
    spawn_log = []
    orig_spawn = sim._world.spawn_scenario_vehicle

    def traced_spawn(start=None, destination=None, target_speed=None, ignore_lights=False,
                     stationary=False, blueprint=None):
        actor = orig_spawn(start=start, destination=destination, target_speed=target_speed,
                           ignore_lights=ignore_lights, stationary=stationary, blueprint=blueprint)
        # Deliberately NOT reading the actor's transform here: reset spawns run while the world is
        # asynchronous, and get_transform() before the server has ticked returns (0,0,0) -- a
        # meaningless "drift" that reads like a real finding. The t=0 block below is the honest
        # position measurement; this record only covers what reset_spawn *asked* for.
        spawn_log.append({
            "requested": None if start is None else (start.location.x, start.location.y, start.rotation.yaw),
            "stationary": bool(stationary),
            "target_speed": target_speed,
            "id": None if actor is None else int(actor.id),
            "type": None if actor is None else actor.type_id,
        })
        return actor

    sim._world.spawn_scenario_vehicle = traced_spawn
    env.reset()
    sim._world.spawn_scenario_vehicle = orig_spawn
    carla_map = sim._world._map

    print("\n=== spawn_scenario_vehicle 请求记录（reset_spawn 要的是什么，拿回什么 id）===")
    for i, r in enumerate(spawn_log):
        req = "随机(background)" if r["requested"] is None else \
            f"({r['requested'][0]:7.2f},{r['requested'][1]:8.2f}) yaw={r['requested'][2]:7.1f}"
        got = "spawn 失败(None)" if r["id"] is None else f"id={r['id']}"
        speed = "" if r["target_speed"] is None else f" target_speed={r['target_speed']:g}"
        print(f"  #{i:2} {got:>10} stationary={str(r['stationary']):5} 请求={req:34}"
              f"{speed}  {r['type'] or ''}")

    ids = sorted(int(v) for v in sim.coop_participant_ids)
    print(f"\n协同候选车 {len(ids)} 辆: {ids}   (warmup={args.warmup} ticks)\n")

    spawn = {}
    first_off = {}
    dead_at_zero = []
    for vid in ids:
        actor = sim._world.actor_dict.get(vid)
        if actor is None:
            print(f"  id={vid} 不在 actor_dict 里（spawn 返回 None?）")
            continue
        type_id = getattr(actor, "type_id", "?")
        t, _ = read_pose(actor)
        if t is None:
            # Already despawned before the episode even starts -- it can never cooperate.
            # v2v_comm_mixin._update_group_observations silently drops these, which is why the
            # problem stays invisible in a normal run.
            dead_at_zero.append(vid)
            print(f"  id={vid} {type_id:34} <<< reset 结束时已被 CARLA 销毁")
            continue
        off, lat, road, junc, mis = lane_report(carla_map, t)
        spawn[vid] = (t.location.x, t.location.y)
        print(
            f"  id={vid} {type_id:34} spawn=({t.location.x:7.2f},{t.location.y:8.2f},{t.location.z:5.2f}) "
            f"yaw={t.rotation.yaw:7.1f} road={road:5} junction={junc!s:5} lat={lat:5.2f} "
            f"{'<<< 生成即偏离车道' if off else ''}"
        )
    print(f"\n  t=0 存活 {len(ids) - len(dead_at_zero)}/{len(ids)}"
          + (f"，已销毁: {dead_at_zero}" if dead_at_zero else ""))

    header = " step | " + " | ".join(f"{vid:^28}" for vid in ids)
    print("\n" + header)
    print("-" * len(header))

    for step in range(1, args.steps + 1):
        env.step(env.action_space.sample())
        cells = []
        for vid in ids:
            actor = sim._world.actor_dict.get(vid)
            if actor is None:
                cells.append(f"{'<no actor>':^28}")
                continue
            t, spd = read_pose(actor)
            if t is None:
                if vid not in dead_at_zero:
                    dead_at_zero.append(vid)
                    print(f"  !! id={vid} 在第 {step} 步被 CARLA 销毁")
                cells.append(f"{'<destroyed>':^28}")
                continue
            off, lat, road, junc, mis = lane_report(carla_map, t)
            if off and vid not in first_off:
                first_off[vid] = (step, t.location.x, t.location.y, lat, road, spd)
            flag = "OFF" if off else "   "
            cells.append(f"({t.location.x:7.1f},{t.location.y:7.1f}) l={lat:4.1f} {flag}")
        if step % args.print_every == 0 or step == 1:
            print(f"{step:5} | " + " | ".join(cells))

    print("\n=== 结果 ===")
    for vid in ids:
        if vid in dead_at_zero:
            print(f"  id={vid} 被 CARLA 销毁 —— 它从未参与协同（mixin 会把它静默踢出协同组）")
        elif vid in first_off:
            s, x, y, lat, road, spd = first_off[vid]
            dx = math.hypot(x - spawn[vid][0], y - spawn[vid][1]) if vid in spawn else float("nan")
            print(
                f"  id={vid} 第 {s:4} 步 于 ({x:7.2f},{y:8.2f}) 偏离车道 {lat:5.2f}m  road={road:5} "
                f"speed={spd:4.1f}m/s  距生成点 {dx:6.1f}m"
            )
        else:
            print(f"  id={vid} 全程 {args.steps} 步都在车道上 ✓")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
