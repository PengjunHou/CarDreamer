#!/usr/bin/env python3
"""Visualize per-vehicle visibility-aware WAM BEV frames from a live CARLA env.

Each vehicle gets its own vehicle-centric BEV image, similar to ``data/birdeye_frames``. Objects are
filtered per vehicle before rendering: ego uses ``visible_to_ego`` and a candidate ``vid`` uses
``vid in visible_to_collaborators``. Objects invisible to that vehicle are not drawn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_carla_pythonapi() -> None:
    carla_root = os.environ.get("CARLA_ROOT", "/home/peh324/carla_simulator")
    for sub in ("PythonAPI", "PythonAPI/carla"):
        path = os.path.join(carla_root, sub)
        if path not in sys.path:
            sys.path.append(path)


def build_env(task: str, env_args: List[str]):
    import gymnasium as gym

    import car_dreamer
    from car_dreamer import toolkit

    config = car_dreamer.load_task_configs(task)
    config, _ = toolkit.Flags(config).parse_known(env_args)
    env = gym.make(config.env.name, config=config.env)
    return env, config


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Visualize per-vehicle WAM BEV frames.")
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/wam_per_vehicle_bev"))
    parser.add_argument("--jsonl-name", default="per_vehicle_bev.jsonl")
    parser.add_argument("--no-jsonl", action="store_true", default=False)
    parser.add_argument("--bev-range-m", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--ego-offset-m", type=float, default=12.0)
    parser.add_argument("--pixels-per-meter", type=float, default=8.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    known, passthrough = parser.parse_known_args()
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def _cfg_get(node, key: str, default):
    if node is None:
        return default
    try:
        if key in node:
            value = node[key]
            return value if value is not None else default
    except TypeError:
        pass
    return getattr(node, key, default)


def _carla_world(sim):
    return getattr(getattr(sim, "_world", None), "_world", None)


def _candidate_ids(sim) -> List[int]:
    participants = getattr(sim, "coop_participant_ids", None)
    if participants:
        return sorted(int(i) for i in participants)
    return sorted(int(getattr(v, "id", -1)) for v in getattr(sim, "group_vehs", []) if v is not None)


def _actor_for_vehicle(sim, vehicle_id: int):
    getter = getattr(sim, "_get_group_member_actor", None)
    if getter is not None:
        actor = getter(int(vehicle_id))
        if actor is not None:
            return actor
    world = _carla_world(sim)
    if world is None:
        return None
    return world.get_actor(int(vehicle_id))


def _route_xy(sim):
    route = getattr(sim, "_wam_route_xy", None)
    return tuple(route()) if route is not None else ()


def _actor_record(actor) -> Dict[str, object]:
    tf = actor.get_transform()
    bb = actor.bounding_box
    yaw = float(tf.rotation.yaw)
    x, y, z = float(tf.location.x), float(tf.location.y), float(tf.location.z)
    import math

    cos_y, sin_y = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    hl, hw = float(bb.extent.x), float(bb.extent.y)
    bbox = []
    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        bbox.append((x + lx * cos_y - ly * sin_y, y + lx * sin_y + ly * cos_y))
    return {
        "actor_id": int(actor.id),
        "position": (x, y, z),
        "yaw": yaw,
        "bbox": tuple(bbox),
    }


def _object_record(obj) -> Dict[str, object]:
    return {
        "actor_id": int(obj.actor_id),
        "position": (float(obj.x), float(obj.y), float(obj.z)),
        "yaw": float(obj.yaw),
        "bbox": tuple(obj.bbox or ()),
    }


def _build_vehicle_views(sim) -> Tuple[List[Dict[str, object]], List[int]]:

    ego = getattr(sim, "ego", None)
    if ego is None:
        return [], []
    objects = list(getattr(sim, "_wam_object_states", []))
    policy = getattr(sim, "_wam_policy", None)
    selected = set(int(i) for i in getattr(policy, "selected_vehicle_ids", ()))

    vehicles = [("ego", int(ego.id), ego)]
    skipped: List[int] = []
    for vid in _candidate_ids(sim):
        actor = _actor_for_vehicle(sim, vid)
        if actor is None:
            skipped.append(int(vid))
            continue
        vehicles.append(("candidate", int(vid), actor))

    views: List[Dict[str, object]] = []
    for role, vid, actor in vehicles:
        if role == "ego":
            visible = [s for s in objects if bool(getattr(s, "visible_to_ego", False))]
            route_xy = _route_xy(sim)
        else:
            visible = [s for s in objects if int(vid) in getattr(s, "visible_to_collaborators", ())]
            route_xy = ()
        visible_ids = [int(s.actor_id) for s in visible]
        selected_by_policy = int(vid) in selected
        views.append(
            {
                "role": role,
                "vehicle_id": int(vid),
                "vehicle": _actor_record(actor),
                "visible_objects": [_object_record(s) for s in visible],
                "visible_object_ids": visible_ids,
                "route_xy": route_xy,
                "selected_by_policy": selected_by_policy,
            }
        )
    return views, skipped


def _view_json(view: Mapping[str, object]) -> Dict[str, object]:
    vehicle = view["vehicle"]
    position = vehicle["position"]
    return {
        "vehicle_id": int(view["vehicle_id"]),
        "role": str(view["role"]),
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "yaw": float(vehicle["yaw"]),
        "visible_object_ids": [int(i) for i in view["visible_object_ids"]],
        "selected_by_policy": bool(view["selected_by_policy"]),
    }


def main() -> int:
    known, passthrough = parse_args()
    if known.steps <= 0:
        raise ValueError("--steps must be positive")
    if known.render_every <= 0 or known.print_every <= 0:
        raise ValueError("--render-every and --print-every must be positive")
    if known.bev_range_m is not None and known.bev_range_m <= 0:
        raise ValueError("--bev-range-m must be positive")
    if known.image_size <= 0 or known.ego_offset_m < 0 or known.pixels_per_meter <= 0:
        raise ValueError("--image-size, --ego-offset-m, and --pixels-per-meter must be valid")

    _setup_carla_pythonapi()

    from car_dreamer.toolkit.observer.handlers.renderer.map_renderer import MapRenderer
    from car_dreamer.toolkit.wam.visualization import render_vehicle_centric_wam_bev

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        "--env.wam.build_graph=True",
        *passthrough,
    ]
    env, config = build_env(known.task, env_args)
    wam_cfg = _cfg_get(config.env, "wam", None)
    graph_cfg = _cfg_get(wam_cfg, "graph", None)
    bev_range_m = float(known.bev_range_m or _cfg_get(graph_cfg, "bev_range_m", 50.0))
    sim = env.unwrapped
    known.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = known.out_dir / known.jsonl_name

    jsonl_file = None
    if not known.no_jsonl and known.jsonl_name:
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

    try:
        env.reset(seed=known.seed)
        world = _carla_world(sim)
        if world is None:
            raise RuntimeError("env does not expose a CARLA world")
        renderer = MapRenderer(world, world.get_map(), pixels_per_meter=float(known.pixels_per_meter))
        print(
            f"Rendering per-vehicle WAM BEV for {known.steps} steps to {known.out_dir} "
            f"(image={known.image_size}, range={bev_range_m:.1f}m, every={known.render_every})",
            flush=True,
        )

        for _ in range(int(known.steps)):
            step = int(getattr(sim, "_time_step", 0))
            update = getattr(sim, "_update_wam_runtime_state", None)
            if update is not None:
                update(force=True)

            if step % int(known.render_every) == 0:
                views, skipped = _build_vehicle_views(sim)
                for view in views:
                    vehicle_id = int(view["vehicle_id"])
                    render_vehicle_centric_wam_bev(
                        map_renderer=renderer,
                        vehicle=view["vehicle"],
                        visible_objects=view["visible_objects"],
                        route_xy=view["route_xy"],
                        output_path=known.out_dir / f"vehicle_{vehicle_id}" / f"bev_{step:06d}.png",
                        image_size_px=int(known.image_size),
                        bev_range_m=float(bev_range_m),
                        ego_offset_m=float(known.ego_offset_m),
                    )
                if jsonl_file is not None:
                    jsonl_file.write(
                        json.dumps(
                            {
                                "step": step,
                                "vehicles": [_view_json(v) for v in views],
                                "skipped_vehicle_ids": skipped,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    jsonl_file.flush()

            if step % int(known.print_every) == 0:
                print(f"step={step} out_dir={known.out_dir}", flush=True)

            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset(seed=known.seed)

        print(f"Done. BEV frames: {known.out_dir}", flush=True)
        if jsonl_file is not None:
            print(f"JSONL: {jsonl_path}", flush=True)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
