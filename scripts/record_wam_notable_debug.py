#!/usr/bin/env python3
"""Record WAM notable-object debug JSONL and sampled BEV frames.

This script runs a CarDreamer task directly, records the current WAM notable
objects for the first --steps timesteps, and keeps stepping for the requested
future horizon so each JSONL row can include ground-truth future waypoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


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
    parser = argparse.ArgumentParser(
        description="Record WAM notable-object JSONL and sampled BEV debug frames.",
    )
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--future-horizon-s", type=float, default=3.0)
    parser.add_argument("--future-waypoints", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/wam_notable_debug"))
    parser.add_argument("--jsonl-name", default="notable_motion.jsonl")
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--bev-range-m", type=float, default=64.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--ego-offset-m", type=float, default=12.0)
    parser.add_argument("--pixels-per-meter", type=float, default=8.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    known, passthrough = parser.parse_known_args()
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def _fixed_dt(sim, config) -> float:
    settings = getattr(getattr(sim, "_world", None), "_settings", None)
    if settings is not None and getattr(settings, "fixed_delta_seconds", None):
        return float(settings.fixed_delta_seconds)
    return float(getattr(getattr(config.env, "world", None), "fixed_delta_seconds", 0.1))


def _carla_world(sim):
    return getattr(getattr(sim, "_world", None), "_world", None)


def _actor_snapshots(sim) -> List:
    from car_dreamer.toolkit.wam import snapshot_from_carla_actor

    world = _carla_world(sim)
    if world is None:
        return []
    actors = world.get_actors()
    snapshots = []
    for actor in list(actors.filter("vehicle.*")) + list(actors.filter("walker.pedestrian.*")):
        try:
            snapshots.append(snapshot_from_carla_actor(actor))
        except RuntimeError:
            pass
    return snapshots


def _ego_snapshot(sim):
    from car_dreamer.toolkit.wam import snapshot_from_carla_actor

    ego = getattr(sim, "ego", None)
    if ego is None:
        raise RuntimeError("env does not expose an ego actor")
    return snapshot_from_carla_actor(ego)


def _wam_state(sim) -> Tuple[Iterable, Mapping[int, object], Dict[str, object]]:
    update = getattr(sim, "_update_wam_runtime_state", None)
    if update is not None:
        update(force=True)

    notable = list(getattr(sim, "_wam_notable_records", []))
    predictions = dict(getattr(sim, "_wam_motion_predictions", {}))
    policy = getattr(sim, "_wam_policy", None)
    coop_request = getattr(sim, "_wam_coop_request", None)
    max_uncertainty = 0.0
    if predictions:
        max_uncertainty = max(float(pred.uncertainty_score) for pred in predictions.values())
    wam = {
        "coop_triggered": bool(coop_request is not None),
        "uncertainty_max": float(max_uncertainty),
        "policy_selected_vehicle_ids": list(policy.selected_vehicle_ids) if policy is not None else [],
        "policy_modality_by_vehicle": dict(policy.modality_by_vehicle) if policy is not None else {},
        "notable_object_ids": [int(record.object_state.actor_id) for record in notable],
        "visible_notable_object_ids": [
            int(record.object_state.actor_id) for record in notable if bool(record.visible)
        ],
        "invisible_notable_object_ids": [
            int(record.object_state.actor_id) for record in notable if bool(record.invisible)
        ],
    }
    return notable, predictions, wam


def _write_records(
    *,
    records: Iterable[Mapping[str, object]],
    jsonl_file,
    renderer,
    out_dir: Path,
    render_every: int,
    render_enabled: bool,
    bev_range_m: float,
    image_size_px: int,
    ego_offset_m: float,
) -> int:
    render_wam_bev_record = None
    if render_enabled and renderer is not None:
        from car_dreamer.toolkit.wam.visualization import render_wam_bev_record

    count = 0
    frames_dir = out_dir / "frames"
    for record in records:
        jsonl_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        count += 1
        step = int(record["step"])
        if render_wam_bev_record is not None and step % render_every == 0:
            render_wam_bev_record(
                map_renderer=renderer,
                record=record,
                output_path=frames_dir / f"step_{step:06d}.png",
                bev_range_m=bev_range_m,
                image_size_px=image_size_px,
                ego_offset_m=ego_offset_m,
            )
    jsonl_file.flush()
    return count


def main() -> int:
    known, passthrough = parse_args()
    if known.steps <= 0:
        raise ValueError("--steps must be positive")
    if known.future_horizon_s <= 0:
        raise ValueError("--future-horizon-s must be positive")
    if known.future_waypoints <= 0:
        raise ValueError("--future-waypoints must be positive")
    if known.render_every <= 0:
        raise ValueError("--render-every must be positive")
    if known.bev_range_m <= 0 or known.image_size <= 0 or known.ego_offset_m < 0 or known.pixels_per_meter <= 0:
        raise ValueError("--bev-range-m, --image-size, --ego-offset-m, and --pixels-per-meter must be valid")
    if known.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if known.sleep < 0:
        raise ValueError("--sleep must be non-negative")

    _setup_carla_pythonapi()

    from car_dreamer.toolkit.observer.handlers.renderer.map_renderer import MapRenderer
    from car_dreamer.toolkit.wam import WAMNotableDebugRecorder

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        *passthrough,
    ]
    env, config = build_env(known.task, env_args)
    sim = env.unwrapped
    known.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = known.out_dir / known.jsonl_name

    written = 0
    try:
        env.reset(seed=known.seed)
        fixed_dt = _fixed_dt(sim, config)
        recorder = WAMNotableDebugRecorder(
            fixed_dt=fixed_dt,
            horizon_s=known.future_horizon_s,
            future_samples=known.future_waypoints,
        )

        renderer = None
        if not known.no_render:
            world = _carla_world(sim)
            if world is not None:
                renderer = MapRenderer(world, world.get_map(), pixels_per_meter=float(known.pixels_per_meter))

        if known.display:
            print(f"Web visualization: http://localhost:{known.carla_port + 7000}/", flush=True)
        print(
            f"Recording {known.steps} steps to {jsonl_path} "
            f"(future={known.future_horizon_s:.1f}s/{known.future_waypoints} waypoints, "
            f"extra_steps={recorder.horizon_steps})",
            flush=True,
        )

        last_record_step = int(known.steps) - 1
        last_observe_step = last_record_step + int(recorder.horizon_steps)
        terminated = False
        truncated = False

        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            while int(getattr(sim, "_time_step", 0)) <= last_observe_step:
                current_step = int(getattr(sim, "_time_step", 0))
                notable, predictions, wam = _wam_state(sim)
                ready = recorder.observe(
                    step=current_step,
                    time_s=float(current_step) * float(fixed_dt),
                    ego=_ego_snapshot(sim),
                    actors=_actor_snapshots(sim),
                    notable_records=notable,
                    predictions=predictions,
                    wam=wam,
                    include_record=current_step <= last_record_step,
                )
                written += _write_records(
                    records=ready,
                    jsonl_file=jsonl_file,
                    renderer=renderer,
                    out_dir=known.out_dir,
                    render_every=int(known.render_every),
                    render_enabled=not known.no_render,
                    bev_range_m=float(known.bev_range_m),
                    image_size_px=int(known.image_size),
                    ego_offset_m=float(known.ego_offset_m),
                )

                if current_step % known.print_every == 0:
                    print(
                        f"step={current_step} written={written} "
                        f"notable={wam['notable_object_ids']} triggered={wam['coop_triggered']}",
                        flush=True,
                    )

                if current_step >= last_observe_step:
                    break
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    print(
                        f"episode ended early at step={current_step}: "
                        f"terminated={terminated} truncated={truncated}",
                        flush=True,
                    )
                    break
                if known.sleep:
                    time.sleep(float(known.sleep))

            written += _write_records(
                records=recorder.flush_all(),
                jsonl_file=jsonl_file,
                renderer=renderer,
                out_dir=known.out_dir,
                render_every=int(known.render_every),
                render_enabled=not known.no_render,
                bev_range_m=float(known.bev_range_m),
                image_size_px=int(known.image_size),
                ego_offset_m=float(known.ego_offset_m),
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    finally:
        env.close()

    print(f"Wrote {written} JSONL rows to {jsonl_path}", flush=True)
    if not known.no_render:
        print(f"BEV frames: {known.out_dir / 'frames'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
