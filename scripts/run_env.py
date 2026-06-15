#!/usr/bin/env python3
"""Run a CarDreamer task environment directly, without any DreamerV3 training.

This mirrors how ``dreamerv3/my_train.py`` builds the environment
(``car_dreamer.create_task``) but skips everything after it: no gym wrappers,
no replay buffer, no agent, no training loop. It just resets the env and steps
it with sampled actions so you can watch the scene in the web visualization
(http://localhost:<carla_port + 7000>/).

The ego vehicle in the right-turn-auto task is driven by CARLA's BasicAgent, so
the sampled action is ignored -- it only exists to satisfy the Gym interface.
Standard tasks instead drive the ego from the action; with random actions the
ego barely moves, so pass --autopilot to let a BasicAgent drive it for viewing.

Examples
--------
    # Watch the default task for 500 steps, then hold the viz for 5 min
    python scripts/run_env.py --carla-port 2000 --steps 500 --hold-seconds 300

    # Watch a standard task with the ego driven by a BasicAgent (so it moves)
    python scripts/run_env.py --task carla_roundabout --autopilot

    # Run exactly one full episode (until terminated/truncated), then exit
    python scripts/run_env.py --episodes 1

    # Forward arbitrary env overrides straight to create_task
    python scripts/run_env.py -- --env.num_group_vehs=2 --env.num_background_vehs=4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_carla_pythonapi() -> None:
    """Put the CARLA PythonAPI on sys.path (same as my_train.py)."""
    carla_root = os.environ.get("CARLA_ROOT", "/home/peh324/carla_simulator")
    for sub in ("PythonAPI", "PythonAPI/carla"):
        path = os.path.join(carla_root, sub)
        if path not in sys.path:
            sys.path.append(path)


def build_env(task: str, env_args: List[str]):
    """Build the gym env exactly like create_task (common.yaml + tasks.yaml + CLI overrides)."""
    import gymnasium as gym

    import car_dreamer
    from car_dreamer import toolkit

    config = car_dreamer.load_task_configs(task)
    config, _ = toolkit.Flags(config).parse_known(env_args)
    env = gym.make(config.env.name, config=config.env)
    return env, config


def _live_actor_counts(sim) -> Tuple[int, int]:
    world = getattr(getattr(sim, "_world", None), "_world", None)
    if world is None:
        return 0, 0
    actors = world.get_actors()
    return len(list(actors.filter("vehicle.*"))), len(list(actors.filter("walker.pedestrian.*")))


def _ego_speed_kmh(sim) -> float:
    ego = getattr(sim, "ego", None)
    if ego is None:
        return 0.0
    v = ego.get_velocity()
    return 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Optional BasicAgent autopilot for standard (RL-controlled) tasks
#
# Standard tasks drive the ego from the action passed to env.step(); with random
# actions the ego barely moves. To watch the scene we instead let a CARLA
# BasicAgent drive. The env re-applies the action-derived control every step via
# apply_control(), so we replace that bound method on the instance with a closure
# that ignores the action and applies the agent's control. Must be reinstalled
# after every reset because the ego actor is respawned.
# ---------------------------------------------------------------------------


def _resolve_goal_location(sim):
    """A task goal from config (ego_path end / lane_end_point), or None."""
    import carla

    cfg = getattr(sim, "_config", None)
    if cfg is None:
        return None
    ego_path = getattr(cfg, "ego_path", None)
    if ego_path:
        p = list(ego_path)[-1]
        return carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]))
    lane_end = getattr(cfg, "lane_end_point", None)
    if lane_end:
        return carla.Location(x=float(lane_end[0]), y=float(lane_end[1]), z=float(lane_end[2]))
    return None


def _forward_destination(sim, carla_map, dist: float = 40.0):
    """A waypoint ~dist meters ahead of the ego along the road (rolling goal)."""
    if carla_map is None:
        return None
    try:
        wp = carla_map.get_waypoint(sim.ego.get_location())
        nxt = wp.next(float(dist))
        if nxt:
            return nxt[-1].transform.location
    except Exception:
        pass
    return None


def install_autopilot(sim, target_speed_kmh: float) -> bool:
    """Make a CARLA BasicAgent drive the ego. Re-call after every reset.

    Returns False (and changes nothing) if the env already self-drives, e.g. the
    carla_group_right_turn_auto task, which owns its own BasicAgent.
    """
    if getattr(sim, "agent", None) is not None:
        return False

    import carla
    from agents.navigation.basic_agent import BasicAgent

    ego = getattr(sim, "ego", None)
    if ego is None:
        return False

    world = getattr(getattr(sim, "_world", None), "_world", None)
    carla_map = world.get_map() if world is not None else None

    agent = BasicAgent(ego)
    try:
        agent.set_target_speed(float(target_speed_kmh))
    except Exception:
        pass
    goal = _resolve_goal_location(sim) or _forward_destination(sim, carla_map)
    if goal is not None:
        try:
            agent.set_destination(goal)
        except Exception:
            pass

    sim._autopilot_agent = agent
    sim._autopilot_map = carla_map

    def _autopilot_apply_control(action):
        del action
        ag = getattr(sim, "_autopilot_agent", None)
        if ag is None:
            sim.ego.apply_control(carla.VehicleControl())
            return
        if ag.done():  # reached goal -> keep rolling forward so the car doesn't freeze
            nxt = _forward_destination(sim, getattr(sim, "_autopilot_map", None))
            if nxt is not None:
                try:
                    ag.set_destination(nxt)
                except Exception:
                    pass
        sim.ego.apply_control(ag.run_step())

    sim.apply_control = _autopilot_apply_control
    return True


def print_scene(env, tag: str) -> None:
    sim = env.unwrapped
    ego = getattr(sim, "ego", None)
    ego_id = int(ego.id) if ego is not None else None
    group_vehs = list(getattr(sim, "group_vehs", []))
    comm_proc = getattr(sim, "_comm_process", None)
    step = int(getattr(sim, "_time_step", 0))
    if comm_proc is not None:
        in_flight = len(comm_proc.in_flight)
        ego_msgs = len(comm_proc.available_messages(step)) if comm_proc.policy is not None else 0
    else:
        in_flight = 0
        ego_msgs = 0
    vehicles, walkers = _live_actor_counts(sim)

    print(
        f"[{tag}] step={getattr(sim, '_time_step', '?')} "
        f"ego={ego_id} speed={_ego_speed_kmh(sim):.1f}km/h "
        f"group_vehs={len(group_vehs)} {[int(v.id) for v in group_vehs]} "
        f"| comm in_flight={in_flight} ego_received={ego_msgs} "
        f"| live vehicles={vehicles} walkers={walkers}",
        flush=True,
    )


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run a CarDreamer task env directly (no training).",
    )
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=500,
                        help="Step-budget mode: total steps to run, auto-resetting on episode end. "
                             "Ignored when --episodes is set.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episode mode: run this many COMPLETE episodes (each until "
                             "terminated/truncated), then stop. Overrides --steps.")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="Episode-mode safety cap: force-end an episode after this many steps "
                             "if it never terminates (default: uncapped).")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Seconds to sleep between steps (slows things down so you can watch).")
    parser.add_argument("--hold-seconds", type=float, default=0.0,
                        help="Keep stepping after the main run so the web viz stays live.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--autopilot", action="store_true", default=False,
                        help="Drive the ego with a CARLA BasicAgent (follows roads toward the "
                             "task goal) instead of random actions. No effect on tasks that "
                             "already self-drive, e.g. carla_group_right_turn_auto.")
    parser.add_argument("--autopilot-speed", type=float, default=20.0,
                        help="Target speed in km/h for --autopilot (default: 20).")
    parser.add_argument("--no-display", dest="display", action="store_false", default=True,
                        help="Disable the web visualization monitor.")
    # Anything after `--` (or any unrecognized --env.* flag) is forwarded to create_task.
    known, passthrough = parser.parse_known_args()
    # argparse keeps a literal `--` separator in the remainder; drop it.
    passthrough = [arg for arg in passthrough if arg != "--"]
    return known, passthrough


def main() -> int:
    known, passthrough = parse_args()
    if known.steps < 0:
        raise ValueError("--steps must be non-negative")
    if known.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if known.sleep < 0 or known.hold_seconds < 0:
        raise ValueError("--sleep and --hold-seconds must be non-negative")
    if known.episodes is not None and known.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if known.max_episode_steps is not None and known.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive")

    _setup_carla_pythonapi()

    env_args = [
        f"--env.world.carla_port={known.carla_port}",
        f"--env.display.enable={bool(known.display)}",
        *passthrough,
    ]

    env, _ = build_env(known.task, env_args)
    sim = env.unwrapped
    viz_url = f"http://localhost:{known.carla_port + 7000}/"

    def reset_and_setup(tag: str) -> None:
        env.reset(seed=known.seed)
        if known.autopilot:
            installed = install_autopilot(sim, known.autopilot_speed)
            if tag == "after reset":
                if installed:
                    print(f"Autopilot ON: ego driven by CARLA BasicAgent at ~{known.autopilot_speed:.0f} "
                          f"km/h (sampled action ignored).", flush=True)
                else:
                    print("Autopilot requested, but this task already self-drives; "
                          "--autopilot ignored.", flush=True)
        print_scene(env, tag)

    try:
        if known.display:
            print(f"Web visualization: {viz_url}", flush=True)

        if known.episodes is not None:
            # Episode mode: run N complete episodes, each until terminated/truncated.
            for ep in range(known.episodes):
                reset_and_setup(f"episode {ep} reset")
                ep_step = 0
                while True:
                    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                    if ep_step % known.print_every == 0:
                        print_scene(env, f"ep {ep} step {ep_step}")
                    if terminated or truncated:
                        print(f"episode {ep} finished at step {ep_step}: "
                              f"terminated={terminated} truncated={truncated}", flush=True)
                        break
                    ep_step += 1
                    if known.max_episode_steps and ep_step >= known.max_episode_steps:
                        print(f"episode {ep} reached --max-episode-steps={known.max_episode_steps} "
                              f"without terminating; moving on.", flush=True)
                        break
                    if known.sleep:
                        time.sleep(known.sleep)
            print(f"Completed {known.episodes} episode(s); exiting.", flush=True)
        else:
            # Step-budget mode: run a fixed number of steps, auto-resetting on episode end.
            reset_and_setup("after reset")
            for step in range(known.steps):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if step % known.print_every == 0:
                    print_scene(env, f"step {step}")
                if terminated or truncated:
                    print(f"episode ended: terminated={terminated} truncated={truncated}", flush=True)
                    reset_and_setup("after auto-reset")
                if known.sleep:
                    time.sleep(known.sleep)

            if known.hold_seconds:
                end = time.monotonic() + known.hold_seconds
                print(f"Holding for {known.hold_seconds:.1f}s so the viz stays live. Ctrl-C to stop.", flush=True)
                hold_step = 0
                while time.monotonic() < end:
                    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                    if hold_step % known.print_every == 0:
                        print_scene(env, f"hold {hold_step}")
                    if terminated or truncated:
                        reset_and_setup("after hold-reset")
                    hold_step += 1
                    if known.sleep:
                        time.sleep(known.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
