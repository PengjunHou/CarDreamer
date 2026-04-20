from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from .policy import list_policy_ids


@dataclass
class CARLARolloutCollectorConfig:
    task_name: str = "carla_group_right_turn_auto"
    output_dir: str = "data/emulation"
    episodes_per_policy: int = 1
    policy_ids: List[str] = field(default_factory=list)
    policy_mode: str = "fixed"
    policy_selector_id: str = "default"
    policy_override: str = ""
    payload_selector_id: str = "default"
    payload_override_type: str = ""
    payload_enabled_types: List[str] = field(default_factory=lambda: ["images", "tokens"])
    payload_image_jpeg_quality: int = 80
    max_steps: int = 256
    seed: int = 0
    task_argv: List[str] = field(default_factory=list)
    speed_preset: str = "fast_episode"
    force_dump_on_max_steps: bool = True


def build_policy_rollout_argv(
    config: CARLARolloutCollectorConfig,
    *,
    policy_id: str,
    episode_index: int,
    policy_dir: str | Path,
) -> List[str]:
    policy_mode = str(config.policy_mode).strip().lower() or "fixed"
    scene_id = (
        f"adaptive_scene_{episode_index:04d}"
        if policy_mode == "adaptive"
        else f"{policy_id}_scene_{episode_index:04d}"
    )
    argv = list(config.task_argv)
    argv.extend(
        [
            f"--env.policy_mode={policy_mode}",
            f"--env.policy_id={policy_id}",
            f"--env.policy_selector_id={config.policy_selector_id}",
            f"--env.policy_override={config.policy_override}",
            f"--env.scene_id={scene_id}",
            f"--env.speed_preset={config.speed_preset}",
            "--env.display.enable=False",
            f"--env.emulation_dump_dir={policy_dir}",
            f"--env.payload.selector_id={config.payload_selector_id}",
            f"--env.payload.override_type={config.payload_override_type}",
            f"--env.payload.enabled_types=[{','.join(config.payload_enabled_types)}]",
            f"--env.payload.image_jpeg_quality={int(config.payload_image_jpeg_quality)}",
            "--env.dump_emulation_records_on_episode_end=True",
            "--env.dump_vlm_records_on_episode_end=False",
        ]
    )
    return argv


def _unwrap_env(env):
    current = env
    seen_ids = set()
    while hasattr(current, "unwrapped") and id(current) not in seen_ids:
        seen_ids.add(id(current))
        unwrapped = getattr(current, "unwrapped")
        if unwrapped is current:
            break
        current = unwrapped
    return current


def _force_dump_episode(env, *, suffix: str = "max_steps") -> str:
    base_env = _unwrap_env(env)
    if not hasattr(base_env, "dump_emulation_episode"):
        raise RuntimeError("Environment does not support dump_emulation_episode for forced dumps.")
    dump_dir = Path(str(getattr(base_env, "_emulation_dump_dir", "data")))
    dump_dir.mkdir(parents=True, exist_ok=True)
    step = int(getattr(base_env, "_time_step", 0))
    path = dump_dir / f"emulation_episode_{suffix}_step_{step}.json"
    base_env.dump_emulation_episode(str(path))
    if hasattr(base_env, "_emulation_episode_dumped"):
        base_env._emulation_episode_dumped = True
    return str(path)


def rollout_single_episode(
    env,
    *,
    seed: int,
    max_steps: int,
    force_dump_on_max_steps: bool = True,
) -> str:
    _, info = env.reset(seed=int(seed))
    done = False
    steps = 0
    while not done and steps < int(max_steps):
        action_space = getattr(env, "action_space", None)
        action = action_space.sample() if action_space is not None else 0
        _, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    if not done:
        if bool(force_dump_on_max_steps):
            return _force_dump_episode(env, suffix="max_steps")
        raise RuntimeError(
            f"Episode did not finish within max_steps={int(max_steps)}; no emulation dump was created."
        )
    dump_path = str(info.get("emulation_dump_path", "")).strip()
    if not dump_path:
        raise RuntimeError("Episode ended without an emulation_dump_path in info.")
    return dump_path


def collect_policy_rollouts(
    config: CARLARolloutCollectorConfig,
    *,
    task_factory: Callable[[str, Sequence[str] | None], Tuple[object, object]] | None = None,
) -> Dict[str, List[str]]:
    if task_factory is None:
        import car_dreamer

        task_factory = car_dreamer.create_task

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_mode = str(config.policy_mode).strip().lower() or "fixed"
    policy_ids = list(config.policy_ids or list_policy_ids())
    if policy_mode == "adaptive":
        policy_ids = ["adaptive"]
    saved: Dict[str, List[str]] = {}

    for policy_id in policy_ids:
        policy_dir = output_dir / str(policy_id)
        policy_dir.mkdir(parents=True, exist_ok=True)
        for episode_index in range(int(config.episodes_per_policy)):
            argv = build_policy_rollout_argv(
                config,
                policy_id=str(policy_id),
                episode_index=int(episode_index),
                policy_dir=policy_dir,
            )
            env, _ = task_factory(config.task_name, argv)
            try:
                dump_path = rollout_single_episode(
                    env,
                    seed=int(config.seed) + int(episode_index),
                    max_steps=int(config.max_steps),
                    force_dump_on_max_steps=bool(config.force_dump_on_max_steps),
                )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            saved.setdefault(str(policy_id), []).append(dump_path)
    return saved


def resolve_episode_paths(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            paths.extend(sorted(expanded))
        elif Path(pattern).exists():
            paths.append(str(Path(pattern)))
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect policy-conditioned CARLA rollout episodes.")
    parser.add_argument("--task-name", default="carla_group_right_turn_auto")
    parser.add_argument("--output-dir", default="data/emulation")
    parser.add_argument("--episodes-per-policy", type=int, default=1)
    parser.add_argument("--policy-ids", nargs="*", default=[])
    parser.add_argument("--policy-mode", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--policy-selector-id", default="default")
    parser.add_argument("--policy-override", default="")
    parser.add_argument("--payload-selector-id", default="default")
    parser.add_argument("--payload-override-type", default="")
    parser.add_argument("--payload-enabled-types", nargs="*", default=["images", "tokens"])
    parser.add_argument("--payload-image-jpeg-quality", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-preset", default="fast_episode")
    parser.add_argument(
        "--force-dump-on-max-steps",
        type=int,
        default=1,
        help="Force-write the current emulation episode when max_steps is reached before termination.",
    )
    parser.add_argument(
        "--task-argv",
        nargs="*",
        default=[],
        help="Additional task overrides forwarded to car_dreamer.create_task().",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, List[str]]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = CARLARolloutCollectorConfig(
        task_name=str(args.task_name),
        output_dir=str(args.output_dir),
        episodes_per_policy=int(args.episodes_per_policy),
        policy_ids=list(args.policy_ids),
        policy_mode=str(args.policy_mode),
        policy_selector_id=str(args.policy_selector_id),
        policy_override=str(args.policy_override),
        payload_selector_id=str(args.payload_selector_id),
        payload_override_type=str(args.payload_override_type),
        payload_enabled_types=list(args.payload_enabled_types),
        payload_image_jpeg_quality=int(args.payload_image_jpeg_quality),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        task_argv=list(args.task_argv),
        speed_preset=str(args.speed_preset),
        force_dump_on_max_steps=bool(int(args.force_dump_on_max_steps)),
    )
    saved = collect_policy_rollouts(config)
    total = sum(len(paths) for paths in saved.values())
    print(f"[carla_rollout_collector] Saved {total} episodes to '{config.output_dir}':")
    for policy_id, paths in sorted(saved.items()):
        print(f"  {policy_id}: {len(paths)} episodes")
    return saved


if __name__ == "__main__":
    main()
