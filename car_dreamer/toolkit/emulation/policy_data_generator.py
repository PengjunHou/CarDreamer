"""Policy-conditioned synthetic data generator (Section IV.E — Trajectory Generation).

For each fixed policy in the P1-P8 family, generates multiple simulation episodes
to produce a diverse dataset covering the full range of collaboration behaviors.

Usage (CLI):
    python -m car_dreamer.toolkit.emulation.policy_data_generator \\
        --out-dir data/emulation \\
        --episodes-per-policy 20 \\
        --num-steps 30 \\
        --num-vehicles 4 \\
        --scene-type right_turn

Usage (API):
    from car_dreamer.toolkit.emulation.policy_data_generator import generate_policy_dataset
    episodes = generate_policy_dataset(episodes_per_policy=10)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from .policy import ALL_POLICIES, list_policy_ids
from .schema import CanonicalEpisodeRecord, episode_to_dict
from .synthetic import generate_synthetic_canonical_episode


@dataclass
class PolicyDataGeneratorConfig:
    out_dir: str = "data/emulation"
    episodes_per_policy: int = 20
    num_steps: int = 30
    num_vehicles: int = 4
    scene_type: str = "right_turn"
    dt: float = 0.1
    seed: int = 42
    policy_ids: List[str] = field(default_factory=list)  # empty = all P1-P8
    save_json: bool = True  # write one JSON file per episode


def generate_policy_dataset(
    config: PolicyDataGeneratorConfig | None = None,
    *,
    episodes_per_policy: int = 20,
    num_steps: int = 30,
    num_vehicles: int = 4,
    scene_type: str = "right_turn",
    dt: float = 0.1,
    seed: int = 42,
    policy_ids: Sequence[str] | None = None,
) -> List[CanonicalEpisodeRecord]:
    """Generate synthetic episodes for every requested policy.

    Returns a list of CanonicalEpisodeRecord, one per (policy, episode_index) pair.
    """
    if config is None:
        config = PolicyDataGeneratorConfig(
            episodes_per_policy=episodes_per_policy,
            num_steps=num_steps,
            num_vehicles=num_vehicles,
            scene_type=scene_type,
            dt=dt,
            seed=seed,
            policy_ids=list(policy_ids or []),
        )

    active_policy_ids = config.policy_ids if config.policy_ids else list_policy_ids()
    rng = random.Random(config.seed)

    episodes: List[CanonicalEpisodeRecord] = []
    for pid in active_policy_ids:
        for ep_idx in range(config.episodes_per_policy):
            ep_seed = rng.randint(0, 2**31 - 1)
            episode = generate_synthetic_canonical_episode(
                scene_type=config.scene_type,
                scene_id=f"{pid}_scene_{ep_idx:04d}",
                episode_id=f"{pid}_ep_{ep_idx:04d}",
                num_steps=config.num_steps,
                num_vehicles=config.num_vehicles,
                dt=config.dt,
                seed=ep_seed,
                policy_id=pid,
            )
            episodes.append(episode)

    return episodes


def save_policy_dataset(
    episodes: List[CanonicalEpisodeRecord],
    out_dir: str | Path,
) -> Dict[str, List[str]]:
    """Save each episode as a separate JSON file under out_dir/<policy_id>/.

    Returns a dict mapping policy_id -> list of saved file paths.
    """
    out = Path(out_dir)
    saved: Dict[str, List[str]] = {}
    for episode in episodes:
        pid = episode.policy_id or "unknown"
        policy_dir = out / pid
        policy_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{episode.episode_id}.json"
        fpath = policy_dir / filename
        with open(fpath, "w", encoding="utf-8") as fh:
            json.dump(episode_to_dict(episode), fh, ensure_ascii=False)
        saved.setdefault(pid, []).append(str(fpath))
    return saved


def generate_and_save(config: PolicyDataGeneratorConfig) -> Dict[str, List[str]]:
    """End-to-end: generate episodes and save to disk."""
    episodes = generate_policy_dataset(config)
    if config.save_json:
        saved = save_policy_dataset(episodes, config.out_dir)
        total = sum(len(v) for v in saved.values())
        print(f"[policy_data_generator] Saved {total} episodes to '{config.out_dir}':")
        for pid, paths in sorted(saved.items()):
            print(f"  {pid}: {len(paths)} episodes")
        return saved
    return {}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate policy-conditioned synthetic emulation data.")
    parser.add_argument("--out-dir", default="data/emulation")
    parser.add_argument("--episodes-per-policy", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--num-vehicles", type=int, default=4)
    parser.add_argument("--scene-type", default="right_turn")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy-ids",
        nargs="*",
        default=[],
        help="Subset of policy IDs to generate (e.g. P1 P4 P7). Defaults to all P1-P8.",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing JSON files.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = PolicyDataGeneratorConfig(
        out_dir=args.out_dir,
        episodes_per_policy=args.episodes_per_policy,
        num_steps=args.num_steps,
        num_vehicles=args.num_vehicles,
        scene_type=args.scene_type,
        dt=args.dt,
        seed=args.seed,
        policy_ids=list(args.policy_ids),
        save_json=not args.no_save,
    )
    generate_and_save(config)


if __name__ == "__main__":
    main()
