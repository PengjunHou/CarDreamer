from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_dreamer.toolkit.emulation.carla_rollout_collector import build_arg_parser, collect_policy_rollouts, CARLARolloutCollectorConfig


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = CARLARolloutCollectorConfig(
        task_name=args.task_name,
        output_dir=args.output_dir,
        episodes_per_policy=args.episodes_per_policy,
        policy_ids=list(args.policy_ids),
        max_steps=args.max_steps,
        seed=args.seed,
        task_argv=list(args.task_argv),
        speed_preset=args.speed_preset,
    )
    collect_policy_rollouts(config)


if __name__ == "__main__":
    main()
