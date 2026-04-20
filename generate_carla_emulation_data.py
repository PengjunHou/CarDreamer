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
        policy_mode=str(args.policy_mode),
        policy_selector_id=str(args.policy_selector_id),
        policy_override=str(args.policy_override),
        payload_selector_id=str(args.payload_selector_id),
        payload_override_type=str(args.payload_override_type),
        payload_enabled_types=list(args.payload_enabled_types),
        payload_image_jpeg_quality=int(args.payload_image_jpeg_quality),
        max_steps=args.max_steps,
        seed=args.seed,
        task_argv=list(args.task_argv),
        speed_preset=args.speed_preset,
        force_dump_on_max_steps=bool(int(args.force_dump_on_max_steps)),
    )
    collect_policy_rollouts(config)


if __name__ == "__main__":
    main()
