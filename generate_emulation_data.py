"""Generate policy-conditioned emulation data (Section IV.E — Trajectory Generation).

Usage:
    conda run -n cardreamer_gnn python generate_emulation_data.py \
        --out-dir data/emulation \
        --episodes-per-policy 20 \
        --num-steps 30 \
        --num-vehicles 4 \
        --scene-type right_turn

    # generate only specific policies:
    conda run -n cardreamer_gnn python generate_emulation_data.py \
        --out-dir data/emulation \
        --policy-ids P1 P4 P7 \
        --episodes-per-policy 10
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_dreamer.toolkit.emulation.policy_data_generator import PolicyDataGeneratorConfig, generate_and_save, _build_arg_parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
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
