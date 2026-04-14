from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
EMULATION_ROOT = REPO_ROOT / "car_dreamer" / "toolkit" / "emulation"


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(module_name: str):
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    full_name = f"car_dreamer.toolkit.emulation.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, EMULATION_ROOT / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCHEMA = _load_module("schema")
SYNTHETIC = _load_module("synthetic")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical synthetic emulation episodes as JSON files."
    )
    parser.add_argument("--output-dir", default="data/emulation_synth")
    parser.add_argument(
        "--scene-type",
        default="right_turn",
        choices=["right_turn", "left_turn", "lane_change", "car_following"],
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--num-vehicles", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="canonical_emulation")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(int(args.num_episodes)):
        episode = SYNTHETIC.generate_synthetic_canonical_episode(
            scene_type=str(args.scene_type),
            scene_id=f"{args.scene_type}_scene_{index}",
            episode_id=f"{args.scene_type}_episode_{index}",
            num_steps=int(args.num_steps),
            num_vehicles=int(args.num_vehicles),
            dt=float(args.dt),
            seed=int(args.seed) + index,
        )
        payload = SCHEMA.episode_to_dict(episode)
        filename = (
            f"{args.prefix}_{args.scene_type}_ep_{index:03d}"
            f"_steps_{int(args.num_steps)}_veh_{int(args.num_vehicles)}.json"
        )
        path = output_dir / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
