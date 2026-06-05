"""Apply P1-P8 policies to real CARLA canonical episodes and save variants.

Usage:
    conda run -n cardreamer_gnn python apply_policies_to_real_data.py \
        --episodes data/emulation_episode_*.json \
        --out-dir data/emulation_real_policy \
        --policy-ids P1 P2 P3 P4 P5 P6 P7 P8
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EMULATION_ROOT = REPO_ROOT / "car_dreamer" / "toolkit" / "emulation"


def _ensure_pkg(name, path):
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m


def _load(full_name, filename):
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, EMULATION_ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    schema = _load("car_dreamer.toolkit.emulation.schema", "schema.py")
    _load("car_dreamer.toolkit.emulation.features", "features.py")
    _load("car_dreamer.toolkit.emulation.queries", "queries.py")
    policy_mod = _load("car_dreamer.toolkit.emulation.policy", "policy.py")
    training = _load("car_dreamer.toolkit.emulation.training", "training.py")

    parser = argparse.ArgumentParser(description="Apply fixed policies to real CARLA episodes.")
    parser.add_argument("--episodes", nargs="+", required=True,
                        help="Canonical episode JSON files or glob patterns.")
    parser.add_argument("--out-dir", default="data/emulation_real_policy")
    parser.add_argument("--policy-ids", nargs="+",
                        default=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"])
    args = parser.parse_args()

    # Expand globs
    paths = []
    for pattern in args.episodes:
        expanded = glob.glob(pattern, recursive=True)
        paths.extend(expanded if expanded else [pattern])

    out_dir = Path(args.out_dir)
    policies = [policy_mod.get_policy(pid) for pid in args.policy_ids]

    total = 0
    for path in paths:
        episode = training.load_episode_from_path(path)
        for pol in policies:
            new_ep = policy_mod.apply_action_to_episode(episode, pol)
            save_dir = out_dir / pol.policy_id
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(path).stem
            out_path = save_dir / f"{stem}__{pol.policy_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(schema.episode_to_dict(new_ep), f, ensure_ascii=False)
            print(f"  saved {out_path}")
            total += 1

    print(f"\nDone: {total} episodes saved to {out_dir}/")


if __name__ == "__main__":
    main()
