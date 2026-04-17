"""Evaluate a trained emulation model checkpoint.

Usage:
    conda run -n cardreamer_gnn python evaluate_emulation.py \\
        --checkpoint logdir/emulation/checkpoint_best.pt \\
        --data data/emulation/**/*.json

    # unseen-policy evaluation (hold out P5/P7/P8):
    conda run -n cardreamer_gnn python evaluate_emulation.py \\
        --checkpoint logdir/emulation/checkpoint_best.pt \\
        --data data/emulation/**/*.json \\
        --eval-mode unseen \\
        --unseen-policy-ids P5 P7 P8

    # perturbed-policy evaluation (scale nu by 1.2):
    conda run -n cardreamer_gnn python evaluate_emulation.py \\
        --checkpoint logdir/emulation/checkpoint_best.pt \\
        --data data/emulation/**/*.json \\
        --eval-mode perturbed \\
        --perturbed-nu-scale 1.2
"""
from __future__ import annotations

import importlib.util
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


def _load_module(full_name: str, file_name: str):
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, EMULATION_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_training_module():
    _ensure_pkg("car_dreamer", REPO_ROOT / "car_dreamer")
    _ensure_pkg("car_dreamer.toolkit", REPO_ROOT / "car_dreamer" / "toolkit")
    _ensure_pkg("car_dreamer.toolkit.emulation", EMULATION_ROOT)
    return _load_module("car_dreamer.toolkit.emulation.training", "training.py")


def main() -> None:
    import argparse
    import glob
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate a trained emulation model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt checkpoint file (e.g. logdir/emulation/checkpoint_best.pt).",
    )
    parser.add_argument(
        "--data", nargs="+", required=True,
        help="Episode JSON files or glob patterns (e.g. data/emulation/**/*.json).",
    )
    parser.add_argument(
        "--eval-mode", default="seen", choices=["seen", "unseen", "perturbed"],
        help="Evaluation split strategy.",
    )
    parser.add_argument(
        "--unseen-policy-ids", nargs="*", default=["P5", "P7", "P8"],
        metavar="PID",
        help="Policy IDs held out for val when --eval-mode=unseen.",
    )
    parser.add_argument(
        "--perturbed-nu-scale", type=float, default=1.2,
        help="Frequency multiplier applied to nu when --eval-mode=perturbed.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
    )
    parser.add_argument(
        "--history-len", type=int, default=8,
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device to run inference on (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--scene-type", default="right_turn",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
    )
    parser.add_argument(
        "--loss-type", default="huber", choices=["huber", "mse", "mae"],
    )
    parser.add_argument(
        "--huber-delta", type=float, default=1.0,
    )
    args = parser.parse_args()

    training = _load_training_module()
    torch = sys.modules.get("torch") or __import__("torch")

    # Expand glob patterns
    data_paths: list[str] = []
    for pattern in args.data:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            data_paths.extend(expanded)
        else:
            data_paths.append(pattern)  # may be a direct path
    if not data_paths:
        parser.error("No episode files found for the given --data patterns.")

    device = training.resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {len(data_paths)} files")
    print(f"Eval mode: {args.eval_mode}")

    # Load episodes
    sources = [
        training.parse_episode_source_spec(p, default_scene_type=args.scene_type, default_dt=args.dt)
        for p in data_paths
    ]
    episodes = training.load_episodes_from_sources(sources)

    # Build config for split/loader/eval
    config = training.EmulationTrainingConfig(
        data=data_paths,
        scene_type=args.scene_type,
        dt=args.dt,
        history_len=args.history_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        device=device,
        eval_mode=args.eval_mode,
        unseen_policy_ids=args.unseen_policy_ids or [],
        perturbed_nu_scale=args.perturbed_nu_scale,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
    )

    _, val_dataset, train_indices, val_indices = training.build_dataset_splits(episodes, config)

    if val_dataset is None:
        print("\nNo validation set was created (too few episodes or no unseen policy episodes).")
        return

    print(f"Train episodes: {len(train_indices)}  |  Val episodes: {len(val_indices) or len(train_indices)} (perturbed)")

    _, val_loader = training.build_dataloaders(
        # train_dataset is not used; pass val_dataset as a dummy train
        val_dataset, val_dataset, config
    )

    if val_loader is None:
        print("\nVal loader is empty — no steps with future supervision in the val set.")
        return

    # Load checkpoint and rebuild model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    from car_dreamer.toolkit.emulation.model import GraphGRUEmulationConfig, GraphGRUEmulationModel
    model_config = GraphGRUEmulationConfig(**checkpoint["model_config"])
    model = GraphGRUEmulationModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nLoaded checkpoint (epoch {checkpoint.get('epoch', '?')}, best_metric={checkpoint.get('best_metric', '?'):.6g})")

    # Run evaluation
    metrics = training.evaluate_emulation_model(model, val_loader, device=device, config=config)

    print("\n--- Evaluation Results ---")
    for key, value in sorted(metrics.items()):
        print(f"  {key:<30s}  {value:.6g}")
    print("--------------------------")
    print(f"  total_loss  (weighted)     {metrics.get('total_loss', float('nan')):.6g}")


if __name__ == "__main__":
    main()
