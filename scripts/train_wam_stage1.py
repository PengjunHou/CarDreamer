#!/usr/bin/env python3
"""Offline WAM Stage-1 trainer (WAM Design §16.1). No CARLA needed.

Loads recorded Stage-1 windows (from ``scripts/record_wam_stage1_data.py``) and pretrains the graph
encoder + temporal encoder + deterministic heads (notable + Gaussian trajectory) by minimizing
perception BCE (§15.1) + notable-weighted Gaussian NLL (§15.2). The checkpoint can warm-start Stage 2's
encoder via ``scripts/train_wam_stage2.py --init-from-stage1``.

Example:
    python scripts/train_wam_stage1.py --data-dir data/wam_stage1 --task carla_group_right_turn_auto \
        --steps 2000 --batch-size 8 --ckpt-dir outputs/wam_stage1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WAM perception model (Stage 1).")
    parser.add_argument("--data-dir", required=True, help="directory of recorded .pt window samples")
    parser.add_argument("--task", default="carla_group_right_turn_auto", help="task config for env.wam.*")
    parser.add_argument("--steps", type=int, default=None, help="override env.wam.stage1.steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    import car_dreamer
    from car_dreamer.toolkit.wam import (
        WAMPerceptionModel,
        WAMStage1Dataset,
        WAMStage1Trainer,
        wam_stage1_configs_from_env,
    )

    config = car_dreamer.load_task_configs(args.task)
    perc_cfg, stage1_cfg = wam_stage1_configs_from_env(config)

    if args.steps is not None:
        stage1_cfg.max_steps = args.steps
    if args.batch_size is not None:
        stage1_cfg.batch_size = args.batch_size
    if args.lr is not None:
        stage1_cfg.lr = args.lr
    if args.ckpt_dir is not None:
        stage1_cfg.ckpt_dir = args.ckpt_dir
    stage1_cfg.device = args.device
    stage1_cfg.seed = args.seed

    dataset = WAMStage1Dataset(args.data_dir)
    train_set, val_set = dataset, None
    if args.val_fraction and args.val_fraction > 0.0:
        n_val = max(int(len(dataset) * args.val_fraction), 1)
        n_train = max(len(dataset) - n_val, 1)
        generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)

    model = WAMPerceptionModel(perc_cfg)
    trainer = WAMStage1Trainer(model, stage1_cfg)
    print(
        f"[wam-stage1] samples={len(dataset)} device={stage1_cfg.device} hidden={perc_cfg.hidden_dim} "
        f"steps={stage1_cfg.max_steps} bs={stage1_cfg.batch_size} lr={stage1_cfg.lr} "
        f"history_window={stage1_cfg.history_window} sample_period_s={stage1_cfg.sample_period_s} "
        f"ckpt_dir={stage1_cfg.ckpt_dir}",
        flush=True,
    )
    result = trainer.train(train_set, val_set)
    msg = f"[wam-stage1] done final_loss={result['final_loss']:.4f} steps={int(result['steps'])}"
    for key in ("val_notable_f1", "val_invisible_recall", "val_ade", "val_fde", "val_mean_uncertainty"):
        if key in result:
            msg += f" {key}={result[key]:.4f}"
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
