#!/usr/bin/env python3
"""Offline WAM Stage-2 trainer (WAM Design §16.2). No CARLA needed.

Loads recorded flow-matching samples (from ``scripts/record_wam_flow_data.py``) and trains the Graph
Flow-Matching UWM by minimizing ``L_FM``. The graph encoder is trained jointly (use ``--freeze-encoder``
to keep it fixed).

Example:
    python scripts/train_wam_stage2.py --data-dir data/wam_flow --task carla_group_right_turn_auto \
        --steps 2000 --batch-size 16 --ckpt-dir outputs/wam_stage2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WAM Graph Flow-Matching UWM (Stage 2).")
    parser.add_argument("--data-dir", required=True, help="directory of recorded .pt samples")
    parser.add_argument("--task", default="carla_group_right_turn_auto", help="task config for env.wam.*")
    parser.add_argument("--steps", type=int, default=None, help="override env.wam.stage2.steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--init-from-stage1", default=None,
                        help="Stage-1 checkpoint to warm-start the graph encoder from (§16.1 -> §16.2)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    import car_dreamer
    from car_dreamer.toolkit.wam import (
        WAMFlowDataset,
        WAMStage2Trainer,
        WAMUnifiedWorldModel,
        wam_configs_from_env,
    )

    config = car_dreamer.load_task_configs(args.task)
    graph_cfg, flow_cfg, stage2_cfg = wam_configs_from_env(config)

    if args.steps is not None:
        stage2_cfg.max_steps = args.steps
    if args.batch_size is not None:
        stage2_cfg.batch_size = args.batch_size
    if args.lr is not None:
        stage2_cfg.lr = args.lr
    if args.ckpt_dir is not None:
        stage2_cfg.ckpt_dir = args.ckpt_dir
    stage2_cfg.device = args.device
    stage2_cfg.seed = args.seed
    stage2_cfg.freeze_encoder = args.freeze_encoder

    dataset = WAMFlowDataset(args.data_dir)
    train_set, val_set = dataset, None
    if args.val_fraction and args.val_fraction > 0.0:
        n_val = max(int(len(dataset) * args.val_fraction), 1)
        n_train = max(len(dataset) - n_val, 1)
        generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)

    model = WAMUnifiedWorldModel(graph_cfg, flow_cfg)
    if args.init_from_stage1:
        from car_dreamer.toolkit.wam import init_encoder_from_stage1

        missing, unexpected = init_encoder_from_stage1(model, args.init_from_stage1)
        print(
            f"[wam-stage2] warm-started encoder from {args.init_from_stage1} "
            f"(missing={len(missing)} unexpected={len(unexpected)})",
            flush=True,
        )
    trainer = WAMStage2Trainer(model, stage2_cfg)
    print(
        f"[wam-stage2] samples={len(dataset)} device={stage2_cfg.device} hidden={flow_cfg.hidden_dim} "
        f"steps={stage2_cfg.max_steps} bs={stage2_cfg.batch_size} lr={stage2_cfg.lr} "
        f"freeze_encoder={stage2_cfg.freeze_encoder} ckpt_dir={stage2_cfg.ckpt_dir}",
        flush=True,
    )
    result = trainer.train(train_set, val_set)
    msg = f"[wam-stage2] done final_loss={result['final_loss']:.4f} steps={int(result['steps'])}"
    if "val_loss" in result:
        msg += f" val_loss={result['val_loss']:.4f}"
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
