#!/usr/bin/env python3
"""Evaluate policy-conditioned Stage-1 uncertainty on recorded samples.

Writes one CSV row per Stage-1 policy sample:
``step, episode_id, policy_type, selected_vehicle_ids, modality_by_vehicle,
notable_object_ids, uncertainty, ade, fde``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WAM Stage-1 policy uncertainty.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Stage-1 checkpoint path")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wam_stage1_policy"))
    parser.add_argument("--task", default="carla_group_right_turn_auto")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/wam_stage1_uncertainty"))
    parser.add_argument("--csv-name", default="uncertainty_by_policy.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    import car_dreamer
    from car_dreamer.toolkit.wam import (
        WAMPerceptionModel,
        WAMPerceptionConfig,
        WAMStage1Dataset,
        COMM_REPLAY_METADATA_FIELDS,
        evaluate_stage1_uncertainty_rows,
        wam_stage1_configs_from_env,
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    perc_cfg = ckpt.get("perception_config") if isinstance(ckpt, dict) else None
    if perc_cfg is None:
        config = car_dreamer.load_task_configs(args.task)
        perc_cfg, _ = wam_stage1_configs_from_env(config)
    elif isinstance(perc_cfg, dict):
        perc_cfg = WAMPerceptionConfig(**perc_cfg)

    model = WAMPerceptionModel(perc_cfg)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)

    dataset = WAMStage1Dataset(args.data_dir)
    rows = evaluate_stage1_uncertainty_rows(model, dataset, device=args.device, limit=args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.csv_name
    fields = [
        "step",
        "episode_id",
        "policy_type",
        "selected_vehicle_ids",
        "modality_by_vehicle",
        "notable_object_ids",
        "uncertainty",
        "motion_uncertainty",
        "coverage_uncertainty",
        "total_uncertainty",
        "route_coverage_quality_mean",
        "poor_coverage_risk_mean",
        "ade",
        "fde",
        *COMM_REPLAY_METADATA_FIELDS,
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in ("selected_vehicle_ids", "modality_by_vehicle", "notable_object_ids"):
                encoded[key] = json.dumps(encoded.get(key, []), sort_keys=True)
            writer.writerow({key: encoded.get(key, "") for key in fields})

    print(f"Wrote {len(rows)} rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
