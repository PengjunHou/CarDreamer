#!/usr/bin/env python3
"""Offline driver for the Lyapunov-guided world-action policy search (V2X paper Sec IV) — CARLA-free.

Runs :class:`LyapunovScheduler.run_offline` over a per-step sequence of :class:`RolloutContext` (a synthetic
episode by default; a pickled context list via ``--contexts``), scoring candidate action chunks with the
reused Stage-1 ``U_φ`` (``--stage1-ckpt``; rule fallback if omitted), and writes a per-slot trace
(``lyapunov_trace.csv`` + ``.jsonl``) and ``summary.json`` (time-avg U_c, budget compliance Z(T)/T, mean
backlog, epochs/replans). The analysis/ablation scripts consume ``lyapunov_trace.csv``.

Examples:
    python scripts/run_wam_lyapunov_offline.py --synthetic --steps 120 --out-dir outputs/wam_lyapunov_smoke
    python scripts/run_wam_lyapunov_offline.py --stage1-ckpt outputs/wam_stage1/stage1_step2000.pt \
        --synthetic --steps 300 --lam 2.0 --budget 0.4 --out-dir outputs/wam_lyapunov_run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Lyapunov world-action policy search offline.")
    p.add_argument("--stage1-ckpt", type=Path, default=None, help="Stage-1 U_φ checkpoint (rule fallback if omitted)")
    p.add_argument("--contexts", type=Path, default=None, help="pickled List[RolloutContext] (overrides --synthetic)")
    p.add_argument("--synthetic", action="store_true", help="generate a synthetic episode (default when no --contexts)")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--n-members", type=int, default=1)
    p.add_argument("--route-len", type=float, default=80.0)
    p.add_argument("--bev-size", type=int, default=24, help="coverage/BEV raster resolution (coarse = faster)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    # scheduler knobs
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--c0", type=float, default=0.5)
    p.add_argument("--budget", type=float, default=0.5, help="B̄_bgt bandwidth-ratio budget")
    p.add_argument("--f-max", type=int, default=20)
    p.add_argument("--n-min", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--ts-seconds", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/wam_lyapunov"))
    return p.parse_args()


# =====================================================================
# Model loading (Stage-1 U_φ) — mirrors v2v_comm_mixin._load_wam_predictor
# =====================================================================


def load_perception_model(ckpt_path: Path, device: str = "cpu"):
    import torch

    from car_dreamer.toolkit.wam import WAMPerceptionConfig, WAMPerceptionModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("perception_config") if isinstance(ckpt, dict) else None
    if cfg is None:
        raise ValueError(f"{ckpt_path} has no 'perception_config'; not a Stage-1 checkpoint")
    if isinstance(cfg, dict):
        cfg = WAMPerceptionConfig(**cfg)
    elif not is_dataclass(cfg):
        raise TypeError(f"unsupported perception_config type: {type(cfg)!r}")
    model = WAMPerceptionModel(cfg).to(device)
    model.load_state_dict(ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt)
    model.eval()
    return model


# =====================================================================
# Synthetic episode
# =====================================================================


def build_synthetic_contexts(
    *, steps: int, n_members: int = 1, route_len: float = 80.0, seed: int = 0,
    dt: float = 0.1, sensor_period: int = 1, link_rate_bps: float = 1.0e5, bev_size: int = 24,
    route_waypoints: int = 2,
) -> List["object"]:
    """A deterministic CARLA-free episode: ego on a straight route, ``n_members`` collaborators ahead, and a
    route-adjacent notable object invisible to ego but visible to a member (so cooperation can help).

    ``bev_size`` sets the coverage/BEV raster resolution; a coarse grid (24) keeps the route-corridor
    coverage cheap enough for long episodes and ablation sweeps (the corridor metric is coarse anyway)."""
    from car_dreamer.toolkit.wam import BevSpec, GraphBuildSpec, ObjectState, RolloutContext, VehicleNodeInput

    rng = random.Random(int(seed))
    bev_spec = BevSpec(size=int(bev_size))
    route = tuple((float(x), 0.0) for x in range(0, int(route_len) + 1, 5))

    def link_rate_fn(mid, dist, ratio):
        return float(link_rate_bps) * max(float(ratio), 0.05) / (1.0 + float(dist) / 50.0)

    contexts = []
    for t in range(int(steps)):
        ego_route = tuple((5.0 * (k + 1), 0.0) for k in range(max(int(route_waypoints), 1)))
        ego = VehicleNodeInput(actor_id=1, is_ego=True, agent_slot=0, x=0.0, y=0.0, z=0.0,
                               vx=3.0, vy=0.0, yaw=0.0, route_xy=ego_route)
        # Collaborators sit DOWNSTREAM (facing forward) on the far part of the corridor that lies beyond the
        # ego's own sensor range, so their shared observations fill a real coverage gap.
        far = 0.55 * float(route_len)
        members = []
        for k in range(int(n_members)):
            mx = far + 6.0 * k + 3.0 * math.sin(0.2 * t + k)
            members.append(VehicleNodeInput(actor_id=7 + k, is_ego=False, agent_slot=1 + k,
                                            x=mx, y=0.0, z=0.0, vx=0.0, vy=0.0, yaw=0.0, route_xy=()))
        member_ids = tuple(int(m.actor_id) for m in members) or (7,)
        ox = far + 3.0 * math.sin(0.15 * t)
        objects = (
            ObjectState(actor_id=100, actor_type="vehicle.x", object_class="vehicle", x=ox, y=1.0, z=0.0,
                        vx=0.0, vy=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5,
                        visible_to_ego=False, visible_to_collaborators=member_ids),  # far, only members see it
            ObjectState(actor_id=101, actor_type="walker", object_class="pedestrian", x=8.0 + rng.uniform(-1, 1),
                        y=0.5, z=0.0, vx=0.0, vy=0.0, yaw=0.0, length=0.6, width=0.6, height=1.7,
                        visible_to_ego=True, visible_to_collaborators=()),  # near, ego sees it
        )
        contexts.append(RolloutContext(
            ego=ego, collaborators=tuple(members), objects=objects, route_xy=route, notable_ids=(100, 101),
            link_rate_fn=link_rate_fn, graph_spec=GraphBuildSpec(route_waypoints=int(route_waypoints)), bev_spec=bev_spec,
            ego_v0=3.0, dt_seconds=dt, sensor_period_steps=int(sensor_period),
        ))
    return contexts


# =====================================================================
# Run + write
# =====================================================================


def run_scheduler(contexts: List["object"], *, model=None, cfg=None, alpha: Optional[float] = None,
                  realized_uncertainty_fn=None):
    """Run the scheduler over ``contexts`` (no file I/O). Returns ``(scheduler, rows, summary)``.

    Reused by the analysis/ablation scripts. ``alpha`` overrides the scorer's motion/coverage mix; if
    ``None`` it falls back to ``cfg._alpha`` (set by the CLI) or 0.5.
    """
    from car_dreamer.toolkit.wam import LyapunovScheduler, SchedulerConfig, WorldActionScorer

    cfg = cfg or SchedulerConfig()
    a = float(alpha) if alpha is not None else float(getattr(cfg, "_alpha", 0.5))
    scorer = WorldActionScorer(perception_model=model, alpha=a)
    sched = LyapunovScheduler(cfg, scorer, request_vehicle_id=int(contexts[0].ego.actor_id))
    rows, summary = sched.run_offline(contexts, realized_uncertainty_fn=realized_uncertainty_fn)
    return sched, rows, summary


def run_and_write(
    contexts: List["object"], *, model=None, cfg=None, out_dir: Path,
    realized_uncertainty_fn=None,
) -> Tuple[List[Dict[str, float]], Dict[str, float], Dict[str, Path]]:
    """Run the scheduler over ``contexts`` and write the trace CSV/JSONL + summary.json. Returns paths."""
    import pandas as pd

    _, rows, summary = run_scheduler(
        contexts, model=model, cfg=cfg, alpha=getattr(cfg, "_alpha", None) if cfg else None,
        realized_uncertainty_fn=realized_uncertainty_fn,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lyapunov_trace.csv"
    jsonl_path = out_dir / "lyapunov_trace.jsonl"
    summary_path = out_dir / "summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return rows, summary, {"csv": csv_path, "jsonl": jsonl_path, "summary": summary_path}


def main() -> int:
    args = parse_args()
    from car_dreamer.toolkit.wam import SchedulerConfig

    model = None
    if args.stage1_ckpt is not None:
        model = load_perception_model(args.stage1_ckpt, device=args.device)

    if args.contexts is not None:
        import torch

        contexts = torch.load(args.contexts, weights_only=False)
    else:
        contexts = build_synthetic_contexts(
            steps=args.steps, n_members=args.n_members, route_len=args.route_len,
            seed=args.seed, bev_size=args.bev_size,
        )

    cfg = SchedulerConfig(
        lam=args.lam, c0=args.c0, budget_bandwidth=args.budget, F_max_slots=args.f_max,
        n_min_slots=args.n_min, ts_seconds=args.ts_seconds,
    )
    cfg._alpha = args.alpha  # passed through to the scorer
    rows, summary, paths = run_and_write(contexts, model=model, cfg=cfg, out_dir=args.out_dir)

    print(f"[lyapunov] steps={int(summary['steps'])} epochs={int(summary['epochs'])} replans={int(summary['replans'])}")
    print(f"  time_avg_uncertainty={summary['time_avg_uncertainty']:.4f}  U_c={summary['u_c']:.4f}")
    print(f"  time_avg_bandwidth(B̄)={summary['time_avg_bandwidth']:.4f}  budget(B̄_bgt)={args.budget:.4f}")
    print(f"  mean_backlog={summary['mean_backlog']:.3g}  Z(T)/T={summary['z_over_t']:.4g}")
    print(f"  wrote: {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
