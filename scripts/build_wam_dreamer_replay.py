#!/usr/bin/env python3
"""Build a Dreamer replay dataset from recorded WAM sub-action rollouts (Dreamer redesign, P1).

Reformats existing Stage-1 sub-action recordings (each sample = a decision point with the observed
object-graph ``window``, the raw ``source_window`` state, and the active ``(S,B,D,n)`` sub-action in
``metadata``) into per-episode Dreamer sequences of the **cooperative-perception MDP** (framing A --
per-sample fixed step, action ``(S,B,D)``, the ``(S,B,D,n)`` trunk is recovered at inference by
grouping consecutive same-action steps):

    obs (object-graph window + scalars [Z, ΣQ_m, ego_v, #cand, #notable]),
    action (s_idx, b_idx, d_idx),
    reward (-P2 cost-rate over the step, WorldActionScorer(U_φ) + action_cost_rate under the running queue),
    is_first / is_terminal.

Reward uses the *current* Stage-1 uncertainty (soft-gated) unchanged (the eq-12 fix is deferred). The
per-link Shannon rate is approximated by ``link_rate_bps * ratio / (1 + dist/d0)`` -- refine to match
the online ``_link_rate_bps`` later if needed (affects only the comm/queue reward terms).

Usage
-----
    python scripts/build_wam_dreamer_replay.py \
        --data-dir data/wam_stage1_chunk_v3 --out-dir data/wam_dreamer_replay_v3 \
        --stage1-ckpt outputs/wam_stage1_chunk_v3/stage1_step2000.pt --device cuda
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from car_dreamer.toolkit.wam import (  # noqa: E402
    ActionSpec,
    MDPConfig,
    RolloutContext,
    WorldActionScorer,
    encode_subaction,
    subaction_cost,
    subaction_from_metadata,
)
from car_dreamer.toolkit.wam.bev import BevSpec, rasterize_bev  # noqa: E402
from car_dreamer.toolkit.wam.graph import OBJECT, VEHICLE, GraphBuildSpec  # noqa: E402
from car_dreamer.toolkit.wam.heads import WAMPerceptionConfig, WAMPerceptionModel  # noqa: E402
from car_dreamer.toolkit.wam.lyapunov import LyapunovConfig, LyapunovState  # noqa: E402


def pool_graph(graph_net, graph, hidden: int, device) -> torch.Tensor:
    """Frozen graph -> pooled embedding ``concat(mean(vehicle nodes), mean(object nodes))`` [2*hidden]."""
    with torch.no_grad():
        H = graph_net(graph.to(device))
    def m(key):
        return H[key].mean(0) if (key in H and H[key].numel()) else torch.zeros(hidden, device=device)
    return torch.cat([m(VEHICLE), m(OBJECT)]).detach().cpu()


def load_stage1_model(ckpt_path: str, device: torch.device) -> WAMPerceptionModel:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    perc_cfg = ck["perception_config"]
    if isinstance(perc_cfg, dict):
        perc_cfg = WAMPerceptionConfig(**perc_cfg)
    model = WAMPerceptionModel(perc_cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def build(args) -> int:
    device = torch.device(args.device)
    model = load_stage1_model(args.stage1_ckpt, device)
    graph_net = model.graph_net
    hidden = int(model.config.hidden_dim)
    bev_spec_target = BevSpec(size=args.bev_target_size, range_m=args.bev_target_range)
    d0 = float(args.link_d0_m)
    base_bps = float(args.link_rate_bps)

    def link_rate_fn(member_id: int, dist_m: float, ratio: float) -> float:
        return base_bps * max(float(ratio), 1e-3) / (1.0 + float(dist_m) / d0)

    scorer = WorldActionScorer(
        perception_model=model, sigma_scale=args.sigma0, alpha=args.alpha,
        history_window=args.history_window, device=args.device,
    )
    spec = ActionSpec(max_members=args.max_members,
                      bandwidth_grid=tuple(args.bandwidth_grid), modalities=tuple(args.modalities),
                      use_duration=False, step_slots=args.step_slots)
    cfg = MDPConfig(lam=args.lam, c0=args.c0, budget_bandwidth=args.budget)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    if args.limit:
        files = files[: int(args.limit)]
    if not files:
        print(f"[error] no .pt files under {args.data_dir}")
        return 1

    # group by episode, keep (step, file)
    by_episode: Dict[int, List] = defaultdict(list)
    for f in files:
        d = torch.load(f, weights_only=False)
        m = d["metadata"]
        by_episode[int(m["episode_id"])].append((int(m["step"]), f, d))
    for ep in by_episode:
        by_episode[ep].sort(key=lambda t: t[0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_T = 0
    total_coop = 0
    all_rewards: List[float] = []
    for ep_id, items in sorted(by_episode.items()):
        lyap = LyapunovState(LyapunovConfig(lam=cfg.lam, c0=cfg.c0,
                                            budget_bandwidth=cfg.budget_bandwidth, ts_seconds=args.dt))
        obs_embeds, bev_targets, scalars, actions, rewards, is_first, is_terminal, cand_ids_all, breakdowns = (
            [], [], [], [], [], [], [], [], []
        )
        steps = []
        for i, (step, _f, d) in enumerate(items):
            sw = d["source_window"][-1]
            ego = sw["ego"]
            collaborators = tuple(sw["collaborators"])
            candidate_ids = [int(c.actor_id) for c in collaborators]
            ctx = RolloutContext(
                ego=ego, collaborators=collaborators, objects=tuple(sw["live_states"]),
                route_xy=tuple(sw["route_xy"]), notable_ids=tuple(sw.get("notable_ids", ())),
                link_rate_fn=link_rate_fn,
                graph_spec=GraphBuildSpec(route_waypoints=args.route_waypoints),
                bev_spec=BevSpec(size=args.bev_size),
                ego_v0=float(math.hypot(float(ego.vx), float(ego.vy))),
                past_route_xy=tuple(sw.get("past_route_xy", ())),
                dt_seconds=float(args.dt), sensor_period_steps=int(args.sensor_period_steps),
            )
            sub = subaction_from_metadata(
                d["metadata"].get("active_subaction"), candidate_ids, duration_slots=args.step_slots
            )
            reward, roll, br = subaction_cost(
                ctx, sub, scorer, cfg, z=lyap.z.value, link_backlogs=lyap.backlogs()
            )
            act = encode_subaction(sub, candidate_ids, spec)

            # frozen graph embedding (encoder input) + BEV occupancy target (decoder reconstruction)
            obs_embeds.append(pool_graph(graph_net, d["window"][-1], hidden, device))
            selected = set(int(x) for x in sub.selected)
            observed = [
                o for o in sw["live_states"]
                if bool(o.visible_to_ego) or (set(int(c) for c in o.visible_to_collaborators) & selected)
            ]
            bev_np = rasterize_bev(ctx.ego_pose, observed, route_xy=ctx.route_xy, spec=bev_spec_target)
            bev_targets.append(torch.from_numpy(bev_np))  # [C, H, W] uint8

            scalars.append([lyap.z.value, lyap.total_backlog(),
                            ctx.ego_v0, float(len(candidate_ids)), float(len(ctx.notable_ids))])
            actions.append(list(act))
            rewards.append(float(reward))
            is_first.append(i == 0)
            is_terminal.append(i == len(items) - 1)
            cand_ids_all.append(candidate_ids)
            breakdowns.append(br.as_dict())
            steps.append(step)

            all_rewards.append(float(reward))
            total_T += 1
            if not sub.is_local_only:
                total_coop += 1

            # advance the running queue over the step's slots (frame totals spread evenly -- approx)
            nslot = max(int(args.step_slots), 1)
            svc = {int(k): float(v) / nslot for k, v in roll.per_member_predicted_service_bits.items()}
            arr = {int(k): float(v) / nslot for k, v in roll.per_member_predicted_load_bits.items()}
            for _ in range(nslot):
                lyap.advance_slot(per_member_service_bits=svc, per_member_arrival_bits=arr,
                                  allocated_bandwidth=float(sub.total_bandwidth()),
                                  budget_bandwidth=cfg.budget_bandwidth)

        episode = {
            "episode_id": ep_id,
            "steps": steps,
            "obs_graph_embed": torch.stack(obs_embeds).float(),        # [T, 2*hidden]
            "bev_target": torch.stack(bev_targets).to(torch.uint8),    # [T, C, H, W]
            "obs_scalars": torch.tensor(scalars, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "is_first": torch.tensor(is_first, dtype=torch.bool),
            "is_terminal": torch.tensor(is_terminal, dtype=torch.bool),
            "candidate_ids": cand_ids_all,
            "cost_breakdown": breakdowns,
            "action_dims": list(spec.dims),
            "scalar_keys": ["z", "total_backlog", "ego_v", "num_candidates", "num_notable"],
        }
        torch.save(episode, out_dir / f"episode_{ep_id:04d}.pt")
        print(f"[ep {ep_id}] T={len(steps)} coop={sum(1 for a in actions if a[0] > 0)} "
              f"reward[mean={sum(rewards)/max(len(rewards),1):.4f}] -> episode_{ep_id:04d}.pt", flush=True)

    r = torch.tensor(all_rewards) if all_rewards else torch.zeros(1)
    print(f"\n[ok] {len(by_episode)} episodes, {total_T} transitions "
          f"({total_coop} cooperative, {100.0*total_coop/max(total_T,1):.1f}%) -> {out_dir}")
    print(f"     reward: mean={float(r.mean()):.4f} std={float(r.std()):.4f} "
          f"min={float(r.min()):.4f} max={float(r.max()):.4f}")
    (out_dir / "meta.txt").write_text(
        f"source={args.data_dir}\nckpt={args.stage1_ckpt}\nframing=A(use_duration=False,step_slots={args.step_slots})\n"
        f"episodes={len(by_episode)} transitions={total_T} cooperative={total_coop}\n"
        f"action_dims={list(spec.dims)} scalar_keys=[z,total_backlog,ego_v,num_candidates,num_notable]\n"
        f"lam={args.lam} c0={args.c0} budget={args.budget} sigma0={args.sigma0}\n"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/wam_stage1_chunk_v3")
    p.add_argument("--out-dir", default="data/wam_dreamer_replay_v3")
    p.add_argument("--stage1-ckpt", default="outputs/wam_stage1_chunk_v3/stage1_step2000.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="cap #source samples (smoke test); 0 = all")
    # MDP / reward
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--c0", type=float, default=0.5)
    p.add_argument("--budget", type=float, default=0.4)
    p.add_argument("--sigma0", type=float, default=4.0, help="U saturation scale (scorer sigma_scale)")
    p.add_argument("--alpha", type=float, default=0.5, help="U = alpha*U_mot + (1-alpha)*U_cov")
    p.add_argument("--history-window", type=int, default=4)
    p.add_argument("--step-slots", type=int, default=2, help="framing-A fixed per-step duration (slots)")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--sensor-period-steps", type=int, default=1)
    p.add_argument("--route-waypoints", type=int, default=6)
    p.add_argument("--bev-size", type=int, default=24, help="coverage raster size for the reward U^cov")
    p.add_argument("--bev-target-size", type=int, default=16, help="BEV occupancy decoder target size (mult of 8)")
    p.add_argument("--bev-target-range", type=float, default=30.0, help="BEV occupancy target half-extent (m)")
    # action space
    p.add_argument("--max-members", type=int, default=8)
    p.add_argument("--bandwidth-grid", type=float, nargs="+", default=[0.2, 0.5, 0.8, 1.0])
    p.add_argument("--modalities", nargs="+", default=["objlist", "bev"])
    # link model (Shannon approx)
    p.add_argument("--link-rate-bps", type=float, default=1.0e5)
    p.add_argument("--link-d0-m", type=float, default=50.0)
    args = p.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
