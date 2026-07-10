#!/usr/bin/env python3
"""Train the WAM Dreamer world model on the seed replay (Dreamer redesign, P2).

Loads the per-episode Dreamer replay (``scripts/build_wam_dreamer_replay.py`` output), samples fixed-length
subsequences, and trains :class:`WAMWorldModel` (RSSM + reward/cont/BEV heads) by the world-model loss
(dyn/rep KL + reward symlog-MSE + cont BCE + BEV occupancy BCE). Actor-critic is P3.

Usage
-----
    python scripts/train_wam_dreamer.py --replay-dir data/wam_dreamer_replay_v3 \
        --ckpt-dir outputs/wam_dreamer_v3 --steps 4000 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from car_dreamer.toolkit.wam import RSSMConfig, WAMWorldModel, WorldModelConfig  # noqa: E402

_KEYS = ["obs_graph_embed", "bev_target", "obs_scalars", "actions", "rewards", "is_first", "is_terminal"]


def load_episodes(replay_dir: str) -> List[Dict[str, torch.Tensor]]:
    files = sorted(glob.glob(os.path.join(replay_dir, "episode_*.pt")))
    if not files:
        raise FileNotFoundError(f"no episode_*.pt under {replay_dir}")
    return [torch.load(f, weights_only=False) for f in files]


def sample_batch(episodes, batch_size: int, seq_len: int, rng: random.Random) -> Dict[str, torch.Tensor]:
    cols: Dict[str, list] = {k: [] for k in _KEYS}
    for _ in range(batch_size):
        ep = episodes[rng.randrange(len(episodes))]
        T = int(ep["rewards"].shape[0])
        if T <= seq_len:  # short episode: take all, pad by repeating the last step
            idx = list(range(T)) + [T - 1] * (seq_len - T)
        else:
            s = rng.randrange(0, T - seq_len + 1)
            idx = list(range(s, s + seq_len))
        idx_t = torch.tensor(idx, dtype=torch.long)
        for k in _KEYS:
            cols[k].append(ep[k][idx_t])
    batch = {k: torch.stack(cols[k], dim=0) for k in _KEYS}
    batch["is_first"] = batch["is_first"].float().clone()
    batch["is_first"][:, 0] = 1.0  # reset RSSM state at each sampled subsequence start
    batch["is_terminal"] = batch["is_terminal"].float()
    return batch


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", default="data/wam_dreamer_replay_v3")
    p.add_argument("--ckpt-dir", default="outputs/wam_dreamer_v3")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    # model dims
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--deter", type=int, default=512)
    p.add_argument("--stoch", type=int, default=32)
    p.add_argument("--classes", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--bev-pos-weight", type=float, default=50.0)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    episodes = load_episodes(args.replay_dir)
    dims = episodes[0].get("action_dims", [9, 4, 2])
    graph_dim = int(episodes[0]["obs_graph_embed"].shape[-1])
    scalar_dim = int(episodes[0]["obs_scalars"].shape[-1])
    bev_c, bev_h = episodes[0]["bev_target"].shape[1], episodes[0]["bev_target"].shape[2]
    total_T = sum(int(e["rewards"].shape[0]) for e in episodes)
    print(f"[data] {len(episodes)} episodes, {total_T} transitions; "
          f"graph_dim={graph_dim} scalar_dim={scalar_dim} bev={bev_c}x{bev_h} action_dims={dims}", flush=True)

    cfg = WorldModelConfig(
        graph_embed_dim=graph_dim, scalar_dim=scalar_dim, action_dims=tuple(int(d) for d in dims),
        embed_dim=args.embed_dim, bev_channels=int(bev_c), bev_size=int(bev_h), bev_pos_weight=args.bev_pos_weight,
        rssm=RSSMConfig(deter=args.deter, stoch=args.stoch, classes=args.classes, hidden=args.hidden),
    )
    wm = WAMWorldModel(cfg).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"[model] WAMWorldModel params={n_params/1e6:.2f}M feat_dim={cfg.rssm.feat_dim}", flush=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    running: Dict[str, float] = {}
    for step in range(1, args.steps + 1):
        batch = to_device(sample_batch(episodes, args.batch_size, args.seq_len, rng), device)
        total, metrics, _ = wm.loss(batch)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), args.grad_clip)
        opt.step()
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + v
        if step % args.log_interval == 0:
            avg = {k: running[k] / args.log_interval for k in running}
            running = {}
            print(f"[{step:5d}] total={avg['total']:7.3f} kl={avg['kl']:.3f}(dyn={avg['dyn']:.2f} "
                  f"rep={avg['rep']:.2f}) reward={avg['reward']:.3f} cont={avg['cont']:.3f} "
                  f"bev={avg['bev']:.3f} bev_iou={avg['bev_iou']:.3f}", flush=True)
        if step % args.ckpt_interval == 0 or step == args.steps:
            path = ckpt_dir / f"world_model_step{step}.pt"
            torch.save({"model": wm.state_dict(), "cfg": cfg, "step": step}, path)
            print(f"[ckpt] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
