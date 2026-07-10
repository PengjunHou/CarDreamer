#!/usr/bin/env python3
"""Train the WAM Dreamer actor-critic in imagination on a frozen world model (Dreamer redesign, P3).

Loads the P2 world model + the seed replay, derives imagination start states from replay posteriors, and
trains the actor (REINFORCE on λ-returns) + critic. Ends with an offline sanity eval comparing the
imagined discounted return of the trained actor vs a random policy vs an always-local policy from the same
held-out start states -- a learned cooperation policy should beat both.

Usage
-----
    python scripts/train_wam_dreamer_ac.py --world-model outputs/wam_dreamer_v3/world_model_step3000.pt \
        --replay-dir data/wam_dreamer_replay_v3 --ckpt-dir outputs/wam_dreamer_v3 --steps 3000 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from car_dreamer.toolkit.wam import ActorCriticConfig, WAMActorCritic, WAMWorldModel  # noqa: E402

_KEYS = ["obs_graph_embed", "obs_scalars", "actions", "rewards", "is_first", "is_terminal"]


def load_episodes(replay_dir: str) -> List[Dict[str, torch.Tensor]]:
    files = sorted(glob.glob(os.path.join(replay_dir, "episode_*.pt")))
    if not files:
        raise FileNotFoundError(f"no episode_*.pt under {replay_dir}")
    return [torch.load(f, weights_only=False) for f in files]


def sample_batch(episodes, batch_size, seq_len, rng) -> Dict[str, torch.Tensor]:
    cols: Dict[str, list] = {k: [] for k in _KEYS}
    for _ in range(batch_size):
        ep = episodes[rng.randrange(len(episodes))]
        T = int(ep["rewards"].shape[0])
        idx = (list(range(T)) + [T - 1] * (seq_len - T)) if T <= seq_len else \
            list(range(s := rng.randrange(0, T - seq_len + 1), s + seq_len))
        it = torch.tensor(idx, dtype=torch.long)
        for k in _KEYS:
            cols[k].append(ep[k][it])
    batch = {k: torch.stack(cols[k], 0) for k in _KEYS}
    batch["is_first"] = batch["is_first"].float().clone()
    batch["is_first"][:, 0] = 1.0
    return batch


def load_world_model(path: str, device) -> WAMWorldModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    wm = WAMWorldModel(ck["cfg"])
    wm.load_state_dict(ck["model"])
    wm.to(device).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


@torch.no_grad()
def start_states(wm: WAMWorldModel, batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """Replay batch -> flattened posterior states [B*T, ...] to seed imagination."""
    b = {k: v.to(device) for k, v in batch.items()}
    embed = wm.encode(b["obs_graph_embed"], b["obs_scalars"])
    onehot = wm.action_onehot(b["actions"])
    prev = torch.cat([torch.zeros_like(onehot[:, :1]), onehot[:, :-1]], dim=1)
    post, _ = wm.rssm.observe(embed, prev, b["is_first"].float())
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in post.items()}


@torch.no_grad()
def imagined_return(ac: WAMActorCritic, start: Dict[str, torch.Tensor], policy: Callable) -> float:
    """Mean discounted imagined return of ``policy`` (feat -> one-hot action) from ``start``."""
    from car_dreamer.toolkit.wam.wam_world_model import symexp
    state = {k: v.detach() for k, v in start.items()}
    total = torch.zeros(next(iter(start.values())).shape[0], device=ac.device)
    disc = torch.ones_like(total)
    for _ in range(ac.cfg.imag_horizon):
        feat = ac.wm.get_feat(state)
        action = policy(feat)
        state = ac.wm.rssm.img_step(state, action)
        f2 = ac.wm.get_feat(state)
        r = symexp(ac.wm.reward_head(f2).squeeze(-1))
        total = total + disc * r
        disc = disc * ac.cfg.gamma * torch.sigmoid(ac.wm.cont_head(f2).squeeze(-1))
    return float(total.mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-model", default="outputs/wam_dreamer_v3/world_model_step3000.pt")
    p.add_argument("--replay-dir", default="data/wam_dreamer_replay_v3")
    p.add_argument("--ckpt-dir", default="outputs/wam_dreamer_v3")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--imag-horizon", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=500)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    wm = load_world_model(args.world_model, device)
    episodes = load_episodes(args.replay_dir)
    action_dims = tuple(int(d) for d in episodes[0].get("action_dims", [9, 4, 2]))
    cfg = ActorCriticConfig(feat_dim=wm.cfg.rssm.feat_dim, action_dims=action_dims,
                            imag_horizon=args.imag_horizon)
    ac = WAMActorCritic(wm, cfg, device=device)
    print(f"[data] {len(episodes)} episodes; feat_dim={cfg.feat_dim} action_dims={action_dims} "
          f"actor_params={sum(p.numel() for p in ac.actor.parameters())/1e3:.1f}K", flush=True)

    run: Dict[str, float] = {}
    for step in range(1, args.steps + 1):
        start = start_states(wm, sample_batch(episodes, args.batch_size, args.seq_len, rng), device)
        m = ac.train_step(start)
        for k, v in m.items():
            run[k] = run.get(k, 0.0) + v
        if step % args.log_interval == 0:
            avg = {k: run[k] / args.log_interval for k in run}
            run = {}
            print(f"[{step:5d}] actor={avg['actor_loss']:7.3f} critic={avg['critic_loss']:8.3f} "
                  f"return={avg['return_mean']:8.3f} reward={avg['reward_mean']:7.3f} "
                  f"ent={avg['entropy']:.3f} value={avg['value_mean']:8.3f}", flush=True)

    # ---- offline sanity eval: trained vs random vs always-local ----
    def local_policy(feat):
        b = feat.shape[0]
        parts = []
        for i, n in enumerate(action_dims):
            oh = torch.zeros(b, n, device=device)
            oh[:, 0] = 1.0  # index 0: local-only S, and first B/D bin
            parts.append(oh)
        return torch.cat(parts, dim=-1)

    def random_policy(feat):
        b = feat.shape[0]
        return torch.cat([
            torch.nn.functional.one_hot(torch.randint(0, n, (b,), device=device), n).float()
            for n in action_dims
        ], dim=-1)

    def trained_policy(feat):
        return ac.actor(feat).mode()

    eval_starts = start_states(wm, sample_batch(episodes, 64, args.seq_len, random.Random(999)), device)
    r_trained = imagined_return(ac, eval_starts, trained_policy)
    r_random = imagined_return(ac, eval_starts, random_policy)
    r_local = imagined_return(ac, eval_starts, local_policy)
    print(f"\n[eval] imagined discounted return (higher=better, reward is -P2):")
    print(f"       trained={r_trained:8.3f}  random={r_random:8.3f}  always_local={r_local:8.3f}")
    print(f"       trained vs random: {r_trained - r_random:+.3f}   vs local: {r_trained - r_local:+.3f}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"actor_critic_step{args.steps}.pt"
    torch.save({"actor_critic": ac.state_dict(), "cfg": cfg, "step": args.steps,
                "eval": {"trained": r_trained, "random": r_random, "local": r_local}}, path)
    print(f"[ckpt] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
