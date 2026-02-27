# coop_sac_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import embodied

# -------------------------
# Helpers
# -------------------------
def _to_torch(x, device):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device)
    x = np.asarray(x)
    return torch.from_numpy(x).to(device)

def _is_time_major(arr: np.ndarray, batch_size: int) -> bool:
    # Heuristic: if first dim equals sequence length (args.batch_length), and second equals batch_size
    # We can't access args here, so: if arr.ndim>=2 and arr.shape[1]==batch_size, treat as [L,B,...]
    return arr.ndim >= 2 and arr.shape[1] == batch_size and arr.shape[0] != batch_size

def _ensure_batch_time_first(seq: Dict[str, Any], batch_size: int) -> Dict[str, np.ndarray]:
    """
    Ensure outputs are [B, L, ...] numpy arrays.
    seq is what replay.dataset yields, then agent.dataset/Batcher may stack into [B,L,...] or [L,B,...].
    """
    out = {}
    for k, v in seq.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
            continue
        if v.ndim >= 2 and _is_time_major(v, batch_size):
            # [L,B,...] -> [B,L,...]
            out[k] = np.swapaxes(v, 0, 1)
        elif v.ndim >= 1 and v.shape[0] == batch_size:
            # already [B,...] or [B,L,...]
            out[k] = v
        else:
            # fallback: keep
            out[k] = v
    return out

def _soft_update(target: nn.Module, online: nn.Module, tau: float):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


# -------------------------
# Graph Encoder (GRU over tokens + SAGE)
# You can replace this with your existing CoopGNNPolicy encoder that outputs h_ego.
# -------------------------
class GRUTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_dim = hidden_dim

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x_seq: [N, Tmax, F], lengths: [N]
        lengths_cpu = lengths.detach().to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, hN = self.gru(packed)   # [1, N, H]
        return hN.squeeze(0)       # [N, H]


class SAGEGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        return x


class GraphEncoder(nn.Module):
    """
    One graph -> h_ego
    Expects Data has:
      x_seq: [N, Tmax, F]
      lengths: [N]
      edge_index: [2, E]
      ego_index: scalar (int64)
    """
    def __init__(self, node_token_dim: int, temporal_hidden: int = 256, gnn_hidden: int = 256, gnn_layers: int = 2):
        super().__init__()
        self.temporal = GRUTemporalEncoder(node_token_dim, temporal_hidden)
        self.gnn = SAGEGNNEncoder(temporal_hidden, gnn_hidden, gnn_layers)
        self.out_dim = gnn_hidden

    def forward(self, data: Data) -> torch.Tensor:
        node_emb = self.temporal(data.x_seq, data.lengths)        # [N,Ht]
        node_emb = self.gnn(node_emb, data.edge_index)            # [N,Hg]
        if hasattr(data, "node_mask") and data.node_mask is not None:
            node_mask = data.node_mask.float().view(-1, 1)  # [N,1]
            node_emb = node_emb * node_mask           
        ego_idx = int(data.ego_index.item()) if hasattr(data, "ego_index") else 0
        return node_emb[ego_idx]                                  # [Hg]


# -------------------------
# SAC modules
# -------------------------
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

class Actor(nn.Module):
    def __init__(self, enc_dim: int, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(h)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(h)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        logp = dist.log_prob(z).sum(-1, keepdim=True)
        logp -= torch.log(torch.clamp(1 - a.pow(2), min=1e-6)).sum(-1, keepdim=True)
        return a, logp

    def mode(self, h: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(h)
        return torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, enc_dim: int, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(enc_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([h, a], dim=-1))


@dataclass
class CoopSACConfig:
    # devices
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # keys in replay batch
    action_key: str = "action"
    reward_key: str = "reward"
    is_last_key: str = "is_last"
    is_first_key: str = "is_first"

    # graph obs keys (current)
    x_seq_key: str = "x_seq"
    lengths_key: str = "lengths"
    edge_index_key: str = "edge_index"
    edge_attr_key: str = "edge_attr"      # optional
    ego_index_key: str = "ego_index"      # optional

    # model dims
    node_token_dim: int = 1034
    temporal_hidden: int = 256
    gnn_hidden: int = 256
    gnn_layers: int = 2

    # SAC hparams
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 100.0

    # entropy
    alpha: float = 0.2
    autotune_alpha: bool = True
    target_entropy: float = -2.0  # -act_dim

    # exploration at collection time
    # Driver/my_train uses mode="explore" for first phase; we’ll treat "explore" == stochastic sample
    # mode="train" == stochastic sample too; mode="eval" == deterministic


class CoopSACAgent(embodied.Agent):
    def __init__(self, obs_space, act_space, step, config: CoopSACConfig):
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step
        self.cfg = CoopSACConfig() #config if isinstance(config, CoopSACConfig) else CoopSACConfig(**dict(config))
        self.device = torch.device(self.cfg.device)

        # models
        self.encoder = GraphEncoder(
            node_token_dim=self.cfg.node_token_dim,
            temporal_hidden=self.cfg.temporal_hidden,
            gnn_hidden=self.cfg.gnn_hidden,
            gnn_layers=self.cfg.gnn_layers,
        ).to(self.device)

        enc_dim = self.encoder.out_dim
        self.actor = Actor(enc_dim, act_dim=2).to(self.device)
        self.q1 = Critic(enc_dim, act_dim=2).to(self.device)
        self.q2 = Critic(enc_dim, act_dim=2).to(self.device)
        self.tq1 = Critic(enc_dim, act_dim=2).to(self.device)
        self.tq2 = Critic(enc_dim, act_dim=2).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        # optimizers
        self.opt_critic = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # alpha
        self.autotune = self.cfg.autotune_alpha
        if self.autotune:
            self.log_alpha = torch.tensor(np.log(self.cfg.alpha), device=self.device, requires_grad=True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.cfg.alpha), device=self.device)
            self.opt_alpha = None

    # keep your pipeline: just return generator_fn (Batcher is outside or in wrapper)
    def dataset(self, generator_fn):
        return generator_fn()

    @torch.no_grad()
    def policy(self, obs: Dict[str, Any], state=None, mode="train"):
        """
        obs comes from env.step, batched over num_envs (Driver ensures that).
        We output dict with key action_key (default 'action'), shape [B,2] float32 in [-1,1].
        """
        self.encoder.eval()
        self.actor.eval()

        # Build per-env graphs and run one by one (safe with variable N/E).
        # Expect obs[*] are batched arrays with leading dim B.
        B = len(obs[self.cfg.is_first_key])

        actions = []
        for b in range(B):
            data = self._build_graph_from_obs_at_t(obs, b)
            h = self.encoder(data).unsqueeze(0)  # [1,H]

            if mode == "eval":
                a = self.actor.mode(h)           # deterministic
            else:
                a, _ = self.actor.sample(h)      # stochastic

            actions.append(a)

        act = torch.cat(actions, dim=0).cpu().numpy().astype(np.float32)
        return {self.cfg.action_key: act}, state

    def train(self, data: Dict[str, Any], state=None):
        """
        data can be:
        - sequence without explicit batch dim: reward [L], x_seq [L,N,T,F], action [L,2]
        - or with batch dim: reward [B,L], x_seq [B,L,N,T,F], action [B,L,2]
        We'll normalize to [B,L,...], then sample one t per batch element and do SAC update.
        """
        # ----------------------------
        # 0) Helper: normalize to [B,L,...]
        # ----------------------------
        def _as_np(x):
            return x if isinstance(x, np.ndarray) else np.asarray(x)

        def _ensure_BL(x: np.ndarray) -> np.ndarray:
            x = _as_np(x)
            # scalar -> [1,1]
            if x.ndim == 0:
                return x.reshape(1, 1)
            # [L] -> [1,L]
            if x.ndim == 1:
                return x[None, :]
            # already has >=2 dims, assume first two are [B,L] or [L,B]? we do NOT guess here.
            return x

        def _ensure_BL_from_data(d: Dict[str, Any]) -> Dict[str, Any]:
            """
            Make sure sequence keys become [B,L,...].
            We rely on reward to infer L and whether batch dim exists.
            """
            out = dict(d)

            rew0 = _as_np(out[self.cfg.reward_key])
            # Case A: reward [L] (no batch dim)
            no_batch = (rew0.ndim == 1)

            # ---- reward, is_last, is_first, is_terminal: ensure [B,L] or [B,L,1] later
            out[self.cfg.reward_key] = _ensure_BL(rew0)  # [B,L]
            out[self.cfg.is_last_key] = _ensure_BL(_as_np(out[self.cfg.is_last_key])).astype(np.bool_)  # [B,L]
            if "is_first" in out:
                out["is_first"] = _ensure_BL(_as_np(out["is_first"])).astype(np.bool_)
            if "is_terminal" in out:
                out["is_terminal"] = _ensure_BL(_as_np(out["is_terminal"])).astype(np.bool_)

            # ---- action
            act = _as_np(out[self.cfg.action_key])
            if no_batch:
                # action could be [L] (discrete) or [L, A] (continuous)
                out[self.cfg.action_key] = act[None, ...]  # -> [1,L] or [1,L,A]
            else:
                out[self.cfg.action_key] = act

            # ---- graph tensors
            # x_seq: [L,N,T,F] -> [1,L,N,T,F]
            x_seq = _as_np(out["x_seq"])
            if no_batch:
                if x_seq.ndim != 4:
                    raise ValueError(f"Expected x_seq [L,N,T,F] when no batch, got {x_seq.shape}")
                out["x_seq"] = x_seq[None, ...]
            else:
                out["x_seq"] = x_seq

            lengths = _as_np(out["lengths"])
            # lengths: [L,N] -> [1,L,N]
            if no_batch:
                if lengths.ndim != 2:
                    raise ValueError(f"Expected lengths [L,N] when no batch, got {lengths.shape}")
                out["lengths"] = lengths[None, ...]
            else:
                out["lengths"] = lengths

            # edge_index: usually [L,2,E] -> [1,L,2,E]
            edge_index = _as_np(out["edge_index"])
            if no_batch:
                if edge_index.ndim != 3:
                    raise ValueError(f"Expected edge_index [L,2,E] when no batch, got {edge_index.shape}")
                out["edge_index"] = edge_index[None, ...]
            else:
                out["edge_index"] = edge_index

            # edge_attr: [L,E,D] -> [1,L,E,D]
            edge_attr = _as_np(out["edge_attr"])
            if no_batch:
                if edge_attr.ndim != 3:
                    raise ValueError(f"Expected edge_attr [L,E,D] when no batch, got {edge_attr.shape}")
                out["edge_attr"] = edge_attr[None, ...]
            else:
                out["edge_attr"] = edge_attr

            # node_ids: [L,N] -> [1,L,N]
            node_ids = _as_np(out["node_ids"])
            if no_batch:
                if node_ids.ndim != 2:
                    raise ValueError(f"Expected node_ids [L,N] when no batch, got {node_ids.shape}")
                out["node_ids"] = node_ids[None, ...]
            else:
                out["node_ids"] = node_ids

            # node_mask: [L,N] -> [1,L,N]
            if "node_mask" in out:
                node_mask = _as_np(out["node_mask"])
                if no_batch:
                    if node_mask.ndim != 2:
                        raise ValueError(f"Expected node_mask [L,N] when no batch, got {node_mask.shape}")
                    out["node_mask"] = node_mask[None, ...]
                else:
                    out["node_mask"] = node_mask

            # ego_index: [L] -> [1,L]  OR scalar -> [1,1]
            if "ego_index" in out:
                out["ego_index"] = _ensure_BL(_as_np(out["ego_index"])).astype(np.int64)

            return out
        
        self.encoder.train()
        self.actor.train()
        self.q1.train()
        self.q2.train()

        # ----------------------------
        # 1) normalize
        # ----------------------------
        seq = _ensure_BL_from_data(data)

        reward = _as_np(seq[self.cfg.reward_key]).astype(np.float32)   # [B,L]
        is_last = _as_np(seq[self.cfg.is_last_key]).astype(np.bool_)   # [B,L]
        action = _as_np(seq[self.cfg.action_key])                      # [B,L,2] or [B,L] (discrete)

        B, L = reward.shape[0], reward.shape[1]
        if L < 2:
            return {}, state, {}

        # Make reward [B,L,1]
        reward = reward[..., None]  # [B,L,1]

        # Ensure action float for continuous SAC; if discrete, this will likely break later (by design)
        if action.ndim == 2:
            # [B,L] (discrete) -> [B,L,1] placeholder (you may later switch to discrete SAC/QL)
            action = action[..., None]
        action = action.astype(np.float32)

        # sample t in [0, L-2]
        ts = np.random.randint(0, L - 1, size=(B,), dtype=np.int64)

        # ----------------------------
        # 2) build transitions (loop over B; graphs are variable objects)
        # ----------------------------
        hs, nhs, a_t, r_t, disc_t = [], [], [], [], []

        for b in range(B):
            t = int(ts[b])

            done = bool(is_last[b, t])
            disc = 0.0 if done else 1.0

            g_t = self._build_graph_from_seq_at_bt(seq, b, t)
            g_tp1 = self._build_graph_from_seq_at_bt(seq, b, t + 1)

            h = self.encoder(g_t)  # [H]
            with torch.no_grad():
                nh = self.encoder(g_tp1)

            hs.append(h)
            nhs.append(nh)
            a_t.append(action[b, t])
            r_t.append(reward[b, t])
            disc_t.append([disc])

        h = torch.stack(hs, dim=0).to(self.device)     # [B,H]
        nh = torch.stack(nhs, dim=0).to(self.device)   # [B,H]
        a = _to_torch(np.asarray(a_t, np.float32), self.device).float()         # [B,act_dim]
        r = _to_torch(np.asarray(r_t, np.float32), self.device).float()         # [B,1]
        disc = _to_torch(np.asarray(disc_t, np.float32), self.device).float()   # [B,1]

        # ----------------------------
        # 3) SAC update
        # ----------------------------
        self.encoder.train()
        self.actor.train()
        self.q1.train()
        self.q2.train()

        with torch.no_grad():
            na, nlogp = self.actor.sample(nh)
            tq1 = self.tq1(nh, na)
            tq2 = self.tq2(nh, na)
            tmin = torch.min(tq1, tq2)
            alpha = self.log_alpha.exp()
            target_q = r + disc * self.cfg.gamma * (tmin - alpha * nlogp)

        q1 = self.q1(h, a)
        q2 = self.q2(h, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()),
                self.cfg.grad_clip,
            )
        self.opt_critic.step()

        # ---- Actor update (freeze encoder for stability)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        a_pi, logp_pi = self.actor.sample(h.detach())
        q1_pi = self.q1(h.detach(), a_pi)
        q2_pi = self.q2(h.detach(), a_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * logp_pi - min_q_pi).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.opt_actor.step()

        for p in self.encoder.parameters():
            p.requires_grad_(True)

        # ---- Alpha autotune
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.cfg.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()

        # ---- Target update
        _soft_update(self.tq1, self.q1, self.cfg.tau)
        _soft_update(self.tq2, self.q2, self.cfg.tau)

        mets = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": float(self.log_alpha.exp().detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()) if self.autotune else 0.0,
            "q1_mean": float(q1.detach().mean().cpu().item()),
            "q2_mean": float(q2.detach().mean().cpu().item()),
        }
        return {}, state, mets

    @torch.no_grad()
    def report(self, data):
        # 你可以后续加 eval metrics；先返回空避免影响 logger
        return {}

    def save(self):
        return {
            "encoder": {k: v.detach().cpu().numpy() for k, v in self.encoder.state_dict().items()},
            "actor": {k: v.detach().cpu().numpy() for k, v in self.actor.state_dict().items()},
            "q1": {k: v.detach().cpu().numpy() for k, v in self.q1.state_dict().items()},
            "q2": {k: v.detach().cpu().numpy() for k, v in self.q2.state_dict().items()},
            "tq1": {k: v.detach().cpu().numpy() for k, v in self.tq1.state_dict().items()},
            "tq2": {k: v.detach().cpu().numpy() for k, v in self.tq2.state_dict().items()},
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict() if self.opt_alpha is not None else None,
            "cfg": self.cfg.__dict__,
        }

    def load(self, data):
        def _load(module, sd_np):
            sd = {k: torch.from_numpy(np.asarray(v)).to(self.device) for k, v in sd_np.items()}
            module.load_state_dict(sd, strict=False)

        _load(self.encoder, data["encoder"])
        _load(self.actor, data["actor"])
        _load(self.q1, data["q1"])
        _load(self.q2, data["q2"])
        _load(self.tq1, data["tq1"])
        _load(self.tq2, data["tq2"])
        if "opt_actor" in data:
            self.opt_actor.load_state_dict(data["opt_actor"])
        if "opt_critic" in data:
            self.opt_critic.load_state_dict(data["opt_critic"])
        if self.opt_alpha is not None and data.get("opt_alpha", None) is not None:
            self.opt_alpha.load_state_dict(data["opt_alpha"])

    def sync(self):
        # 单设备不需要；如需 train/policy 分设备再扩展
        return

    # -------------------------
    # graph builders
    # -------------------------
    def _build_graph_from_obs_at_t(self, obs: Dict[str, Any], b: int) -> Data:
        """
        Build one PyG Data from env obs at current step for env index b.
        Expects obs keys are batched at leading dim B.
        """
        x_seq = _to_torch(obs[self.cfg.x_seq_key][b], self.device).float()           # [N,T,F]
        lengths = _to_torch(obs[self.cfg.lengths_key][b], self.device).long()        # [N]
        edge_index = _to_torch(obs[self.cfg.edge_index_key][b], self.device).long()  # [2,E] or [?,2,E] but expected [2,E]
        edge_attr = None
        if self.cfg.edge_attr_key in obs:
            ea = obs[self.cfg.edge_attr_key]
            if isinstance(ea, np.ndarray):
                edge_attr = _to_torch(ea[b], self.device).float()
        ego_index = None
        if self.cfg.ego_index_key in obs:
            ei = obs[self.cfg.ego_index_key]
            if isinstance(ei, np.ndarray):
                ego_index = _to_torch(ei[b], self.device).long()
        if ego_index is None:
            ego_index = torch.tensor(0, device=self.device, dtype=torch.long)
        node_mask = None
        if "node_mask" in obs:
            node_mask = _to_torch(obs["node_mask"][b], self.device).float()

        return Data(x_seq=x_seq, lengths=lengths, edge_index=edge_index, edge_attr=edge_attr, ego_index=ego_index, node_mask=node_mask)

    def _build_graph_from_seq_at_bt(self, seq: Dict[str, Any], b: int, t: int) -> Data:
        """
        seq arrays are [B,L,...]. Take (b,t) slice.
        """
        x_seq = _to_torch(seq[self.cfg.x_seq_key][b, t], self.device).float()             # [N,T,F]
        lengths = _to_torch(seq[self.cfg.lengths_key][b, t], self.device).long()          # [N]
        edge_index = _to_torch(seq[self.cfg.edge_index_key][b, t], self.device).long()    # [2,E]
        edge_attr = None
        if self.cfg.edge_attr_key in seq and isinstance(seq[self.cfg.edge_attr_key], np.ndarray):
            edge_attr = _to_torch(seq[self.cfg.edge_attr_key][b, t], self.device).float()
        ego_index = None
        if self.cfg.ego_index_key in seq and isinstance(seq[self.cfg.ego_index_key], np.ndarray):
            ego_index = _to_torch(seq[self.cfg.ego_index_key][b, t], self.device).long()
        if ego_index is None:
            ego_index = torch.tensor(0, device=self.device, dtype=torch.long)
        node_mask = None
        if "node_mask" in seq:
            node_mask = _to_torch(seq["node_mask"][b, t], self.device).float()  # [N]

        return Data(x_seq=x_seq, lengths=lengths, edge_index=edge_index, edge_attr=edge_attr, ego_index=ego_index, node_mask=node_mask)