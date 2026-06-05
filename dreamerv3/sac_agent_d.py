# sac_agent_d.py / coop_sac_agent.py  (DISCRETE ONE-HOT ACTION for embodied dict action spaces)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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


@torch.no_grad()
def _soft_update(target: nn.Module, online: nn.Module, tau: float):
    # target <- (1-tau)*target + tau*online
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.mul_(1.0 - tau).add_(op, alpha=tau)


def _onehot(indices: np.ndarray, depth: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], depth), dtype=np.float32)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


# -------------------------
# Graph Encoder (GRU over tokens + SAGE)
# -------------------------
class GRUTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_dim = hidden_dim

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, hN = self.gru(packed)
        return hN.squeeze(0)


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
    def __init__(self, node_token_dim: int, temporal_hidden: int = 256, gnn_hidden: int = 256, gnn_layers: int = 2):
        super().__init__()
        self.temporal = GRUTemporalEncoder(node_token_dim, temporal_hidden)
        self.gnn = SAGEGNNEncoder(temporal_hidden, gnn_hidden, gnn_layers)
        self.out_dim = gnn_hidden

    def forward(self, data: Data) -> torch.Tensor:
        node_emb = self.temporal(data.x_seq, data.lengths)        # [N,H]
        node_emb = self.gnn(node_emb, data.edge_index)            # [N,H]
        if hasattr(data, "node_mask") and data.node_mask is not None:
            node_emb = node_emb * data.node_mask.float().view(-1, 1)
        ego_idx = int(data.ego_index.item()) if hasattr(data, "ego_index") else 0
        return node_emb[ego_idx]


# -------------------------
# Discrete SAC modules
# -------------------------
class DiscreteActor(nn.Module):
    def __init__(self, enc_dim: int, act_n: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logits = nn.Linear(hidden, act_n)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.logits(self.net(h))  # [B,A]

    def sample(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(h)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()                           # [B]
        logp = dist.log_prob(a).unsqueeze(-1)       # [B,1]
        return a, logp

    def mode(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.forward(h)
        return torch.argmax(logits, dim=-1)         # [B]


class DiscreteCritic(nn.Module):
    def __init__(self, enc_dim: int, act_n: int, hidden: int = 256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_n),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.q(h)  # [B,A]


@dataclass
class CoopSACConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    action_key: str = "action"
    reward_key: str = "reward"
    is_last_key: str = "is_last"
    is_first_key: str = "is_first"

    x_seq_key: str = "x_seq"
    lengths_key: str = "lengths"
    edge_index_key: str = "edge_index"
    edge_attr_key: str = "edge_attr"
    ego_index_key: str = "ego_index"

    node_token_dim: int = 3082
    temporal_hidden: int = 256
    gnn_hidden: int = 256
    gnn_layers: int = 2

    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 100.0

    alpha: float = 0.2
    autotune_alpha: bool = True

    # ---- explore behavior (only used in policy when mode=="explore")
    explore_eps: float = 0.3  # epsilon-greedy: with eps choose random action
    explore_uniform: bool = False  # if True: explore always uniform random


class CoopSACAgent(embodied.Agent):
    def __init__(self, obs_space, act_space, step, config: CoopSACConfig):
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step

        # 保持你的外部适配：不改签名
        self.cfg = CoopSACConfig()
        self.device = torch.device(self.cfg.device)

        # ---- Detect action space type
        self._act_is_dict = isinstance(act_space, dict)

        if self._act_is_dict:
            sub = act_space.get(self.cfg.action_key, None)
            if sub is None:
                raise ValueError(
                    f"act_space dict missing key '{self.cfg.action_key}', got keys {list(act_space.keys())}"
                )
            shape = getattr(sub, "shape", None)
            if not shape or len(shape) != 1 or int(shape[0]) <= 0:
                raise ValueError(f"Expected act_space['{self.cfg.action_key}'] to be vector shape (A,), got {shape}")

            self.act_n = int(shape[0])
            self._output_onehot = True
            print(f"[CoopSACAgent] Detected dict action space with key '{self.cfg.action_key}' and shape {shape}, using one-hot output with act_n={self.act_n}")
        else:
            # fallback: gym Discrete(n)
            self.act_n = int(getattr(act_space, "n", 0))
            if self.act_n <= 0:
                raise ValueError(f"Discrete SAC requires Discrete(n) or embodied dict space, got {act_space}")
            self._output_onehot = False

        self.entropy_target = torch.tensor(float(np.log(self.act_n)), device=self.device)

        # ---- models
        self.encoder = GraphEncoder(
            node_token_dim=self.cfg.node_token_dim,
            temporal_hidden=self.cfg.temporal_hidden,
            gnn_hidden=self.cfg.gnn_hidden,
            gnn_layers=self.cfg.gnn_layers,
        ).to(self.device)

        enc_dim = self.encoder.out_dim
        self.actor = DiscreteActor(enc_dim, act_n=self.act_n).to(self.device)
        self.q1 = DiscreteCritic(enc_dim, act_n=self.act_n).to(self.device)
        self.q2 = DiscreteCritic(enc_dim, act_n=self.act_n).to(self.device)
        self.tq1 = DiscreteCritic(enc_dim, act_n=self.act_n).to(self.device)
        self.tq2 = DiscreteCritic(enc_dim, act_n=self.act_n).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        self.opt_critic = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        self.autotune = self.cfg.autotune_alpha
        if self.autotune:
            self.log_alpha = torch.tensor(np.log(self.cfg.alpha), device=self.device, requires_grad=True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.cfg.alpha), device=self.device)
            self.opt_alpha = None

    def dataset(self, generator_fn):
        return generator_fn()

    @torch.no_grad()
    def policy(self, obs: Dict[str, Any], state=None, mode="train"):
        """
        one-hot离散动作 + 根据mode选择 RL策略 or explore
        - eval: RL greedy
        - train: RL stochastic (sample)
        - explore: epsilon-greedy / uniform random
        """
        self.encoder.eval()
        self.actor.eval()

        B = len(obs[self.cfg.is_first_key])
        a_idx_list = []

        for b in range(B):
            data = self._build_graph_from_obs_at_t(obs, b)
            h = self.encoder(data).unsqueeze(0)  # [1,H]

            if mode == "eval":
                # RL greedy
                a = self.actor.mode(h)  # [1]
            # elif mode == "explore":
            #     # explore: uniform random OR epsilon-greedy
            #     if self.cfg.explore_uniform:
            #         a = torch.randint(low=0, high=self.act_n, size=(1,), device=h.device)
            #     else:
            #         # epsilon-greedy: eps random else policy sample
            #         if np.random.rand() < float(self.cfg.explore_eps):
            #             a = torch.randint(low=0, high=self.act_n, size=(1,), device=h.device)
            #         else:
            #             a, _ = self.actor.sample(h)
            else:
                # default "train": RL stochastic (recommended for SAC-style training)
                a, _ = self.actor.sample(h)

            a_idx_list.append(a)

        a_idx = torch.cat(a_idx_list, dim=0).cpu().numpy().astype(np.int64)  # [B]

        if self._output_onehot:
            a_out = _onehot(a_idx, self.act_n)              # [B,A] float32
            reset = np.zeros((B,), dtype=bool)              # [B]
            return {self.cfg.action_key: a_out, "reset": reset}, state
        else:
            a_out = a_idx[:, None]                          # [B,1] int64
            return {self.cfg.action_key: a_out}, state

    def train(self, data: Dict[str, Any], state=None):
        def _as_np(x):
            return x if isinstance(x, np.ndarray) else np.asarray(x)

        def _ensure_BL(x: np.ndarray) -> np.ndarray:
            x = _as_np(x)
            if x.ndim == 0:
                return x.reshape(1, 1)
            if x.ndim == 1:
                return x[None, :]
            return x

        def _normalize_action_to_index(act_arr: np.ndarray) -> np.ndarray:
            """
            action could be:
              - indices: [B,L] or [B,L,1]
              - onehot/prob: [B,L,A]
            Return indices [B,L] int64
            """
            act_arr = _as_np(act_arr)
            if act_arr.ndim == 3 and act_arr.shape[-1] == self.act_n:
                return np.argmax(act_arr, axis=-1).astype(np.int64)  # [B,L]
            if act_arr.ndim == 3 and act_arr.shape[-1] == 1:
                return act_arr.squeeze(-1).astype(np.int64)
            if act_arr.ndim == 2:
                return act_arr.astype(np.int64)
            return np.asarray(act_arr).squeeze().astype(np.int64)

        def _ensure_BL_from_data(d: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(d)
            rew0 = _as_np(out[self.cfg.reward_key])
            no_batch = (rew0.ndim == 1)

            out[self.cfg.reward_key] = _ensure_BL(rew0).astype(np.float32)
            out[self.cfg.is_last_key] = _ensure_BL(_as_np(out[self.cfg.is_last_key])).astype(np.bool_)
            if "is_first" in out:
                out["is_first"] = _ensure_BL(_as_np(out["is_first"])).astype(np.bool_)
            if "is_terminal" in out:
                out["is_terminal"] = _ensure_BL(_as_np(out["is_terminal"])).astype(np.bool_)

            act = out[self.cfg.action_key]
            if no_batch:
                act = _as_np(act)
                if act.ndim == 2 and act.shape[-1] == self.act_n:
                    act = act[None, ...]      # [1,L,A]
                else:
                    act = act.squeeze()
                    act = act[None, ...]      # [1,L]
            else:
                act = _as_np(act)
            out[self.cfg.action_key] = act

            x_seq = _as_np(out["x_seq"])
            if no_batch:
                if x_seq.ndim != 4:
                    raise ValueError(f"Expected x_seq [L,N,T,F] when no batch, got {x_seq.shape}")
                out["x_seq"] = x_seq[None, ...]
            else:
                out["x_seq"] = x_seq

            lengths = _as_np(out["lengths"])
            if no_batch:
                if lengths.ndim != 2:
                    raise ValueError(f"Expected lengths [L,N] when no batch, got {lengths.shape}")
                out["lengths"] = lengths[None, ...]
            else:
                out["lengths"] = lengths

            edge_index = _as_np(out["edge_index"])
            if no_batch:
                if edge_index.ndim != 3:
                    raise ValueError(f"Expected edge_index [L,2,E] when no batch, got {edge_index.shape}")
                out["edge_index"] = edge_index[None, ...]
            else:
                out["edge_index"] = edge_index

            if "edge_attr" in out:
                ea = _as_np(out["edge_attr"])
                out["edge_attr"] = ea[None, ...] if (no_batch and ea is not None) else ea

            if "node_mask" in out:
                nm = _as_np(out["node_mask"])
                out["node_mask"] = nm[None, ...] if no_batch else nm

            if "ego_index" in out:
                out["ego_index"] = _ensure_BL(_as_np(out["ego_index"])).astype(np.int64)

            return out

        self.encoder.train()
        self.actor.train()
        self.q1.train()
        self.q2.train()

        seq = _ensure_BL_from_data(data)

        reward = _as_np(seq[self.cfg.reward_key]).astype(np.float32)  # [B,L]
        is_last = _as_np(seq[self.cfg.is_last_key]).astype(np.bool_)  # [B,L]
        action_raw = seq[self.cfg.action_key]
        action = _normalize_action_to_index(action_raw)               # [B,L]

        B, L = reward.shape
        if L < 2:
            return {}, state, {}

        r_seq = reward[..., None]  # [B,L,1]
        ts = np.random.randint(0, L - 1, size=(B,), dtype=np.int64)

        hs, nhs, a_t, r_t, disc_t = [], [], [], [], []
        for b in range(B):
            t = int(ts[b])
            done = bool(is_last[b, t])
            disc = 0.0 if done else 1.0
            # Data(
            #             x_seq=x_seq,
            #             lengths=lengths,
            #             edge_index=edge_index,
            #             edge_attr=edge_attr,
            #             ego_index=ego_index,
            #             node_mask=node_mask,
            #         )
            g_t = self._build_graph_from_seq_at_bt(seq, b, t)
            g_tp1 = self._build_graph_from_seq_at_bt(seq, b, t + 1)
            
            # print(f"[Train] Graph at time t g_t shape: {g_t.x_seq.shape}, Graph at time t+1 g_tp1 shape: {g_tp1.x_seq.shape}")
            # print(f"[Train] lengths at time t: {g_t.lengths}, lengths at time t+1: {g_tp1.lengths}")
            # print(f"[Train] edge_index at time t shape: {g_t.edge_index.shape}, edge_index at time t+1 shape: {g_tp1.edge_index.shape}")
            # print(f"[Train] Action at time t: {action[b, t]}, Reward at time t: {reward[b, t]}, Done at time t: {done}")

            h = self.encoder(g_t)
            with torch.no_grad():
                nh = self.encoder(g_tp1)

            hs.append(h)
            nhs.append(nh)
            a_t.append(int(action[b, t]))
            r_t.append(r_seq[b, t])
            disc_t.append([disc])

        h = torch.stack(hs, dim=0).to(self.device)  # [B,H]
        nh = torch.stack(nhs, dim=0).to(self.device)
        a = _to_torch(np.asarray(a_t, np.int64), self.device).long()  # [B]
        r = _to_torch(np.asarray(r_t, np.float32), self.device).float().view(B, 1)
        disc = _to_torch(np.asarray(disc_t, np.float32), self.device).float().view(B, 1)

        # ---- Critic target
        with torch.no_grad():
            next_logits = self.actor(nh)                       # [B,A]
            next_logp_all = F.log_softmax(next_logits, dim=-1) # [B,A]
            next_p_all = next_logp_all.exp()

            tmin_all = torch.min(self.tq1(nh), self.tq2(nh))   # [B,A]
            alpha = self.log_alpha.exp()
            v_next = (next_p_all * (tmin_all - alpha * next_logp_all)).sum(dim=-1, keepdim=True)
            target_q = r + disc * self.cfg.gamma * v_next      # [B,1]

        q1_all = self.q1(h)
        q2_all = self.q2(h)
        q1_taken = q1_all.gather(1, a.view(-1, 1))
        q2_taken = q2_all.gather(1, a.view(-1, 1))
        critic_loss = F.mse_loss(q1_taken, target_q) + F.mse_loss(q2_taken, target_q)

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()),
                self.cfg.grad_clip,
            )
        self.opt_critic.step()

        # ---- Actor update (freeze encoder & critics)
        for p in self.encoder.parameters(): p.requires_grad_(False)
        for p in self.q1.parameters(): p.requires_grad_(False)
        for p in self.q2.parameters(): p.requires_grad_(False)

        logits = self.actor(h.detach())
        logp_all = F.log_softmax(logits, dim=-1)
        p_all = logp_all.exp()
        with torch.no_grad():
            min_q_all = torch.min(self.q1(h.detach()), self.q2(h.detach()))
        alpha = self.log_alpha.exp()
        actor_loss = (p_all * (alpha * logp_all - min_q_all)).sum(dim=-1).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.opt_actor.step()

        for p in self.encoder.parameters(): p.requires_grad_(True)
        for p in self.q1.parameters(): p.requires_grad_(True)
        for p in self.q2.parameters(): p.requires_grad_(True)

        # ---- Alpha autotune
        alpha_loss = torch.tensor(0.0, device=self.device)
        entropy = -(p_all * logp_all).sum(dim=-1, keepdim=True)
        if self.autotune:
            alpha_loss = -(self.log_alpha * (self.entropy_target - entropy.detach())).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()

        _soft_update(self.tq1, self.q1, self.cfg.tau)
        _soft_update(self.tq2, self.q2, self.cfg.tau)

        mets = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": float(self.log_alpha.exp().detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()) if self.autotune else 0.0,
            "entropy": float(entropy.detach().mean().cpu().item()),
            "q1_mean": float(q1_taken.detach().mean().cpu().item()),
            "q2_mean": float(q2_taken.detach().mean().cpu().item()),
        }
        return {}, state, mets

    @torch.no_grad()
    def report(self, data):
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
            "act_n": int(self.act_n),
            "output_onehot": bool(self._output_onehot),
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

        if "act_n" in data:
            self.act_n = int(data["act_n"])
            self.entropy_target = torch.tensor(float(np.log(self.act_n)), device=self.device)
        if "output_onehot" in data:
            self._output_onehot = bool(data["output_onehot"])

    def sync(self):
        return

    # -------------------------
    # graph builders (unchanged signatures)
    # -------------------------
    def _build_graph_from_obs_at_t(self, obs: Dict[str, Any], b: int) -> Data:
        x_seq = _to_torch(obs[self.cfg.x_seq_key][b], self.device).float()
        lengths = _to_torch(obs[self.cfg.lengths_key][b], self.device).long()
        edge_index = _to_torch(obs[self.cfg.edge_index_key][b], self.device).long()

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

        return Data(
            x_seq=x_seq,
            lengths=lengths,
            edge_index=edge_index,
            edge_attr=edge_attr,
            ego_index=ego_index,
            node_mask=node_mask,
        )

    def _build_graph_from_seq_at_bt(self, seq: Dict[str, Any], b: int, t: int) -> Data:
        x_seq = _to_torch(seq[self.cfg.x_seq_key][b, t], self.device).float()
        lengths = _to_torch(seq[self.cfg.lengths_key][b, t], self.device).long()
        edge_index = _to_torch(seq[self.cfg.edge_index_key][b, t], self.device).long()

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
            node_mask = _to_torch(seq["node_mask"][b, t], self.device).float()

        return Data(
            x_seq=x_seq,
            lengths=lengths,
            edge_index=edge_index,
            edge_attr=edge_attr,
            ego_index=ego_index,
            node_mask=node_mask,
        )