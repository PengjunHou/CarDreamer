# coop_gnn_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, Generator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import embodied
from .embodied import CoopGNNPolicy

# 你之前写好的模型（我假设你已经放在某个模块里）
# from .coop_gnn_policy import CoopGNNPolicy
# 这里为了完整性，你也可以直接把 CoopGNNPolicy 的代码放在同文件里

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("This agent requires torch-geometric. Please install torch-geometric.") from e


# --------------------------
# Helper: to torch
# --------------------------
def _to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    x = np.asarray(x)
    if x.dtype == np.bool_:
        # bool -> uint8/bool tensor ok
        return torch.from_numpy(x).to(device)
    return torch.from_numpy(x).to(device)


def _is_batched(x: np.ndarray) -> bool:
    return isinstance(x, np.ndarray) and x.ndim >= 1

# --------------------------
# Agent Config
# --------------------------
@dataclass
class CoopGNNAgentConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 100.0

    # action mode
    discrete: bool = False
    n_discrete_actions: int = 0  # if discrete=True, must set

    # model dims
    node_token_dim: int = 1033   # 你现在 token dim（按你新的 _make_token_from_feat 计算）
    temporal_hidden: int = 256
    gnn_hidden: int = 256
    gnn_layers: int = 2

    # exploration
    eps_greedy: float = 0.0      # discrete only
    action_noise_std: float = 0.0 # continuous only


# --------------------------
# The Agent
# --------------------------
class CoopGNNTorchAgent(embodied.Agent):
    """
    - policy(): 用 obs 中的图张量构 Data，前向得到 action
    - train(): 提供一个“可跑通的默认监督/BC训练”骨架
      * discrete: CE loss
      * continuous: MSE loss
      如果你的 RL 算法在别处（Dreamer/PPO），你可以把 train() 改成返回需要的 logits/mu/value 等。
    """

    def __init__(self, obs_space, act_space, step, config: CoopGNNAgentConfig):
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step
        self.config = config if isinstance(config, CoopGNNAgentConfig) else CoopGNNAgentConfig(**dict(config))

        self.device = torch.device(self.config.device)

        self.model = CoopGNNPolicy(
            node_token_dim=self.config.node_token_dim,
            temporal_hidden=self.config.temporal_hidden,
            gnn_hidden=self.config.gnn_hidden,
            gnn_layers=self.config.gnn_layers,
            n_discrete_actions=self.config.n_discrete_actions if self.config.discrete else 0,
            continuous=(not self.config.discrete),
        ).to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # 如果你未来要 policy_device/train_device 分离，这里预留
        self._policy_model = self.model
        self._policy_state_dict = None

    # --------------------------
    # dataset
    # --------------------------
    def dataset(self, generator_fn):
        # 这里保持最简单：直接返回 generator_fn
        return generator_fn

    # --------------------------
    # policy
    # --------------------------
    @torch.no_grad()
    def policy(self, obs: Dict[str, Any], state=None, mode="train"):
        """
        输入 obs（B 批量） -> 输出 action（同 batch）.
        state 在这个实现里不需要（图+GRU 在单步内聚合；如果你要跨步 RNN state，我也可以扩展）。
        """
        self.model.eval() if mode != "train" else self.model.eval()

        # 构 batch 中每个样本的 Data，然后逐个 forward（N/E 可变时最稳）
        # 如果你保证 N/E 固定，也可以拼成大图一次 forward；先给你稳版本
        x_seq = obs["x_seq"]       # [B,N,T,F] or [N,T,F]
        lengths = obs["lengths"]   # [B,N] or [N]
        edge_index = obs["edge_index"]  # [B,2,E] or [2,E]
        edge_attr = obs.get("edge_attr", None)  # optional
        ego_index = obs.get("ego_index", None)  # optional

        x_seq = _to_torch(x_seq, self.device).float()
        lengths = _to_torch(lengths, self.device).long()
        edge_index_t = _to_torch(edge_index, self.device).long()
        edge_attr_t = _to_torch(edge_attr, self.device).float() if edge_attr is not None else None
        ego_index_t = _to_torch(ego_index, self.device).long() if ego_index is not None else None

        # unify batch dimension
        if x_seq.dim() == 3:
            # [N,T,F] -> [1,N,T,F]
            x_seq = x_seq.unsqueeze(0)
            lengths = lengths.unsqueeze(0)
            if edge_index_t.dim() == 2:
                edge_index_t = edge_index_t.unsqueeze(0)
            if edge_attr_t is not None and edge_attr_t.dim() == 2:
                edge_attr_t = edge_attr_t.unsqueeze(0)
            if ego_index_t is None:
                ego_index_t = torch.zeros((1,), dtype=torch.long, device=self.device)

        B = x_seq.shape[0]
        acts = []

        for b in range(B):
            ei = edge_index_t[b] if edge_index_t.dim() == 3 else edge_index_t
            ea = edge_attr_t[b] if (edge_attr_t is not None and edge_attr_t.dim() == 3) else edge_attr_t
            ego_i = ego_index_t[b] if ego_index_t is not None else torch.tensor(0, device=self.device)

            data = Data(
                x_seq=x_seq[b],                # [N,T,F]
                lengths=lengths[b],            # [N]
                edge_index=ei,                 # [2,E]
                edge_attr=ea,                  # [E,A] or None
                ego_index=ego_i,
            )

            out = self.model(data)

            if self.config.discrete:
                logits = out["logits"]
                if mode == "train" and self.config.eps_greedy > 0:
                    if torch.rand(()) < self.config.eps_greedy:
                        a = torch.randint(0, self.config.n_discrete_actions, (1,), device=self.device)
                    else:
                        a = torch.argmax(logits).view(1)
                else:
                    a = torch.argmax(logits).view(1)
                acts.append(a)
            else:
                mu = out["mu"]  # [2]
                a = mu
                if mode == "train" and self.config.action_noise_std > 0:
                    a = a + torch.randn_like(a) * self.config.action_noise_std
                acts.append(a.unsqueeze(0))  # [1,2]

        if self.config.discrete:
            act = torch.cat(acts, dim=0)  # [B]
            act_np = act.detach().cpu().numpy().astype(np.int64)
        else:
            act = torch.cat(acts, dim=0)  # [B,2]
            act_np = act.detach().cpu().numpy().astype(np.float32)

        # outs 的格式：embodied 通常期望 dict
        outs = {"action": act_np}
        return outs, state

    # --------------------------
    # train
    # --------------------------
    def train(self, data: Dict[str, Any], state=None):
        """
        一个“可跑通”的默认训练：行为克隆/监督学习骨架。
        - 需要 data 中包含 'action' 作为标签。
        你以后接 PPO/Dreamer 时，train() 可以改成计算 policy loss/value loss/entropy 等。
        """
        self.model.train()

        assert "action" in data, "train() expects data['action'] as target action (BC-style)."

        x_seq = _to_torch(data["x_seq"], self.device).float()
        lengths = _to_torch(data["lengths"], self.device).long()
        edge_index = _to_torch(data["edge_index"], self.device).long()
        edge_attr = _to_torch(data.get("edge_attr", None), self.device).float() if data.get("edge_attr", None) is not None else None
        ego_index = _to_torch(data.get("ego_index", None), self.device).long() if data.get("ego_index", None) is not None else None

        target = _to_torch(data["action"], self.device)

        # unify batch
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)
            lengths = lengths.unsqueeze(0)
            if edge_index.dim() == 2:
                edge_index = edge_index.unsqueeze(0)
            if edge_attr is not None and edge_attr.dim() == 2:
                edge_attr = edge_attr.unsqueeze(0)
            if ego_index is None:
                ego_index = torch.zeros((1,), dtype=torch.long, device=self.device)

        B = x_seq.shape[0]
        losses = []
        for b in range(B):
            data_b = Data(
                x_seq=x_seq[b],
                lengths=lengths[b],
                edge_index=edge_index[b] if edge_index.dim() == 3 else edge_index,
                edge_attr=edge_attr[b] if (edge_attr is not None and edge_attr.dim() == 3) else edge_attr,
                ego_index=(ego_index[b] if ego_index is not None else torch.tensor(0, device=self.device)),
            )
            out = self.model(data_b)

            if self.config.discrete:
                logits = out["logits"].view(1, -1)
                y = target[b].long().view(1)
                loss = F.cross_entropy(logits, y)
            else:
                mu = out["mu"].view(1, 2)
                y = target[b].float().view(1, 2)
                loss = F.mse_loss(mu, y)

            losses.append(loss)

        loss = torch.stack(losses).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip is not None and self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.opt.step()

        outs = {"loss": float(loss.detach().cpu().item())}
        mets = {"loss": float(loss.detach().cpu().item())}
        return outs, state, mets

    # --------------------------
    # report
    # --------------------------
    @torch.no_grad()
    def report(self, data: Dict[str, Any]):
        # 简单返回 loss（如果提供 action 标签）
        mets = {}
        if "action" in data:
            outs, _, mets = self.train({**data}, state=None)  # reuse train loss computation
            # train() 里会更新参数，不适合 report；所以这里简单给个占位
            mets = {"note": "report() not implemented fully; provide your eval metrics here."}
        return mets

    # --------------------------
    # save / load
    # --------------------------
    def save(self):
        return {
            "model": {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()},
            "opt": self.opt.state_dict(),
            "config": self.config.__dict__,
        }

    def load(self, data):
        state = {k: torch.from_numpy(np.asarray(v)).to(self.device) for k, v in data["model"].items()}
        self.model.load_state_dict(state, strict=False)
        if "opt" in data and data["opt"] is not None:
            self.opt.load_state_dict(data["opt"])

    # --------------------------
    # sync (optional multi-device)
    # --------------------------
    def sync(self):
        # 如果未来你要 train_device/policy_device 分离，在这里把训练参数拷到推理模型
        if self._policy_model is self.model:
            return
        self._policy_model.load_state_dict(self.model.state_dict(), strict=False)