# coop_gnn_policy.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# torch-geometric (PyG)
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# =========================
# 1) GraphBuilder
# =========================

@dataclass
class GraphBuildConfig:
    window_s: float = 2.0          # 时间窗 (秒)
    Tmax: int = 20                 # 每个车辆最多使用多少条消息
    max_nodes: int = 8             # 图中最多多少个车辆节点（含ego），防止爆
    feat_dim_max: int = 1024       # 统一 padding 到这个维度
    include_sender_state: bool = True  # 是否附加 sender 的速度/yaw 等状态（推荐）
    star_graph: bool = True        # True=星形(ego<->others)，False=全连接


def _pad_feat(feat: np.ndarray, feat_dim_max: int) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    out = np.zeros((feat_dim_max,), dtype=np.float32)
    d = min(len(feat), feat_dim_max)
    if d > 0:
        out[:d] = feat[:d]
    return out


def _rel_pose_ego_frame(ego_tf, sender_tf) -> np.ndarray:
    """
    sender 相对 ego 的位姿（用 ego yaw 把世界坐标旋转到 ego 坐标系）
    输出: [dx_ego, dy_ego, dyaw]  (float32)
    """
    ex, ey = ego_tf.location.x, ego_tf.location.y
    sx, sy = sender_tf.location.x, sender_tf.location.y
    dxw, dyw = (sx - ex), (sy - ey)

    yaw = math.radians(float(ego_tf.rotation.yaw))
    c, s = math.cos(-yaw), math.sin(-yaw)  # world->ego rotation
    dx = c * dxw - s * dyw
    dy = s * dxw + c * dyw

    dyaw = float(sender_tf.rotation.yaw - ego_tf.rotation.yaw)
    # wrap to [-180, 180]
    while dyaw > 180:
        dyaw -= 360
    while dyaw < -180:
        dyaw += 360

    return np.array([dx, dy, dyaw], dtype=np.float32)


class VehicleNodeGraphBuilder:
    """
    从 env 的 comm_received[ego_id] (deque/list of msg) 构建“车辆节点图”：

    - 节点0: ego
    - 节点1..N-1: 过去 window_s 内给 ego 发送过消息的 sender（最多 max_nodes-1 个）
    - 每个 sender 节点的序列 token = 时间窗内该 sender 的所有 message（最多 Tmax 条）
    - token 特征包含:
        [feat_padded (feat_dim_max),
         feat_dim_ratio,
         rel_pose (dx,dy,dyaw),
         age_s,
         latency_s,
         distance_m,
         payload_kb,
         (可选) sender_speed, sender_yaw_rate]
    """

    def __init__(self, cfg: GraphBuildConfig):
        self.cfg = cfg

    @torch.no_grad()
    def build(
        self,
        *,
        ego_actor,                 # carla.Vehicle
        carla_world,               # carla.World
        ego_feat: np.ndarray,      # ego 当前特征 (D,)
        ego_feat_dim: int,         # eg. 1024
        msgs: Sequence[Any],       # env.comm_received[ego_id] 里的 msg
        t_step: int,
        dt: float,
        device: torch.device,
    ) -> Data:
        cfg = self.cfg
        t_now = float(t_step) * float(dt)

        ego_tf = ego_actor.get_transform()
        # ego token：用“伪消息”把 ego 自己当作一个节点（序列长度=1）
        ego_token = self._make_token_from_feat(
            feat=ego_feat,
            feat_dim=ego_feat_dim,
            rel_pose=np.zeros(3, np.float32),
            age_s=0.0,
            latency_s=0.0,
            distance_m=0.0,
            payload_bytes=0,
            sender_actor=ego_actor,
            ego_actor=ego_actor,
        )

        # 1) 选窗口内消息 + 按 sender 分组
        msgs_by_sender: Dict[int, List[Any]] = {}
        for m in msgs:
            # 用“到达时间”判定是否在窗口内
            t_deliver = float(m.deliver_step) * float(dt)
            if (t_now - t_deliver) <= cfg.window_s:
                sid = int(m.sender_id)
                msgs_by_sender.setdefault(sid, []).append(m)

        # 2) 选最多 max_nodes-1 个 sender（优先：最近到达 or 最近 created）
        sender_ids = list(msgs_by_sender.keys())

        # 排序：按该 sender 最新 deliver_step 降序（最新先）
        sender_ids.sort(
            key=lambda sid: max(int(mm.deliver_step) for mm in msgs_by_sender[sid]),
            reverse=True,
        )
        sender_ids = sender_ids[: max(cfg.max_nodes - 1, 0)]

        # 3) 构造每个 sender 的 token 序列（最多 Tmax），并记录长度
        node_ids: List[int] = [int(ego_actor.id)] + sender_ids
        N = len(node_ids)

        # 先把每个 node 的 token list 做成 python list，再 pad 成 tensor
        seq_tokens: List[np.ndarray] = []
        seq_lens: List[int] = []

        # node 0: ego
        seq_tokens.append(np.stack([ego_token], axis=0))  # [1, F]
        seq_lens.append(1)

        # other nodes
        for sid in sender_ids:
            ms = msgs_by_sender[sid]
            # 用 created_step 排序更符合物理发生顺序
            ms.sort(key=lambda mm: int(mm.created_step))
            if len(ms) > cfg.Tmax:
                ms = ms[-cfg.Tmax :]  # 取最近 Tmax 条

            sender_actor = carla_world.get_actor(sid)  # carla.Actor
            tokens = []
            for m in ms:
                # payload 必须含 feat/feat_dim
                payload = m.payload
                feat = payload.get("feat", None)
                feat_dim = int(payload.get("feat_dim", cfg.feat_dim_max))
                if feat is None:
                    continue

                sender_tf = sender_actor.get_transform()
                rel_pose = _rel_pose_ego_frame(ego_tf, sender_tf)

                age_s = float(t_now - float(m.created_step) * float(dt))
                latency_s = float(m.latency_s)
                distance_m = float(m.distance_m)
                payload_bytes = int(m.payload_bytes) if hasattr(m, "payload_bytes") else 0

                tok = self._make_token_from_feat(
                    feat=feat,
                    feat_dim=feat_dim,
                    rel_pose=rel_pose,
                    age_s=age_s,
                    latency_s=latency_s,
                    distance_m=distance_m,
                    payload_bytes=payload_bytes,
                    sender_actor=sender_actor,
                    ego_actor=ego_actor,
                )
                tokens.append(tok)

            if len(tokens) == 0:
                # 如果窗口内有消息但 payload 不完整，给一个空 token
                tokens = [self._make_empty_token()]

            tokens_np = np.stack(tokens, axis=0)  # [Li, F]
            seq_tokens.append(tokens_np)
            seq_lens.append(tokens_np.shape[0])

        # 4) pad 到 [N, Tmax, F]
        F = seq_tokens[0].shape[-1]
        Tmax = cfg.Tmax
        X = np.zeros((N, Tmax, F), dtype=np.float32)
        for i in range(N):
            li = min(seq_tokens[i].shape[0], Tmax)
            X[i, :li] = seq_tokens[i][:li]

        lengths = np.asarray(seq_lens, dtype=np.int64)
        # clip lengths 到 Tmax
        lengths = np.clip(lengths, 1, Tmax)

        # 5) 边：星形 ego<->others（最稳）
        edge_index = self._build_edges(N, star=cfg.star_graph)

        # edge_attr（可选）：这里给一个简单的 [dx,dy,dist]，也可以扩展
        edge_attr = self._build_edge_attr(ego_actor, carla_world, node_ids, edge_index)

        # 6) 打包成 PyG Data
        data = Data(
            x_seq=torch.from_numpy(X).to(device),                 # [N, Tmax, F]
            lengths=torch.from_numpy(lengths).to(device),         # [N]
            edge_index=edge_index.to(device),                     # [2, E]
            edge_attr=edge_attr.to(device),                       # [E, edge_dim]
            node_ids=torch.tensor(node_ids, dtype=torch.long, device=device),  # [N]
            ego_index=torch.tensor(0, dtype=torch.long, device=device),
        )
        return data

    def _make_empty_token(self) -> np.ndarray:
        # 与 _make_token_from_feat 的维度保持一致
        feat = np.zeros((self.cfg.feat_dim_max,), dtype=np.float32)
        # [feat, feat_dim_ratio, rel_pose(3), age, latency, dist, payload_kb, speed, yaw_rate]
        extra_dim = 1 + 3 + 4 + (2 if self.cfg.include_sender_state else 0)
        return np.concatenate([feat, np.zeros((extra_dim,), dtype=np.float32)], axis=0)

    def _make_token_from_feat(
        self,
        *,
        feat: np.ndarray,
        feat_dim: int,
        rel_pose: np.ndarray,
        age_s: float,
        latency_s: float,
        distance_m: float,
        payload_bytes: int,
        sender_actor,
        ego_actor,
    ) -> np.ndarray:
        cfg = self.cfg
        feat_pad = _pad_feat(feat, cfg.feat_dim_max)
        feat_dim_ratio = np.array([float(feat_dim) / float(cfg.feat_dim_max)], dtype=np.float32)

        payload_kb = np.array([float(payload_bytes) / 1024.0], dtype=np.float32)
        age = np.array([float(age_s)], dtype=np.float32)
        lat = np.array([float(latency_s)], dtype=np.float32)
        dist = np.array([float(distance_m)], dtype=np.float32)

        parts = [feat_pad, feat_dim_ratio, rel_pose.astype(np.float32), age, lat, dist, payload_kb]

        # if cfg.include_sender_state:
        #     # sender 速度（在 ego 坐标系下的 speed 也行，这里先用标量 speed）
        #     v = sender_actor.get_velocity()
        #     speed = float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
        #     # yaw_rate 需要角速度传感器，CARLA actor 没直接给，这里用 0 占位（你可替换为 IMU/gyro）
        #     yaw_rate = 0.0
        #     parts.append(np.array([speed, yaw_rate], dtype=np.float32))
            
        if cfg.include_sender_state:
            v = sender_actor.get_velocity()
            # ego yaw
            ego_tf = ego_actor.get_transform()
            yaw = math.radians(float(ego_tf.rotation.yaw))
            c, s = math.cos(-yaw), math.sin(-yaw)
            # 世界速度 -> ego 坐标系
            vx_e = c * v.x - s * v.y
            vy_e = s * v.x + c * v.y

            rel_vel = np.array([vx_e, vy_e], dtype=np.float32)
            parts.append(rel_vel)

        return np.concatenate(parts, axis=0).astype(np.float32)

    def _build_edges(self, N: int, star: bool) -> torch.Tensor:
        if N <= 1:
            return torch.zeros((2, 0), dtype=torch.long)

        if star:
            # 0 <-> i
            src = []
            dst = []
            for i in range(1, N):
                src += [0, i]
                dst += [i, 0]
            return torch.tensor([src, dst], dtype=torch.long)
        else:
            # fully connected (no self-loops), bidirectional
            src = []
            dst = []
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    src.append(i)
                    dst.append(j)
            return torch.tensor([src, dst], dtype=torch.long)

    def _build_edge_attr(self, ego_actor, carla_world, node_ids: List[int], edge_index: torch.Tensor) -> torch.Tensor:
        # edge_attr: [dx, dy, dist] in ego frame
        if edge_index.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)

        ego_tf = ego_actor.get_transform()

        # cache actors transforms
        tfs = {}
        for vid in node_ids:
            act = ego_actor if vid == int(ego_actor.id) else carla_world.get_actor(int(vid))
            tfs[vid] = act.get_transform()

        E = edge_index.shape[1]
        out = torch.zeros((E, 3), dtype=torch.float32)
        for e in range(E):
            u = int(edge_index[0, e].item())
            v = int(edge_index[1, e].item())
            vid_u = node_ids[u]
            vid_v = node_ids[v]

            rel = _rel_pose_ego_frame(ego_tf, tfs[vid_v])  # v 相对 ego
            dx, dy = float(rel[0]), float(rel[1])
            dist = float(math.sqrt(dx * dx + dy * dy))
            out[e, 0] = dx
            out[e, 1] = dy
            out[e, 2] = dist
        return out


# =========================
# 2) Temporal Encoder (GRU)
# =========================

class GRUTemporalEncoder(nn.Module):
    """
    输入: x_seq [N, Tmax, F], lengths [N]
    输出: node_emb [N, H]  (每个节点用 GRU 最后时刻 hidden)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_dim = hidden_dim

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # lengths 必须是 CPU tensor 给 pack_padded_sequence
        lengths_cpu = lengths.detach().to("cpu")
        packed = pack_padded_sequence(
            x_seq, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, hN = self.gru(packed)  # hN: [1, N, H]
        return hN.squeeze(0)      # [N, H]


# =========================
# 3) GNN Encoder (GraphSAGE)
# =========================

class SAGEGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
        return x


# =========================
# 4) Policy Head (continuous + discrete)
# =========================

class CoopGNNPolicy(nn.Module):
    """
    forward(data: PyG Data) -> dict with:
      - 'logits' for discrete
      - 'mu' for continuous (acc, steer)
    """
    def __init__(
        self,
        node_token_dim: int,     # F
        temporal_hidden: int = 256,
        gnn_hidden: int = 256,
        gnn_layers: int = 2,
        n_discrete_actions: int = 0,  # >0 means enable discrete head
        continuous: bool = True,      # enable continuous head
    ):
        super().__init__()
        self.temporal = GRUTemporalEncoder(node_token_dim, temporal_hidden)
        self.gnn = SAGEGNNEncoder(temporal_hidden, gnn_hidden, num_layers=gnn_layers)

        self.has_discrete = n_discrete_actions > 0
        self.has_continuous = bool(continuous)

        if self.has_discrete:
            self.discrete_head = nn.Sequential(
                nn.Linear(gnn_hidden, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, n_discrete_actions),
            )

        if self.has_continuous:
            # 输出 (acc, steer) 的均值；你也可以加 log_std 做 SAC
            self.cont_head = nn.Sequential(
                nn.Linear(gnn_hidden, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, 2),
            )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        # 1) per-node temporal encoding
        node_emb = self.temporal(data.x_seq, data.lengths)  # [N, Ht]

        # 2) GNN fusion
        node_emb = self.gnn(node_emb, data.edge_index)      # [N, Hg]

        # 3) ego embedding
        ego_idx = int(data.ego_index.item()) if hasattr(data, "ego_index") else 0
        h_ego = node_emb[ego_idx]                           # [Hg]

        out: Dict[str, torch.Tensor] = {"h_ego": h_ego}

        if self.has_discrete:
            out["logits"] = self.discrete_head(h_ego)

        if self.has_continuous:
            out["mu"] = self.cont_head(h_ego)  # acc, steer (raw)
        return out